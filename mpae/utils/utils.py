import glob
import os
import os.path as osp

import numpy as np
import torch
from Bio.PDB import MMCIFParser, PDBParser, Polypeptide
from sklearn.neighbors import radius_neighbors_graph
from sklearn.preprocessing import LabelEncoder
from torch_geometric.data import Data
from torch_geometric.nn import VGAE
from torch_geometric.utils import negative_sampling

from mpae.nn import esm
from mpae.nn.pae import PointAutoencoder


def tokenize_sequence(seq: str, padding=True) -> torch.Tensor:
    return esm.esm_tokenizer(seq, padding=padding, return_tensors="pt")["input_ids"]

def structure_file_to_sequence(structure_path: str, parser) -> str:
    """
    Create a sequence from a protein structure by evaluating the residues.
    This function will ignore non-standard residues and return a string of amino acids.

        structure: A BioPython structure object
    """
    try:
        structure = parser.get_structure('protein', structure_path)
    except Exception as e:
        return None
    

    sequence = ""
    for residue in structure.get_residues():
        if "CA" in residue:  # Check if the residue is part of the main chain
            try:
                index_code = Polypeptide.three_to_index(residue.get_resname())
                aa_code = Polypeptide.index_to_one(index_code)
                sequence += aa_code
            except KeyError:
                # continue
                # opting to drop the protein since that is what the graphs do.
                return None
    return sequence


def structure_file_to_graph(structure_file: str, radius: float=6.0, parser=PDBParser(QUIET=True)) -> Data:
    amino_acids = list(range(20))
    label_encoder = LabelEncoder()
    label_encoder.fit(amino_acids)
    num_amino_acids = len(amino_acids)

    # Function to one-hot encode amino acid sequences
    def one_hot_encode_amino_acid(sequence):
        amino_acid_indices = label_encoder.transform(list(sequence))
        one_hot = np.zeros((len(sequence), num_amino_acids), dtype=np.float32)
        one_hot[np.arange(len(sequence)), amino_acid_indices] = 1
        return one_hot
    
    try:
        structure = parser.get_structure('protein', structure_file)
    except Exception:
        return None

    node_features = []
    coordinates = []
    sequence = []

    for residue in structure.get_residues():
        if 'CA' in residue:
            try:
                aa_code = Polypeptide.three_to_index(residue.get_resname())
            except KeyError:
                # unexpected amino acid
                return None
            sequence.append(aa_code)
            coordinates.append(residue['CA'].get_coord())
    coordinates = np.array(coordinates, dtype=np.float32)
    try:
        node_features = one_hot_encode_amino_acid(sequence)
    except IndexError:
        # unexpected amino acid
        return None
    x = torch.tensor(node_features, dtype=torch.float32)

    # Calculate edges based on distance
    edge_index = radius_neighbors_graph(coordinates, radius, mode='connectivity', include_self=False)
    edge_index = torch.tensor(np.array(edge_index.nonzero()), dtype=torch.long).contiguous()
    
    # Generate negative edge samples
    neg_edge_index = negative_sampling(
        edge_index=edge_index,
        num_nodes=x.size(0),
        num_neg_samples=edge_index.size(1) // 2
    )

    # Create a PyTorch Geometric Data object
    data = Data(x=x, edge_index=edge_index, neg_edge_index=neg_edge_index)

    return data


def structure_file_to_pointcloud(structure_file: str, desired_num_points: float=2048, parser=PDBParser(QUIET=True)) -> torch.Tensor:
    try:
        structure = parser.get_structure('protein', structure_file)
    except Exception:
        return None
    
    # cap the number of points to add
    coordinates = np.zeros((desired_num_points, 3))

    for i, atom in enumerate(structure.get_atoms()):
        if i == desired_num_points:
            break
        coordinates[i] = atom.get_coord()
    
    # Center and normalize the point cloud
    # I think that there is a better way to do this. oh well.
    coordinates = torch.tensor(coordinates, dtype=torch.float32)
    coordinates -= coordinates.mean(0)
    d = torch.sqrt((coordinates ** 2.0).sum(1))
    coordinates /= d.max()
    coordinates = torch.FloatTensor(coordinates).permute(1, 0)
    return coordinates

# Function for Z-score standardization
def z_score_standardization(tensor):
    std = tensor.std()
    if std == 0:
        return tensor
    mean = tensor.mean()
    standardized_tensor = (tensor - mean) / std
    return standardized_tensor

def process_encoded_graph(encoded_graph: torch.Tensor, fixed_size=640, feature_dim=10):
    num_nodes = encoded_graph.size(0)
    if num_nodes > fixed_size:
        # Randomly sample nodes to downsample
        indices = torch.randperm(num_nodes)[:fixed_size]
        processed_encoded_graph = encoded_graph[indices]
    elif num_nodes < fixed_size:
        # Pad with zeros if not enough nodes
        padding_size = fixed_size - num_nodes
        zero_padding = torch.zeros(padding_size, feature_dim, device=encoded_graph.device)
        processed_encoded_graph = torch.cat((encoded_graph, zero_padding), dim=0)
    else:
        processed_encoded_graph = encoded_graph
    return processed_encoded_graph

def fuse_modalities(graph: Data, tokenized_seq: torch.Tensor, pointcloud: torch.Tensor, vgae_model: VGAE, pae_model: PointAutoencoder, device: torch.device):
    # Encode sequence data using ESM
    vgae_model = vgae_model.to(device)
    vgae_model.eval()
    pae_model = pae_model.to(device)
    pae_model.eval()
    esm_model = esm.esm_model.to(device)
    esm_model.eval()
    with torch.no_grad():
        tokenized_seq = tokenized_seq.to(device)
        encoded_sequence = esm_model(tokenized_seq, output_hidden_states=True)["hidden_states"][-1][0, -1].to(device)
        encoded_sequence = z_score_standardization(encoded_sequence)

    # Encode graph data using VGAE
    with torch.no_grad():
        graph = graph.to(device)
        encoded_graph = vgae_model.encode(graph.x, graph.edge_index).to(device)
        encoded_graph = process_encoded_graph(encoded_graph)
        encoded_graph = torch.mean(encoded_graph, dim=1)
        encoded_graph = z_score_standardization(encoded_graph)

    # Encode point cloud data using PAE
    with torch.no_grad():
        pointcloud = pointcloud.to(device)
        encoded_point_cloud = pae_model.encode(pointcloud[None, :]).squeeze()
        encoded_point_cloud = z_score_standardization(encoded_point_cloud)

    fused_data = torch.cat((encoded_sequence.to(device), encoded_graph.to(device), encoded_point_cloud.to(device)), dim=0)
    
    return fused_data
