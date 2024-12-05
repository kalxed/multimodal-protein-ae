import glob
import os
import os.path as osp

import numpy as np
import torch
import transformers
from Bio.PDB import MMCIFParser, PDBParser, Polypeptide
from sklearn.neighbors import radius_neighbors_graph
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.utils import negative_sampling

esm_id = "facebook/esm2_t30_150M_UR50D"
esm_model = transformers.AutoModelForMaskedLM.from_pretrained(esm_id)
esm_tokenizer = transformers.AutoTokenizer.from_pretrained(esm_id)

def tokenize_sequence(seq: str, padding=True) -> torch.Tensor:
    return esm_tokenizer(seq, padding=padding, return_tensors="pt")["input_ids"]

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


def structure_file_to_pointcloud(structure_file: str, desired_num_points: float=2048, parser=PDBParser(QUIET=True)) -> torch.FloatTensor:
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
