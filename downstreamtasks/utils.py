import os
import torch
import transformers
from Bio.PDB import PDBParser, Polypeptide
import numpy as np
from torch_geometric.data import Data
from torch_geometric.utils import negative_sampling
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings("ignore")
import random
random.seed(42)
import pickle
from math import sqrt
from scipy.stats import spearmanr, pearsonr
from lifelines.utils import concordance_index
import torch
import numpy as np
import transformers
import os
import h5py
from sklearn.neighbors import radius_neighbors_graph
from sklearn.preprocessing import LabelEncoder
from torch_geometric.nn import TopKPooling
from torch_geometric.utils import negative_sampling
from torch_geometric.data import Data
from Bio.PDB import PDBParser, PPBuilder, Polypeptide

# Check if GPU is available
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using GPU:", torch.cuda.get_device_name(0))
else:
    device = torch.device("cpu")
    print("No GPU available, using CPU.")

model_token = "facebook/esm2_t30_150M_UR50D"
tokenizer = transformers.AutoTokenizer.from_pretrained(model_token)

amino_acids = "ACDEFGHIKLMNPQRSTVWYX"
label_encoder = LabelEncoder()
label_encoder.fit(list(amino_acids))
num_amino_acids = len(amino_acids)

parser = PDBParser(QUIET=True)
ppb = PPBuilder()

import sys

sys.path.append(".")

from model.Concrete_Autoencoder import *
from model.ESM import *
from model.vgae import *
from model.PAE import *
from model.Attention import *

def load_models():  
    # Load pre-trained models
    vgae_model = torch.load(f"./data/models/VGAE.pt", map_location=device)
    pae_model = torch.load(f"./data/models/PAE.pt", map_location=device)
    model_token = "facebook/esm2_t30_150M_UR50D"
    esm_model = transformers.AutoModelForMaskedLM.from_pretrained(model_token)
    esm_model = esm_model.to(device)
    concrete_model = torch.load(f"./data/models/Concrete.pt", map_location=device)
    print("Pre-trained models loaded successfully.")
    return vgae_model, pae_model, esm_model, concrete_model

def pickle_dump(data_folder, mulmodal, sequence, graph, point_cloud):
    with open(f'{data_folder}/multimodal.pkl', 'wb') as f:
        pickle.dump(mulmodal, f)

    with open(f'{data_folder}/sequence.pkl', 'wb') as f:
        pickle.dump(sequence, f)

    with open(f'{data_folder}/graph.pkl', 'wb') as f:
        pickle.dump(graph, f)

    with open(f'{data_folder}/point_cloud.pkl', 'wb') as f:
        pickle.dump(point_cloud, f)
    print("Features saved successfully.")

def one_hot_encode_amino_acid(sequence):
    amino_acids = list(range(20))
    label_encoder = LabelEncoder()
    label_encoder.fit(amino_acids)
    num_amino_acids = len(amino_acids)
    amino_acid_indices = label_encoder.transform(list(sequence))
    one_hot = np.zeros((len(sequence), num_amino_acids), dtype=np.float32)
    one_hot[np.arange(len(sequence)), amino_acid_indices] = 1
    return one_hot

def read_pdb(pdb_path):
    model_token = "facebook/esm2_t30_150M_UR50D"
    esm_tokenizer = transformers.AutoTokenizer.from_pretrained(model_token)
    
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('protein', pdb_path)

    # Graph
    coordinates = []
    sequence = ""
    radius=6.0
    for residue in structure.get_residues():
        if 'CA' in residue:
            try:  
                aa_code = Polypeptide.three_to_index(residue.get_resname())
            except KeyError:
                # Skip this protein if it contains non-standard amino acids
                return None, None, None
            sequence += str(aa_code)
            coordinates.append(residue['CA'].get_coord())        
    coordinates = np.array(coordinates, dtype=np.float32)
    node_features = one_hot_encode_amino_acid(sequence)
    x = torch.tensor(node_features, dtype=torch.float32)
    edge_index = radius_neighbors_graph(coordinates, radius, mode='connectivity', include_self='auto')
    edge_index = edge_index.nonzero()
    edge_index = np.array(edge_index)
    edge_index = torch.from_numpy(edge_index).to(torch.long).contiguous()
    neg_edge_index = negative_sampling(
        edge_index= edge_index,
        num_nodes= x.size(0),
        num_neg_samples= edge_index.size(1)//2
    )
    graph = Data(x=x, edge_index=edge_index, neg_edge_index=neg_edge_index)

        # Point Cloud
    desired_num_points = 2048
    coordinates = np.zeros((desired_num_points, 3))
    for i, atom in enumerate(structure.get_atoms()):
        if i == desired_num_points:
            break
        coordinates[i] = atom.get_coord()
    # coordinates = np.array(coordinates, dtype=np.float32)
    # num_points = coordinates.shape[0]
    # if num_points < desired_num_points:
    #     padding = np.zeros((desired_num_points - num_points, 3), dtype=np.float32)
    #     coordinates = np.concatenate((coordinates, padding), axis=0)
    # elif num_points > desired_num_points:
    #     coordinates = coordinates[:desired_num_points, :]
    coordinates = torch.tensor(coordinates, dtype=torch.float32)
    coordinates -= coordinates.mean(0)
    d = np.sqrt((coordinates ** 2).sum(1))
    coordinates /= d.max()
    point_cloud = torch.FloatTensor(coordinates).permute(1, 0)

    # Sequence
    sequence = esm_tokenizer(sequence, return_tensors="pt", padding=True, truncation=True, max_length=2048)["input_ids"]

    return sequence, graph, point_cloud

def get_modalities(protein_path, ESM, VGAE, PAE, Fusion):
    # Check file_extension
    file_name, file_extension = os.path.splitext(protein_path)
    if file_extension == ".pdb":
        sequence, graph, point_cloud = read_pdb(protein_path)
    elif file_extension == ".hdf5":
        sequence, graph, point_cloud = read_hdf5(protein_path)

    if sequence is None or graph is None or point_cloud is None:
        return None, None, None, None
    
    # Pass the sequence data through ESM for encoding
    with torch.no_grad():
        encoded_sequence = ESM(sequence, output_hidden_states=True)["hidden_states"][
            -1
        ][0, -1].to("cpu")
        encoded_sequence = z_score_standardization(encoded_sequence)

    # Pass the graph data through VGAE for encoding
    with torch.no_grad():
        encoded_graph = VGAE.encode(graph.x, graph.edge_index).to("cpu")
        encoded_graph = torch.mean(encoded_graph, dim=1)
        encoded_graph = z_score_standardization(encoded_graph)

    # Pass the point cloud data through PAE for encoding
    with torch.no_grad():
        encoded_point_cloud = PAE.encode(point_cloud[None, :]).squeeze().to("cpu")
        encoded_point_cloud = z_score_standardization(encoded_point_cloud)

    # Initialize AttentionFusion
    input_dims = {"sequence": encoded_sequence.shape[0], "graph": encoded_graph.shape[0], "point_cloud": encoded_point_cloud.shape[0]}
    shared_dim = 640
    attention_fusion = AttentionFusion(input_dims, shared_dim)

    fused_rep, _ = attention_fusion(encoded_sequence, encoded_graph, encoded_point_cloud)

    with torch.no_grad():
        multimodal_representation = Fusion.encode(fused_rep).squeeze().to("cpu")

    return multimodal_representation, encoded_sequence, encoded_graph, encoded_point_cloud


def z_score_standardization(tensor):
    mean = tensor.mean()
    std = tensor.std()
    if std != 0:
        standardized_tensor = (tensor - mean) / std
    else:
        standardized_tensor = tensor  # Handle the case when std is 0
    return standardized_tensor


def process_encoded_graph(encoded_graph, edge_index, fixed_size=640, feature_dim=10):
    num_nodes = encoded_graph.size(0)

    if num_nodes > fixed_size:
        ratio = fixed_size / num_nodes
        with torch.no_grad():
            pooling_layer = TopKPooling(in_channels=feature_dim, ratio=ratio)
            pooled_x, edge_index, edge_attr, batch, perm, score = pooling_layer(
                encoded_graph, edge_index
            )
        processed_encoded_graph = pooled_x
    else:
        padding_size = fixed_size - num_nodes
        zero_padding = torch.zeros(padding_size, feature_dim)
        processed_encoded_graph = torch.cat((encoded_graph, zero_padding), dim=0)

    return processed_encoded_graph[:fixed_size]



def read_hdf5(hdf5_path):
    hdf5_file = h5py.File(hdf5_path, "r")
    # Sequence Processing
    amino_acid_indices = hdf5_file["amino_types"][:]
    amino_acid_indices[amino_acid_indices > 20] = 20
    amino_acid_indices[amino_acid_indices == -1] = 20
    sequence = "".join(label_encoder.inverse_transform(amino_acid_indices))
    sequence_token = tokenizer(
        sequence, return_tensors="pt", padding=True, truncation=True, max_length=2048
    )["input_ids"]

    # Graph Processing
    amino_pos = hdf5_file["amino_pos"][:]
    coordinates = np.array(amino_pos, dtype=np.float32).squeeze()
    node_features = one_hot_encode_amino_acid(sequence)
    x = torch.tensor(node_features, dtype=torch.float32)
    edge_index = radius_neighbors_graph(coordinates, radius=6.0, mode='connectivity', include_self='auto')
    edge_index = edge_index.nonzero()
    edge_index = np.array(edge_index)
    edge_index = torch.from_numpy(edge_index).to(torch.long).contiguous()
    neg_edge_index = negative_sampling(
        edge_index=edge_index,
        num_nodes=x.size(0),
        num_neg_samples=edge_index.size(1) // 2,
    )
    graph = Data(x=x, edge_index=edge_index, neg_edge_index=neg_edge_index)

    # Point Cloud Processing
    atom_pos = hdf5_file["atom_pos"][:]
    desired_num_points = 2048
    coordinates = np.array(atom_pos, dtype=np.float32).squeeze()
    num_points = coordinates.shape[0]
    if num_points < desired_num_points:
        padding = np.zeros((desired_num_points - num_points, 3), dtype=np.float32)
        coordinates = np.concatenate((coordinates, padding), axis=0)
    elif num_points > desired_num_points:
        coordinates = coordinates[:desired_num_points, :]
    coordinates = torch.tensor(coordinates, dtype=torch.float32)
    coordinates -= coordinates.mean(0)
    d = np.sqrt((coordinates**2).sum(1))
    coordinates /= d.max()
    point_cloud = torch.FloatTensor(coordinates).permute(1, 0)

    return sequence_token, graph, point_cloud


def suffle(list1, list2):
    temp = list(zip(list1, list2))
    random.shuffle(temp)
    res1, res2 = zip(*temp)
    # res1 and res2 come out as tuples, and so must be converted to lists.
    res1, res2 = list(res1), list(res2)
    return res1, res2


def get_cindex(Y, P):
    return concordance_index(Y, P)


# Prepare for rm2
def get_k(y_obs, y_pred):
    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)
    return sum(y_obs * y_pred) / sum(y_pred**2)


# Prepare for rm2
def squared_error_zero(y_obs, y_pred):
    k = get_k(y_obs, y_pred)
    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)
    y_obs_mean = np.mean(y_obs)
    upp = sum((y_obs - k * y_pred) ** 2)
    down = sum((y_obs - y_obs_mean) ** 2)

    return 1 - (upp / down)


# Prepare for rm2
def r_squared_error(y_obs, y_pred):
    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)
    y_obs_mean = np.mean(y_obs)
    y_pred_mean = np.mean(y_pred)
    mult = sum((y_obs - y_obs_mean) * (y_pred - y_pred_mean)) ** 2
    y_obs_sq = sum((y_obs - y_obs_mean) ** 2)
    y_pred_sq = sum((y_pred - y_pred_mean) ** 2)
    return mult / (y_obs_sq * y_pred_sq)


def get_rm2(Y, P):
    r2 = r_squared_error(Y, P)
    r02 = squared_error_zero(Y, P)

    return r2 * (1 - np.sqrt(np.absolute(r2**2 - r02**2)))


def get_rmse(y, f):
    rmse = sqrt(((y - f) ** 2).mean(axis=0))
    return rmse


def get_mae(y, f):
    mae = (np.abs(y - f)).mean()
    return mae


def get_pearson(y, f):
    rp = pearsonr(y, f)[0]
    return rp


def get_spearman(y, f):
    sp = spearmanr(y, f)[0]

    return sp
