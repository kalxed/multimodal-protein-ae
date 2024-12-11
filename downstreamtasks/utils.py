import os
import torch
import transformers
from Bio.PDB import PDBParser, Polypeptide
import numpy as np
from torch_geometric.data import Data
from torch_geometric.utils import negative_sampling
from sklearn.preprocessing import LabelEncoder
import warnings
import collections
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

parser = PDBParser(QUIET=True)
ppb = PPBuilder()

import sys

sys.path.append(".")

from model.Concrete_Autoencoder import *
from model.ESM import *
from model.vgae import *
from model.PAE import *
from model.Attention import *

import torch
import collections
import transformers

def load_models():
    # Ititialize VGAE model
    out_channels = 10
    num_features = 20
    vgae_model_path = "./data/models/VGAE.pt"
    vgae_model = VGAE(VariationalGCNEncoder(num_features, out_channels))
    
    state_dict = torch.load(vgae_model_path, map_location=device)
    if isinstance(state_dict, collections.OrderedDict):
        vgae_model.load_state_dict(state_dict)
        vgae_model.eval()
    else:
        raise ValueError("The VGAE file does not contain a valid state dictionary.")

    # Load PAE model
    k = 640
    num_points = 2048
    pae_model_path = "./data/models/PAE.pt"
    pae_model = PointAutoencoder(k, num_points)
    
    state_dict = torch.load(pae_model_path, map_location=device)
    if isinstance(state_dict, collections.OrderedDict):
        pae_model.load_state_dict(state_dict)
        pae_model.eval()
    else:
        raise ValueError("The PAE file does not contain a valid state dictionary.")
    
    # Load ESM model
    model_token = "facebook/esm2_t30_150M_UR50D"
    esm_model = transformers.AutoModelForMaskedLM.from_pretrained(model_token)
    esm_model = esm_model.to(device)
    
    # Load CAE model
    input_dim = 640 * 3 # Input dimension after fusion
    shared_dim = 640  # Shared dimension after fusion
    latent_dim = 64  # Latent space size
    temperature = .5  # Concrete distribution temperature
    concrete_model_path = "./data/models/CAE-ATTENTION.pt"
    concrete_model = ConcreteAutoencoder(input_dim, latent_dim, shared_dim, temperature).to(device)
    state_dict = torch.load(concrete_model_path, map_location=device)
    if isinstance(state_dict, collections.OrderedDict):
        concrete_model.load_state_dict(state_dict)
        concrete_model.eval()
    else:
        raise ValueError("The CAE file does not contain a valid state dictionary.")
    # concrete_model = None
    
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
        
def pickle_batch_dump(batch_num, data_folder, mulmodal, sequence, graph, point_cloud):
    # Create directories if they do not exist
    if not os.path.exists(f'{data_folder}/multimodal'):
        os.makedirs(f'{data_folder}/multimodal')
    if not os.path.exists(f'{data_folder}/sequences'):
        os.makedirs(f'{data_folder}/sequences')
    if not os.path.exists(f'{data_folder}/graphs'):
        os.makedirs(f'{data_folder}/graphs')
    if not os.path.exists(f'{data_folder}/pointclouds'):
        os.makedirs(f'{data_folder}/pointclouds')
    
    # Save the features to pickle files    
    with open(f'{data_folder}/multimodal/{batch_num}.pkl', 'wb') as f:
        pickle.dump(mulmodal, f)

    with open(f'{data_folder}/sequences/{batch_num}.pkl', 'wb') as f:
        pickle.dump(sequence, f)

    with open(f'{data_folder}/graphs/{batch_num}.pkl', 'wb') as f:
        pickle.dump(graph, f)

    with open(f'{data_folder}/pointclouds/{batch_num}.pkl', 'wb') as f:
        pickle.dump(point_cloud, f)

def load_batch_data(modality_folder, modality):
    """
    Load data in batches from a folder and return as a list.
    """
    batch_files = [os.path.join(modality_folder, name) for name in os.listdir(modality_folder) if name.endswith('.pkl')]

    all_data = []
    for batch_file in batch_files:
        with open(batch_file, 'rb') as f:
            data = pickle.load(f)
            all_data.extend(data)

    return np.array(all_data)

def load_data(data_folder, modal):
    """
    Load data from a specific modality pickle file.
    """
    with open(f'{data_folder}/{modal}.pkl', 'rb') as f:
        tensor_list = pickle.load(f)
        data = [tensor.detach().numpy() for tensor in tensor_list]

    return np.array(data)

def pad_truc(seq, max_len):
    """
    Pad or truncate a sequence to the specified length.
    """
    if len(seq) < max_len:
        return np.pad(seq, (0, max_len - len(seq)), 'constant')
    else:
        return seq[:max_len]

        
def get_aa_label_encoder(file_type):
    if file_type == "pdb":
        amino_acids = list(range(20))
    elif file_type == "hdf5":
        amino_acids = [
            'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
            'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y'
        ]
    else:
        raise ValueError("Invalid file type. Please choose from 'pdb' or 'hdf5'.")
    
    label_encoder = LabelEncoder()
    label_encoder.fit(list(amino_acids))
    return label_encoder, amino_acids

def one_hot_encode_amino_acid(sequence, file_type):
    label_encoder, amino_acids = get_aa_label_encoder(file_type)
    
    # Ensure the sequence contains only valid amino acids
    try:
        amino_acid_indices = label_encoder.transform(list(sequence))
    except ValueError as e:
        print(f"Invalid amino acid in sequence: {e}")
        return None  # Skip invalid sequences
    
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
    node_features = one_hot_encode_amino_acid(sequence, "pdb")
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
    coordinates = torch.tensor(coordinates, dtype=torch.float32)
    coordinates -= coordinates.mean(0)
    d = np.sqrt((coordinates ** 2).sum(1))
    coordinates /= d.max()
    point_cloud = torch.FloatTensor(coordinates).permute(1, 0)

    # Sequence
    sequence = esm_tokenizer(sequence, return_tensors="pt", padding=True, truncation=True, max_length=2048)["input_ids"]

    return sequence, graph, point_cloud

# Function to process and adjust encoded graph data to a fixed size
def process_encoded_graph(encoded_graph, edge_index, fixed_size=640, feature_dim=10):
    num_nodes = encoded_graph.size(0)
    if num_nodes > fixed_size:
        ratio = fixed_size / num_nodes
        with torch.no_grad():
            pooling_layer = TopKPooling(in_channels=feature_dim, ratio=ratio)
            pooled_x, edge_index, edge_attr, batch, perm, score = pooling_layer(encoded_graph, edge_index)
        processed_encoded_graph = pooled_x
    else:
        padding_size = fixed_size - num_nodes
        zero_padding = torch.zeros(padding_size, feature_dim)
        processed_encoded_graph = torch.cat((encoded_graph, zero_padding), dim=0)

    return processed_encoded_graph

def get_modalities(protein_path, ESM, VGAE, PAE, CAE):
    # Check file_extension
    file_name, file_extension = os.path.splitext(protein_path)
    if file_extension == ".pdb":
        sequence, graph, point_cloud = read_pdb(protein_path)
    elif file_extension == ".hdf5":
        sequence, graph, point_cloud = read_hdf5(protein_path)

    if sequence is None or graph is None or point_cloud is None:
        print(f'Failed {file_name}')
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
        encoded_graph = process_encoded_graph(encoded_graph, graph.edge_index)
        encoded_graph = torch.mean(encoded_graph, dim=1)
        encoded_graph = z_score_standardization(encoded_graph)

    # Pass the point cloud data through PAE for encoding
    with torch.no_grad():
        encoded_point_cloud = PAE.encode(point_cloud[None, :]).squeeze().to("cpu")
        encoded_point_cloud = z_score_standardization(encoded_point_cloud)
    
    fused_rep = torch.cat((encoded_sequence, encoded_graph, encoded_point_cloud), dim=0)
    if fused_rep.size(0) == 1920:
        with torch.no_grad():
            multimodal_representation = CAE.encode(fused_rep).squeeze().to("cpu")
    else: 
        return None, None, None, None

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

# Standard amino acid mapping
amino_acid_mapping = {
        'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4,
        'G': 5, 'H': 6, 'I': 7, 'K': 8, 'L': 9,
        'M': 10, 'N': 11, 'P': 12, 'Q': 13, 'R': 14,
        'S': 15, 'T': 16, 'V': 17, 'W': 18, 'Y': 19
    }

def read_hdf5(hdf5_path):
    hdf5_file = h5py.File(hdf5_path, "r")
    # Sequence Processing
    amino_acid_indices = hdf5_file["amino_types"][:]
    amino_acid_indices[amino_acid_indices > 20] = 20
    amino_acid_indices[amino_acid_indices == -1] = 20
    label_encoder, amino_acids = get_aa_label_encoder("hdf5")
    try:
        sequence = "".join(label_encoder.inverse_transform(amino_acid_indices))
    except ValueError as e:
        print(f"Error converting amino acid indices at {hdf5_path}: {e}")
        return None, None, None

    
    sequence_token = tokenizer(
        sequence, return_tensors="pt", padding=True, truncation=True, max_length=2048
    )["input_ids"]

    # Graph Processing
    amino_pos = hdf5_file["amino_pos"][:]
    coordinates = np.array(amino_pos, dtype=np.float32).squeeze()
    node_features = one_hot_encode_amino_acid(sequence, "hdf5")
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
