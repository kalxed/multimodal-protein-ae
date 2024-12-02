import os

import numpy as np
import torch
from Bio.PDB import MMCIFParser, Polypeptide, PPBuilder
from sklearn.neighbors import radius_neighbors_graph
from sklearn.preprocessing import LabelEncoder
from torch_geometric.data import Data
from torch_geometric.utils import negative_sampling

# Initialize a PDB parser and PPBuilder for protein structure parsing
parser = MMCIFParser(QUIET=True)
ppb = PPBuilder()

# Define the standard amino acids and create a LabelEncoder for encoding them
amino_acids = list(range(0, 21))
label_encoder = LabelEncoder()
label_encoder.fit(amino_acids)
num_amino_acids = len(amino_acids)

# Function to one-hot encode amino acid sequences
def one_hot_encode_amino_acid(sequence):
    amino_acid_indices = label_encoder.transform(list(sequence))
    one_hot = np.zeros((len(sequence), num_amino_acids), dtype=np.float32)
    one_hot[np.arange(len(sequence)), amino_acid_indices] = 1
    return one_hot

# Function to convert a PDB file to a PyTorch Geometric graph data object
def pdb_to_graph(pdb_path, radius=7):
    try:
      structure = parser.get_structure('protein', pdb_path)
    except ValueError:
      print(f"value error encountered when parsing {os.path.basename(pdb_path)}")
      return None
    except Exception:
      print(f"unexpected error encountered when parsing {os.path.basename(pdb_path)}")
      return None

    node_features = []
    coordinates = []
    sequence = []

    for residue in structure.get_residues():
      if 'CA' in residue:
        try:
          aa_code = Polypeptide.three_to_index(residue.get_resname())
        except KeyError:
          print(f"unexpected amino acid in {os.path.basename(pdb_path)}, it was {residue.get_resname()}")
          return None
        sequence.append(aa_code)
        coordinates.append(residue['CA'].get_coord())
    coordinates = np.array(coordinates, dtype=np.float32)
    try:
      node_features = one_hot_encode_amino_acid(sequence)
    except IndexError:
      print(f"unexpected amino acid encountered in {os.path.basename(pdb_path)}")
      return None
    x = torch.tensor(node_features, dtype=torch.float32)
    print(f"coords: {coordinates.shape}")
    print(f"x: {x.shape}")

    # Calculate edges based on distance
    edge_index = radius_neighbors_graph(coordinates, radius, mode='connectivity', include_self=False)
    print(f"edge: {edge_index.shape}")
    print(f"coo edge: {edge_index.tocoo().shape}")
    # edge_index = np.array(edge_index.nonzero())
    edge_index = edge_index.tocoo()
    edge_index = torch.tensor(np.array([edge_index.row, edge_index.col]), dtype=torch.long).contiguous()

    
    # Generate negative edge samples
    neg_edge_index = negative_sampling(
        edge_index=edge_index,
        num_nodes=x.size(0),
        num_neg_samples=edge_index.size(1) // 2
    )

    # Create a PyTorch Geometric Data object
    data = Data(x=x, edge_index=edge_index, neg_edge_index=neg_edge_index)

    return data

# Directory containing PDB files
pdb_directory = "data/structures"
pdb_files = [
    f
    for f in os.listdir(pdb_directory)
    if f.split('.')[-1] == "cif"
]
total_files = len(pdb_files)

# Process PDB files to create graph data objects
print(f"Total number of files: {total_files}")

res_dir = os.path.join('data', 'graphs')
if not os.path.exists(res_dir):
   os.mkdir(res_dir)

idx = 0
for i, pdb_file in enumerate(pdb_files):
    pdb_path = os.path.join(pdb_directory, pdb_file)
    data = pdb_to_graph(pdb_path)
    if data is not None:
        torch.save(data, os.path.join(res_dir, f"data_{idx}.pt"))
        idx += 1
    if (i + 1) % 1000 == 0:
        print(f"{i + 1} files processed")

print("Done")

print(f"{idx} graphs successfully created")


