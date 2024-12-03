import glob
import os
import os.path as osp

import numpy as np
import torch
from Bio.PDB import MMCIFParser, Polypeptide, PPBuilder
from sklearn.neighbors import radius_neighbors_graph
from sklearn.preprocessing import LabelEncoder
from torch_geometric.data import Data
from torch_geometric.utils import negative_sampling

# Initialize a PDB parser and PPBuilder for protein structure parsing
parser = MMCIFParser(QUIET=True)

# Define the standard amino acids and create a LabelEncoder for encoding them
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

# Function to convert a PDB file to a PyTorch Geometric graph data object
def structure_file_to_graph(pdb_path, radius=6.0):
    try:
        structure = parser.get_structure('protein', pdb_path)
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

structure_dir = osp.join("data", "raw-structures", "")

# sort it to ensure that they are correctly divided among the different array tasks
structure_files = sorted([f for f in glob.glob(f"{structure_dir}*.cif")])
total_files = len(structure_files)

ntasks = int(os.environ.get("SLURM_ARRAY_TASK_COUNT", "1"))
task_idx = int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))

# determine which files to process based on what task number we are
files_to_process = total_files // ntasks
first_file = files_to_process * task_idx
last_file = total_files if task_idx == (ntasks - 1) else (first_file + files_to_process)

res_dir = osp.join('data', 'graphs', '')
os.makedirs(res_dir, exist_ok=True)

# We want to construct a pytorch geometric graph object for each protein, and then save this to a file. 
n = 0
for i in range(first_file, last_file):
    structure_file = structure_files[i]
    data = structure_file_to_graph(structure_file)
    if data:
        # get the basename of the file without the extension, get new file path with the .pt extension in result directory
        fname = f"{res_dir}{osp.splitext(osp.basename(structure_file))[0]}.pt" 
        torch.save(data, fname)
        n += 1
    if (i + 1 - first_file) % 1000 == 0:
        print(f"{i + 1 - first_file} files processed")

print(f"done. processed {n} files")
