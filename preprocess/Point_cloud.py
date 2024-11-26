import pickle
import numpy as np
from Bio import PDB
import os
import torch
from tqdm import tqdm

# Function to convert a CIF file to a point cloud
def cif_to_point_cloud(cif_path, desired_num_points=2048):
    parser = PDB.MMCIFParser(QUIET=True)
    try:
        structure = parser.get_structure('protein', cif_path)
    except ValueError:
        return None
    coordinates = []
    for atom in structure.get_atoms():
        coordinates.append(atom.get_coord())
    coordinates = np.array(coordinates, dtype=np.float32)
    num_points = coordinates.shape[0]

    # I DON'T LIKE THIS AT ALL
    # Pad or truncate the point cloud to the desired number of points
    if num_points < desired_num_points:
        padding = np.zeros((desired_num_points - num_points, 3), dtype=np.float32)
        coordinates = np.concatenate((coordinates, padding), axis=0)
    elif num_points > desired_num_points:
        coordinates = coordinates[:desired_num_points, :]
    
    # Center and normalize the point cloud
    coordinates = torch.tensor(coordinates, dtype=torch.float32)
    coordinates -= coordinates.mean(0)
    d = np.sqrt((coordinates ** 2).sum(1))
    coordinates /= d.max()
    coordinates = torch.FloatTensor(coordinates).permute(1, 0)
    return coordinates

# Function to convert a PDB file to a point cloud
def pdb_to_point_cloud(pdb_path, desired_num_points=2048):
    parser = PDB.PDBParser(QUIET=True)
    try:
        structure = parser.get_structure('protein', pdb_path)
    except ValueError:
        return None
    coordinates = []
    for atom in structure.get_atoms():
        coordinates.append(atom.get_coord())
    coordinates = np.array(coordinates, dtype=np.float32)
    num_points = coordinates.shape[0]

    # Pad or truncate the point cloud to the desired number of points
    if num_points < desired_num_points:
        padding = np.zeros((desired_num_points - num_points, 3), dtype=np.float32)
        coordinates = np.concatenate((coordinates, padding), axis=0)
    elif num_points > desired_num_points:
        coordinates = coordinates[:desired_num_points, :]

    # Center and normalize the point cloud
    coordinates = torch.tensor(coordinates, dtype=torch.float32)
    coordinates -= coordinates.mean(0)
    d = np.sqrt((coordinates ** 2).sum(1))
    coordinates /= d.max()
    coordinates = torch.FloatTensor(coordinates).permute(1, 0)
    return coordinates

# This directory is relative to where the script is run
# Directory containing PDB files
file_directory = "./data"
pdb_files = [f for f in os.listdir(file_directory) if os.path.splitext(f)[1] == ".pdb"]
cif_files = [f for f in os.listdir(file_directory) if os.path.splitext(f)[1] == ".cif"]
print("The Number of files:", len(pdb_files) + len(cif_files))
dataset = []

# Process PDB files to create point clouds
for i, pdb_file in tqdm(enumerate(pdb_files)):
    pdb_path = os.path.join(file_directory, pdb_file)
    data = pdb_to_point_cloud(pdb_path)
    dataset.append(data)
    if (i + 1) % 1000 == 0:
        print(f"{i + 1} PDB files processed")
print("Done")

# Process CIF files to create point clouds
for i, cif_file in tqdm(enumerate(cif_files)):
    cif_path = os.path.join(file_directory, cif_file)
    data = cif_to_point_cloud(cif_path)
    dataset.append(data)
    if (i + 1) % 1000 == 0:
        print(f"{i + 1} CIF files processed")

# Save the dataset of point clouds to a file
with open(f'{file_directory}/pointclouds.pkl', 'wb') as f:
    pickle.dump(dataset, f)
