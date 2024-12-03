import os

import numpy as np
import torch
from Bio.PDB import MMCIFParser

parser = MMCIFParser(QUIET=True)

# Function to convert a CIF file to a point cloud
def cif_to_point_cloud(pdb_path, desired_num_points=2048):
    try:
        structure = parser.get_structure('protein', pdb_path)
    except ValueError:
        print(f"value error encountered when parsing {os.path.basename(pdb_path)}")
        return None
    except Exception:
        print(f"unexpected error encountered when parsing {os.path.basename(pdb_path)}")
        return None

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

# Directory containing PDB files
file_directory = os.path.join("data", "raw-structures")

pdb_files = sorted([f for f in os.listdir(file_directory) if os.path.splitext(f)[1] == ".cif"])

total_files = len(pdb_files)
print(f"The Number of files: {total_files}")

ntasks = int(os.environ.get("SLURM_ARRAY_TASK_COUNT", '1'))

task_idx = int(os.environ.get("SLURM_ARRAY_TASK_ID", '0'))
files_to_process = total_files // ntasks
first_file = files_to_process * task_idx
last_file = total_files if task_idx == (ntasks - 1) else (first_file + files_to_process)
res_dir = os.path.join('data', 'pointclouds', '')
idx = first_file

fidx = []

for i in range(first_file, last_file):
    pdb_file = pdb_files[i]
    cif_path = os.path.join(file_directory, pdb_file)
    data = cif_to_point_cloud(cif_path)
    if data is not None:
        fidx.append(i)
        torch.save(data, f"{res_dir}data_{idx}.pt")
        idx += 1
    if (i + 1 - first_file) % 1000 == 0:
        print(f"{i + 1 - first_file} CIF files processed")

with open("data/id-maps/pointcloud.csv", 'w') as f:
    f.write("uniprot_id,data_idx\n")
    for i in fidx:
        f.write(f"{os.path.basename(pdb_files[i]).split('.')[0]},{i}\n")
print(f"Done.\nCreated {idx} point clouds")
