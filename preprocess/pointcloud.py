import glob
import os
import os.path as osp

import numpy as np
import torch
from Bio.PDB import MMCIFParser

parser = MMCIFParser(QUIET=True)

# Function to convert a CIF file to a point cloud
def structure_file_to_pointcloud(pdb_path, desired_num_points=2048):
    try:
        structure = parser.get_structure('protein', pdb_path)
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

# Directory containing PDB files
structure_dir = os.path.join("data", "raw-structures", "")

# sort it to ensure that they are correctly divided among the different array tasks
structure_files = sorted([f for f in glob.glob(f"{structure_dir}*.cif")])
total_files = len(structure_files)

ntasks = int(os.environ.get("SLURM_ARRAY_TASK_COUNT", "1"))
task_idx = int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))

# determine which files to process based on what task number we are
files_to_process = total_files // ntasks
first_file = files_to_process * task_idx
last_file = total_files if task_idx == (ntasks - 1) else (first_file + files_to_process)

res_dir = osp.join('data', 'pointclouds', '')
os.makedirs(res_dir, exist_ok=True)

# We want to construct a pytorch geometric graph object for each protein, and then save this to a file. 
n = 0
for i in range(first_file, last_file):
    structure_file = structure_files[i]
    data = structure_file_to_pointcloud(structure_file)
    if data:
        # get the basename of the file without the extension, get new file path with the .pt extension in result directory
        fname = f"{res_dir}{osp.splitext(osp.basename(structure_file))[0]}.pt" 
        torch.save(data, fname)
        n += 1
    if (i + 1 - first_file) % 1000 == 0:
        print(f"{i + 1 - first_file} files processed")

print(f"done. processed {n} files")

