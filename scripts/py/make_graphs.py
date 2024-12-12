import glob
import os
import os.path as osp

import torch
from Bio.PDB import MMCIFParser

from mpae.utils import structure_file_to_graph

# Directory containing protein structure files
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

res_dir = osp.join("data", "graphs", "")

os.makedirs(res_dir, exist_ok=True)

parser=MMCIFParser(QUIET=True)

# We want to construct a pytorch geometric graph object for each protein, and then save this to a file. 
n = 0
for i in range(first_file, last_file):
    structure_file = structure_files[i]
    data = structure_file_to_graph(structure_file, parser=parser)
    if data is not None:
        # get the basename of the file without the extension, get new file path with the .pt extension in result directory
        fname = f"{res_dir}{osp.splitext(osp.basename(structure_file))[0]}.pt" 
        torch.save(data, fname)
        n += 1
    if (i + 1 - first_file) % 1000 == 0:
        print(f"{i + 1 - first_file} files processed")

print(f"done. made {n} total graphs")
