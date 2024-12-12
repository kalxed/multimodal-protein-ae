import glob
import os
import os.path as osp

import torch
from Bio.PDB import MMCIFParser

from mpae.utils import structure_file_to_sequence, tokenize_sequence

# Data paths, change these to your own paths
structure_dir = osp.join("data", "raw-structures", "")

res_dir = osp.join("data", "sequences", "")

os.makedirs(res_dir, exist_ok=True)

parser = MMCIFParser(QUIET=True)

structure_files = sorted([f for f in glob.glob(f"{structure_dir}*.cif")])

ntasks = int(os.environ.get("SLURM_ARRAY_TASK_COUNT", "1"))
task_idx = int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))

total_files = len(structure_files)



# determine which files to process based on what task number we are
files_to_process = total_files // ntasks
first_file = files_to_process * task_idx
last_file = total_files if task_idx == (ntasks - 1) else (first_file + files_to_process)
n = 0
for i, structure_file in enumerate(structure_files):
    if osp.isfile(structure_file):
        sequence = structure_file_to_sequence(structure_file, parser)
        if sequence:
            tokenized_sequence = tokenize_sequence(sequence)
            # get the basename of the file without the extension, get new file path with the .pt extension in result directory
            fname = osp.join(res_dir, f"{osp.splitext(osp.basename(structure_file))[0]}.pt")
            torch.save(tokenized_sequence, fname)
            n+=1

    if (i + 1) % 1000 == 0:
        print(f"\n{i + 1} files processed")

print(f"done. succesfully tokenized sequences of {n} total proteins")