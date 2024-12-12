import glob
import os.path as osp

"""
This determines the proteins that were succesfully transfered to graph, point cloud, and tokenized sequences.
Then it dumps these proteins into a file.
"""

root = "data"

modes = ["graphs", "pointclouds", "sequences"]

# get the proteins that were succesfully transfered to graph, point cloud, and tokenizes sequences
protein_ids = set.intersection(*[set(osp.basename(x) for x in glob.glob(osp.join(root, mode, "*.pt"))) for mode in modes])

protein_ids = sorted(protein_ids)

with open("proteins", 'w') as f:
    for i in range(len(protein_ids)-1):
        f.write(f"{protein_ids[i]}\n")
    f.write(f"{protein_ids[-1]}")

