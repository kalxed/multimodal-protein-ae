#!/bin/bash
#SBATCH -J GRAPH-MAKE
#SBATCH --array=0-4
#SBATCH -c 1
#SBATCH -o logs/graph/graph-construct-%j.out
#SBATCH --mem=16G

source ~/miniconda3/bin/activate dlprotproj

echo "$SLURM_ARRAY_TASK_ID"

python preprocess/graph.py

echo "done"

exit 0
