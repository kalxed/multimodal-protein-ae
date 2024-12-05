#!/bin/bash
#SBATCH -J TEST-GRAPH
#SBATCH -n 1
#SBATCH --gpus-per-task=1
#SBATCH -o logs/test-graph.out

echo "Slurm job id: $SLURM_JOB_ID"


source ~/miniconda3/bin/activate dlprotproj

python pretrain/VGAE.py --mode test

echo "Done"

exit 0