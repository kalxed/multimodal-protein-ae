#!/bin/bash
#SBATCH -J TRAIN-GRAPH
#SBATCH -n 1
#SBATCH --gpus-per-task=1
#SBATCH -o logs/pretrain-graph.out

echo "Slurm job id: $SLURM_JOB_ID"


source ~/miniconda3/bin/activate dlprotproj

python scripts/py/pretrain_vgae.py --mode train

echo "Done"

exit 0

