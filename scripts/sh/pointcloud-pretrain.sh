#!/bin/bash
#SBATCH -J TRAIN-POINTCLOUD
#SBATCH -n 1
#SBATCH --gpus-per-task=1
#SBATCH -o logs/pretrain-points-200.out
#SBATCH --mem=16G

echo "Slurm job id: $SLURM_JOB_ID"

source ~/miniconda3/bin/activate dlprotproj

python scripts/py/pretrain_pae.py --mode train --epochs 200 --model-path models/PAE-200.pt

echo "Done"

exit 0
