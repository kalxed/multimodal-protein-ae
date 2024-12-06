#!/bin/bash
#SBATCH -J TEST-POINTCLOUD
#SBATCH -n 1
#SBATCH --gpus-per-task=1
#SBATCH -o logs/test-pae.out
#SBATCH --mem=16G

echo "Slurm job id: $SLURM_JOB_ID"

source ~/miniconda3/bin/activate dlprotproj

python scripts/py/pretrain_pae.py --mode test --epochs 200 --model-path models/PAE-200.pt

echo "Done"

exit 0
