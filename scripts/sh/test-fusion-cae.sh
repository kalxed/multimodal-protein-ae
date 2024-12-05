#!/bin/bash
#SBATCH -J CAE-FUSION-TEST
#SBATCH -n 1
#SBATCH --gpus-per-task=1
#SBATCH -o logs/fusion.out

echo "Slurm job id: $SLURM_JOB_ID"

source ~/miniconda3/bin/activate dlprotproj

python scripts/py/encode_fuse.py --mode train --pae-path models/PAE-200.pt

echo "Done training"

exit 0
