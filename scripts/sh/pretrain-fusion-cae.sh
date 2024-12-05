#!/bin/bash
#SBATCH -J CAE-FUSION-TRAIN
#SBATCH -n 1
#SBATCH --gpus-per-task=1
#SBATCH -o logs/cae-pretrain.out

echo "Slurm job id: $SLURM_JOB_ID"

source ~/miniconda3/bin/activate dlprotproj

python scripts/py/encode_fuse.py --mode test --pae-path models/PAE-200.pt

echo "Done training"

exit 0
