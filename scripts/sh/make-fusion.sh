#!/bin/bash
#SBATCH -J FUSION
#SBATCH -n 1
#SBATCH --gpus-per-task=1
#SBATCH -o logs/fusion.out

echo "Slurm job id: $SLURM_JOB_ID"

source ~/miniconda3/bin/activate dlprotproj

python scripts/py/encode_fuse.py --mode process --pae-path models/PAE-200.pt

echo "Done."

exit 0
