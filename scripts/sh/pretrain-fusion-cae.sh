#!/bin/bash
#SBATCH -J CAE-FUSION-TRAIN
#SBATCH -n 1
#SBATCH --gpus-per-task=1
#SBATCH -o logs/cae-pretrain.out

echo "Slurm job id: $SLURM_JOB_ID"

source ~/miniconda3/bin/activate dlprotproj

python scripts/py/fusion_cae.py --mode test --pae-path models/PAE-200.pt --id-file fused-proteins

echo "Done training"

exit 0
