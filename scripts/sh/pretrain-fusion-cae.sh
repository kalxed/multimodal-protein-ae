#!/bin/bash
#SBATCH -J TRAIN-CAE
#SBATCH -n 1
#SBATCH --gpus-per-task=1
#SBATCH -o logs/cae-attention-pretrain-100.out

echo "Slurm job id: $SLURM_JOB_ID"

source ~/miniconda3/bin/activate dlprotproj

python scripts/py/fusion_cae.py --mode train --pae-path models/PAE-200.pt --id-file fused-proteins --model-path models/CAE-ATTENTION-100.pt --epochs 100

echo "Done training"

exit 0
