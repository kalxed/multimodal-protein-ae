#!/bin/bash
#SBATCH -J POINTCLOUD
#SBATCH --array=0-4
#SBATCH -c 1
#SBATCH -o logs/point/pointclouds-make-%j.out
#SBATCH --mem=16G

source ~/miniconda3/bin/activate dlprotproj

echo "slurm array task id: $SLURM_ARRAY_TASK_ID"

python preprocess/pointcloud.py

echo "done"

exit 0
