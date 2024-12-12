#!/bin/bash
#SBATCH -J SEQ-TOKEN
#SBATCH --array=0-4
#SBATCH -o logs/seq/seq-process-%j.out
#SBATCH --mem=16G
#SBATCH -c 1

echo "slurm array task id: $SLURM_ARRAY_TASK_ID"

source ~/miniconda3/bin/activate dlprotproj

python scripts/py/make_tokenized_seqs.py

echo "done"

exit 0