#!/bin/bash
#SBATCH -J datadown
#SBATCH -n 1
#SBATCH -c 1
#SBATCH -o logs/download-%j.out

bash scripts/sh/get-protein-ids.sh

echo "downloaded the protein ids succesfully"

bash scripts/sh/get-protein-structures.sh

echo "all done"

exit 0
