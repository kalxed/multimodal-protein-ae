#!/bin/bash
#SBATCH -J datadown
#SBATCH -n 1
#SBATCH -c 1
#SBATCH -o download-%j.out

bash get-data/get-protein-ids.sh

echo "downloaded the protein ids succesfully"

bash get-data/get-protein-structures.sh

echo "all done"

exit 0
