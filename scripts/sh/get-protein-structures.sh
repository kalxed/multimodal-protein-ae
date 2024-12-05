#!/bin/bash

input_file="${1:-"protein-ids.tsv.gz"}"
echo "input file: $input_file"
output_dir="${2:-"structures"}"
mkdir -p "$output_dir"
echo "uniprot,pdb" > protein-ids.csv
counter=0
zcat "$input_file" | awk -F'\t' 'NR > 1 { split($3, pdbs, ";"); print $1, pdbs[1] }' | while read -r entry pdb; do
    if [[ -n "$pdb" ]]; then
        # echo "Downloading $pdb for entry $entry..."
        wget -q -O "${output_dir}/${entry}.cif" "https://files.rcsb.org/download/${pdb}.cif"
        echo "${entry},${pdb}" >> protein-ids.csv
    fi
    counter=$((counter + 1))
    if (( counter % 1000 == 0)); then
        echo "processed $counter files"
    fi
done
