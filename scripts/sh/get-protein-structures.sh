#!/bin/bash

input_file="${1:-"protein-ids.tsv.gz"}"
echo "input file: $input_file"
output_dir="${2:-"data/raw-structures"}"
mkdir -p "$output_dir"
counter=0
zcat "$input_file" | awk -F'\t' 'NR > 1 { split($3, pdbs, ";"); print $1, pdbs[1] }' | while read -r entry pdb; do
    if [[ -n "$pdb" ]]; then
        # echo "Downloading $pdb for entry $entry..."
        wget -q -O "${output_dir}/${entry}.cif" "https://files.rcsb.org/download/${pdb}.cif"
    fi
    counter=$((counter + 1))
    if (( counter % 1000 == 0)); then
        echo "processed $counter files"
    fi
done
