#!/bin/bash
#SBATCH -J renumber
#SBATCH --output=renumber-%J.out
#SBATCH --partition=short
#SBATCH -c 1
#SBATCH -n 1
#SBATCH -N 1


in_dir="${1:-"data/graphs"}"
counter=0
echo "$in_dir"
tmp_dir="$in_dir/tem"
mkdir -p "$tmp_dir"
files=$(ls "$in_dir"/*.pt | sort -V)
echo "got the list of files"
for file in $files; do
    new_file="$tmp_dir/data_$counter.pt"
    mv "$file" "$new_file"
    counter=$((counter + 1))
    if [ $(($counter % 1000)) -eq 0 ]; then
        echo "renamed $counter files"
    fi
done

echo "moved all to temp"

# Move renamed files back to the original directory
mv "$tmp_dir"/* "$in_dir/"
rmdir "$tmp_dir"

echo "Renumbering complete. Total files renumbered: $counter"

