#!/bin/bash

min_length="${1:-80}"
max_length="${2:-*}"
fout="${3:-protein-ids.tsv.gz}"
echo "min length: $min_length\nmax length: $max_length\nfout: $fout"

url="https://rest.uniprot.org/uniprotkb/stream?compressed=true&fields=accession%2Clength%2Cxref_pdb&format=tsv&query=%28*%29+AND+%28reviewed%3Atrue%29+AND+%28proteins_with%3A1%29+AND+%28length%3A%5B${min_length}+TO+${max_length}%5D%29"

curl -o "$fout" "$url"

exit 0
