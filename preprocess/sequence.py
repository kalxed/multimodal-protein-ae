import glob
import gzip
import os
import os.path as osp

import torch
import transformers
from Bio import SeqIO
from Bio.PDB import MMCIFParser, PDBParser, Polypeptide

# Data paths, change these to your own paths
structure_dir = "data/raw-structures"
pkl_path = "./data"

# Load the ESM tokenizer 
# * Change this to your own tokenizer
model_token = "facebook/esm2_t30_150M_UR50D"
esm_model = transformers.AutoModelForMaskedLM.from_pretrained(model_token)
esm_tokenizer = transformers.AutoTokenizer.from_pretrained(model_token)

# Parameters for processing files
file_type = "cif"  # Change this to "pdb" or "fasta" to process different file types
cut_num = 10  # swap to len(pdb_files) to process all files
pkl_file = "sequences.pkl"
max_seq_len = 1024  # TODO: Maximum sequence length for the tokenizer


def eval_seq(structure_path: str, parser):
    """
    Create a sequence from a protein structure by evaluating the residues.
    This function will ignore non-standard residues and return a string of amino acids.

        structure: A BioPython structure object
        protein_name: 4 letter protein name
    """
    

    try:
        structure = parser.get_structure('protein', structure_path)
    except Exception as e:
        return None
    

    sequence = ""
    for residue in structure.get_residues():
        if "CA" in residue:  # Check if the residue is part of the main chain
            try:
                index_code = Polypeptide.three_to_index(residue.get_resname())
                aa_code = Polypeptide.index_to_one(index_code)
                sequence += aa_code
            except KeyError:
                # continue
                # opting to drop the protein since that is what the graphs do.
                return None
    return sequence


def fatsa_to_sequence(file_path):
    """
    Extract amino acid sequences from a FASTA file and save them to a new file.
    (Will probably delete this function later)

        file_path: Path to FATSA file
    """
    # Parse the sequences and extract only amino acid sequences
    amino_acid_sequences = []
    for record in SeqIO.parse(
        file_path, "fasta"
    ):  # Change "fasta" to your format if needed
        # Check if the sequence contains amino acids (optional, depends on your data format)
        if all(char in "ACDEFGHIKLMNPQRSTVWY" for char in record.seq):
            amino_acid_sequences.append(f"{record.seq}")

    count = 0
    # Display the extracted amino acid sequences
    for record in amino_acid_sequences:
        if count > 10:
            break
        print(f">{record.id}\nThis is a sequence {record.seq}")
        count += 1

    # Optional: Write these sequences to a new file
    output_file = "amino_acid_sequences.fasta"
    with open(output_file, "w") as f:
        for seq in amino_acid_sequences:
            f.write(f"{seq}\n")
    print(f"Amino acid sequences saved to {output_file}")

def process_files(structure_files: list[str], res_dir:str, parser=PDBParser(QUIET=True)):
    """
    Process files and tokenize the sequences using selected tokenizer.

    """   

    print("Number of Files: ", len(structure_files))
    n = 0
    os.makedirs(res_dir, exist_ok=True)
    for i, structure_file in enumerate(structure_files):
        if osp.isfile(structure_file):
            sequence = eval_seq(structure_file, parser)
            if sequence:
                tokenized_sequence = esm_tokenizer(sequence, return_tensors="pt", padding=True)["input_ids"]
                # get the basename of the file without the extension, get new file path with the .pt extension in result directory
                fname = osp.join(res_dir, f"{osp.splitext(osp.basename(structure_file))[0]}.pt")
                torch.save(tokenized_sequence, fname)
                n += 1

        if (i + 1) % 1000 == 0:
            print(f"\n{i + 1} files processed")
    return n

parser = MMCIFParser(QUIET=True)

structure_files = sorted([f for f in glob.glob(f"{structure_dir}/*.cif")])

ntasks = int(os.environ.get("SLURM_ARRAY_TASK_COUNT", "1"))
task_idx = int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))

total_files = len(structure_files)

# determine which files to process based on what task number we are

files_to_process = total_files // ntasks
first_file = files_to_process * task_idx
last_file = total_files if task_idx == (ntasks - 1) else (first_file + files_to_process)

res_dir = osp.join('data', 'sequences', '')
os.makedirs(res_dir, exist_ok=True)

n = process_files(structure_files[first_file:last_file], "res_dir", parser=parser)

print(f"Done. made {n} succesful tokenized sequences")
