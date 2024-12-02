import os, sys, gzip, csv, pickle, transformers

# from tqdm import tqdm
from Bio.PDB import Polypeptide, MMCIFParser
from Bio import SeqIO
from collections import defaultdict

# Data paths, change these to your own paths
pkl_path = "data/pickles"

# Load the ESM tokenizer 
# * Change this to your own tokenizer
model_token = "facebook/esm2_t30_150M_UR50D"
esm_model = transformers.AutoModelForMaskedLM.from_pretrained(model_token)
esm_tokenizer = transformers.AutoTokenizer.from_pretrained(model_token)

# Parameters for processing files
pkl_file = "smol-sequences.pkl"

parser = MMCIFParser(QUIET=True)

def eval_seq(structure):
    """
    Create a sequence from a protein structure by evaluating the residues.
    This function will ignore non-standard residues and return a string of amino acids.

    Args:
        structure: A BioPython structure object
    """
    sequence = ""

    for residue in structure.get_residues():
        if "CA" in residue:  # Check if the residue is part of the main chain
            try:
                index_code = Polypeptide.three_to_index(residue.get_resname())
                aa_code = Polypeptide.index_to_one(index_code)
                sequence += aa_code
            except KeyError:
                continue

    return sequence


def file_to_seq(file_path):
    """
    Parse a PDB or CIF file and extract protein structure.
    Returns a sequence of amino acids.

    Args:
        file_path: path to the file
    """
    # Set Parser
    structure = parser.get_structure("protein", file_path)

    sequence = eval_seq(structure)
    return sequence if sequence else None


def process_files():
    """
    Process files and tokenize the sequences using selected tokenizer.

    """
    directory = 'data/structures'

    files = [
        f
        for f in os.listdir(directory)
        if f.split('.')[-1] == "cif"
    ]
    tokenized_sequences = []

    for i, file in enumerate(files):
        file_path = os.path.join(directory, file)
        if os.path.isfile(file_path):
            sequence = file_to_seq(file_path)
        if sequence:
            tokenized_sequence = esm_tokenizer(
                sequence, return_tensors="pt", padding=True
            )["input_ids"]
            tokenized_sequences.append(tokenized_sequence)

        # Save intermediate results
        if (i + 1) % 1000 == 0:

            with open(f"{pkl_path}/{pkl_file}", "wb") as f:
                pickle.dump(tokenized_sequences, f)

            print(f"{i + 1} files processed")

    # Save final results
    with open(f"{pkl_path}/{pkl_file}", "wb") as f:
        pickle.dump(tokenized_sequences, f)


process_files()

print("Done!")
