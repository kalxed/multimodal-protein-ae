import os, sys, gzip, csv, pickle, transformers

from tqdm import tqdm
from Bio.PDB import PDBParser, Polypeptide, MMCIFParser
from Bio import SeqIO
from collections import defaultdict

# Data paths, change these to your own paths
data_directory = "./data"
pdb_directory = f"{data_directory}/pdb_files"
cif_directory = f"{data_directory}/cif_files"
fasta_path = f"{data_directory}/uniprotkb_AND_reviewed_true_AND_protein.fasta"
pkl_path = "./data/sequences/pickles"
save_location = "./data/sequences"

# Initialize tracking dictionaries
track = True
non_standard_residues_count = {}
non_standard_residue_labels = defaultdict(int)

# Load the ESM tokenizer
model_token = "facebook/esm2_t30_150M_UR50D"
esm_model = transformers.AutoModelForMaskedLM.from_pretrained(model_token)
esm_tokenizer = transformers.AutoTokenizer.from_pretrained(model_token)

# Parameters for processing files
file_type = "pdb"  # Change this to "pdb" or "fasta" to process different file types
cut_num = 10  # swap to len(pdb_files) to process all files
pkl_file = "smol-sequences.pkl"
max_seq_len = 1024  # TODO: Maximum sequence length for the tokenizer


def eval_seq(structure, protein_name):
    """
    Create a sequence from a protein structure by evaluating the residues.
    This function will ignore non-standard residues and return a string of amino acids.

        structure: A BioPython structure object
        protein_name: 4 letter protein name
    """
    sequence = ""
    non_standard_count = 0

    for residue in structure.get_residues():
        if "CA" in residue:  # Check if the residue is part of the main chain
            try:
                index_code = Polypeptide.three_to_index(residue.get_resname())
                aa_code = Polypeptide.index_to_one(index_code)
                sequence += aa_code
            except KeyError:
                # Count non-standard residues and track them
                residue_name = residue.get_resname()
                if residue_name not in non_standard_residue_labels:
                    non_standard_residue_labels[residue_name] = 0
                non_standard_residue_labels[residue_name] += 1
                non_standard_count += 1
                continue

    non_standard_residues_count[protein_name] = non_standard_count
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


def file_to_seq(file_path, file_type, compressed):
    """
    Parse a PDB or CIF file and extract protein structure.
    Returns a sequence of amino acids.

        file_path: path to the file
        file_type: pdb/cif
        compressed: true/false depending on file path
    """
    # Set Parser
    if file_type == "pdb":
        parser = PDBParser(QUIET=True)
    elif file_type == "cif":
        parser = MMCIFParser(QUIET=True)
    else:
        print("Invalid file type. Please use 'pdb' or 'cif'.")
        return None

    if compressed:
        # Open the compressed CIF file and parse it
        with gzip.open(file_path, "rt") as gz_file:  # 'rt' mode for reading text
            try:
                structure = parser.get_structure(file_path, gz_file)

            except Exception as e:
                print(f"Error parsing {file_path}: {e}")
    else:
        structure = parser.get_structure("protein", file_path)

    sequence = eval_seq(structure, os.path.basename(file_path).split(".")[0])
    return sequence if sequence else None


def track_non_standard_residues(label, count_dict, track):
    """ "
    Track non-standard residues and save them to a CSV file.

        label: protein/residue
        count_dict: dictionary with residue/protein names and their counts
    """
    if not track:
        print("Tracking is disabled.")
        return
    if label == "protein":
        file_name = "protein_NA_residues"
    else:
        file_name = "residue_frequency"

    with open(f"{save_location}/{file_name}.csv", mode="w") as csvfile:
        fieldnames = ["name", "count"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        if label == "protein":
            for protein_name, count in count_dict.items():
                if count > 0:
                    # non_standard_count += 1
                    writer.writerow({"name": protein_name, "count": count})
        else:
            for residue_name, count in count_dict.items():
                writer.writerow({"name": residue_name, "count": count})

    print(f"Non-standard {label} frequencies saved to {file_name}.csv")


def process_files(file_type, cut_num=10):
    """
    Process files and tokenize the sequences using selected tokenizer.

        file_type: cif/pdb
        cut_num (int, optional): Cuts process early, will be removed later. Defaults to 10.
    """
    match file_type:
        case "pdb":
            directory = pdb_directory
        case "cif":
            directory = cif_directory
        case _:
            print("Invalid file type. Please use 'pdb' or 'cif'.")
            return None

    files = [
        f
        for f in os.listdir(directory)
        if os.path.splitext(f)[1].lower() in [".pdb", "cif", ".gz"]
    ]
    tokenized_sequences = []

    for i, file in tqdm(enumerate(files)):
        if i > cut_num:
            break
        file_path = os.path.join(directory, file)
        if os.path.isfile(file_path):
            sequence = file_to_seq(file_path, file_type, file_path.endswith(".gz"))
        if sequence:
            tokenized_sequence = esm_tokenizer(
                sequence, return_tensors="pt", padding=True
            )["input_ids"]
            tokenized_sequences.append(tokenized_sequence)

        # Save intermediate results
        if (i + 1) % 1000 == 0:

            with open(f"{pkl_path}/{pkl_file}", "wb") as f:
                pickle.dump(tokenized_sequences, f)

            track_non_standard_residues("protein", non_standard_residues_count, track)
            track_non_standard_residues("residue", non_standard_residue_labels, track)

            print(f"\n{i + 1} files processed")

    # Save final results
    with open(f"{pkl_path}/{pkl_file}", "wb") as f:
        pickle.dump(tokenized_sequences, f)

    track_non_standard_residues("protein", non_standard_residues_count, track)
    track_non_standard_residues("residue", non_standard_residue_labels, track)


process_files(file_type, cut_num)

print("Done!")
