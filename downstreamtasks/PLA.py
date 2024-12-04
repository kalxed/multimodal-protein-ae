import argparse
import torch
import transformers
import pickle
import os
import pandas as pd
import sys
import tqdm
from rdkit import Chem
from rdkit.Chem import AllChem
sys.path.append(".")
from model.ESM import *
from model.VGAE import *
from model.PAE import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_fusion():
    pass

def get_ligand_representation(ligand_smiles):
    mol = Chem.MolFromSmiles(ligand_smiles)
    fingerprint = AllChem.GetMorganFingerprintAsBitVect(
        mol, 2, nBits=1024
    )  # Change radius as needed
    fingerprint_tensor = torch.tensor(fingerprint, dtype=torch.float32)
    return fingerprint_tensor

def setup():
    """
    Setup selected modality with protein-ligand affinity data
    """
    # Load pre-trained models
    vgae_model = torch.load(f"./data/models/VGAE.pt", map_location=device)
    pae_model = torch.load(f"./data/models/PAE.pt", map_location=device)
    model_token = "facebook/esm2_t30_150M_UR50D"
    esm_model = transformers.AutoModelForMaskedLM.from_pretrained(model_token)
    esm_model = esm_model.to(device)
    fusion_model = torch.load(f"./data/models/Fusion.pt", map_location=device)
    print("Pre-trained models loaded successfully.")
    
    # Specify the datasets you are working with
    datasets = ['KIBA', 'DAVIS']
    data_folders = [f'./data/{dataset}' for dataset in datasets]
    
    # Initialize an empty DataFrame to concatenate all data
    df = pd.DataFrame()

    # Loop through each dataset and concatenate the data
    for data_folder in data_folders:
        temp_df = pd.read_csv(f'{data_folder}/label.csv')
        df = pd.concat([df, temp_df], ignore_index=True)
    
    print("Number of samples:", len(df))

    mulmodal = []
    sequence = []
    graph = []
    point_cloud = []

    # Iterate through the dataset to process each sample
    for i, (ligand_smiles, protein_name) in tqdm(enumerate(zip(df["ligand"], df["protein"]))):
        pdb_path = f"{data_folder}/pdb/{protein_name}.pdb"
        multimodal_representation, encoded_sequence, encoded_graph, encoded_point_cloud = get_fusion(pdb_path, esm_model, vgae_model, pae_model, fusion_model)
        ligand_representation = get_ligand_representation(ligand_smiles)

    #     # Concatenate multimodal representation with ligand representation
    #     mulmodal_feature = torch.cat((multimodal_representation, ligand_representation), dim=0).detach().numpy()
    #     sequence_feature = torch.cat((encoded_sequence, ligand_representation), dim=0).detach().numpy()
    #     graph_feature = torch.cat((encoded_graph, ligand_representation), dim=0).detach().numpy()
    #     point_cloud_feature = torch.cat((encoded_point_cloud, ligand_representation), dim=0).detach().numpy()

    #     # Append the features to their respective lists
    #     mulmodal.append(mulmodal_feature)
    #     sequence.append(sequence_feature)
    #     graph.append(graph_feature)
    #     point_cloud.append(point_cloud_feature)

    # # Save the features to pickle files
    # with open(f'{data_folder}/multimodal.pkl', 'wb') as f:
    #     pickle.dump(mulmodal, f)

    # with open(f'{data_folder}/sequence.pkl', 'wb') as f:
    #     pickle.dump(sequence, f)

    # with open(f'{data_folder}/graph.pkl', 'wb') as f:
    #     pickle.dump(graph, f)

    # with open(f'{data_folder}/point_cloud.pkl', 'wb') as f:
    #     pickle.dump(point_cloud, f)
    

def train():
    pass

def test():
    pass

def main():
    parser = argparse.ArgumentParser(description="Multimodal Fusion Model")
    parser.add_argument(
        "--mode",
        choices=["setup", "train", "test", "all"],
        help="Select mode: setup, train, test, or all",
        required=True,
    )
    parser.add_argument(
        "--modality",
        choices=["sequence", "graph", "point cloud", "multimodal"],
        help="Select modality: sequence, graph, point cloud, or multimodal",
        required=False,
        default="multimodal"
    )
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if args.mode == "setup":
        setup()
        
    if args.mode != "setup":
        pass

    if args.mode == "train":
        pass

    if args.mode == "test":
        pass
    
    if args.mode == "all":
        pass


if __name__ == "__main__":
    main()