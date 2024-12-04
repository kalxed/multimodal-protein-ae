import argparse
import pickle
import os
import sys
import transformers
from torch_geometric.loader import DataLoader
from torch.utils.data import Dataset
import torch
import torch.nn as nn
from tqdm import tqdm
sys.path.append(".")
from model.ESM import *
from model.VGAE import *
from model.PAE import *
from model.Attention import AttentionFusion
from model.Concrete_Autoencoder import ConcreteAutoencoder

class MultimodalDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data_point = self.data[index]
        return data_point  # Returning as-is for now
    
# Concrete autoencoder parameters
input_dim = 640 * 3  # Input dimension after fusion
shared_dim = 640  # Shared dimension after fusion
latent_dim = 64  # Latent space size
temperature = .5  # Concrete distribution temperature
    
# Function for Z-score standardization
def z_score_standardization(tensor):
    mean = tensor.mean()
    std = tensor.std()
    if std != 0:
        standardized_tensor = (tensor - mean) / std
    else:
        standardized_tensor = tensor  # Handle the case when std is 0
    return standardized_tensor

def process(device):
    os.path.dirname(os.path.abspath(__file__))
    vgae_model = torch.load(f"./data/models/VGAE.pt", map_location=device, weights_only=True)
    pae_model = torch.load(f"./data/models/PAE.pt", map_location=device, weights_only=True)
    model_token = "facebook/esm2_t30_150M_UR50D"
    esm_model = transformers.AutoModelForMaskedLM.from_pretrained(model_token)
    esm_model = esm_model.to(device)
    print("Pre-trained models loaded successfully.")

    with open(f"./data/sequences/pickles/graphs.pkl", "rb") as f:
        print("Loading graph data ...")
        graph_data = pickle.load(f)
    print("Graph data loaded successfully.")

    with open(f"./data/sequences/pickles/pointclouds.pkl", "rb") as f:
        print("Loading point cloud data ...")
        point_cloud_data = pickle.load(f)
    print("Point Cloud data loaded successfully.")

    with open(f"./data/sequences/pickles/sequences.pkl", "rb") as f:
        print("Loading sequence data ...")
        sequence_data = pickle.load(f)
    print("Sequence data loaded successfully.")

    modality_dim = 640  # Dimension of each modality
    shared_dim = modality_dim * 3
    processed_data_list = []

    for i, (graph, point_cloud, sequence) in enumerate(
            tqdm(zip(graph_data, point_cloud_data, sequence_data), total=len(sequence_data), desc="Processing modalities")
        ):
        # Encode sequence data using ESM
        with torch.no_grad():
            sequence = sequence.to(device)
            encoded_sequence = esm_model(sequence, output_hidden_states=True)[
                "hidden_states"
            ][-1][0, -1].to("cpu")
            encoded_sequence = z_score_standardization(encoded_sequence)

        # Encode graph data using VGAE
        with torch.no_grad():
            graph = graph.to(device)
            encoded_graph = vgae_model.encode(graph.x, graph.edge_index).to("cpu")
            encoded_graph = torch.mean(encoded_graph, dim=1)
            encoded_graph = z_score_standardization(encoded_graph)

        # Encode point cloud data using PAE
        with torch.no_grad():
            point_cloud = point_cloud.to(device)
            encoded_point_cloud = pae_model.encode(point_cloud[None, :]).squeeze().to("cpu")
            encoded_point_cloud = z_score_standardization(encoded_point_cloud)
            
        # Define Linear Projections for each modality
        sequence_proj = nn.Linear(encoded_sequence.shape[-1], modality_dim).to(device)
        graph_proj = nn.Linear(encoded_graph.shape[-1], modality_dim).to(device)
        point_cloud_proj = nn.Linear(encoded_point_cloud.shape[-1], modality_dim).to(device)

        # Apply Linear Projections
        projected_sequence = sequence_proj(encoded_sequence)
        projected_graph = graph_proj(encoded_graph)
        projected_point_cloud = point_cloud_proj(encoded_point_cloud)

        attention_fusion = AttentionFusion(
            input_dims={"sequence": modality_dim, "graph": modality_dim, "point_cloud": modality_dim},
            shared_dim=shared_dim
        ).to(device)
        
        # Perform attention-based fusion using learned projections
        fused_data = attention_fusion(
            projected_sequence.to(device),
            projected_graph.to(device),
            projected_point_cloud.to(device),
        ).to("cpu")

        processed_data_list.append(fused_data)

    print("Attention Fusion Completed Successfully.")

    # Print the shapes of the encoded data
    print("Fused Data Shape: ", fused_data.shape)

    with open(f"./data/sequences/pickles/fusion.pkl", "wb") as f:
        pickle.dump(processed_data_list, f)
        
def load_data():
    # Load the preprocessed data
    with open("./data/sequences/pickles/fusion.pkl", "rb") as f:
        print("Loading data ...")
        dataset = pickle.load(f)
    print("Data loaded successfully.")    
    dataset = MultimodalDataset(dataset) # Create a custom dataset from the loaded data

    # Split the dataset into train, validation, and test sets
    train_ratio = 0.8
    valid_ratio = 0.1
    test_ratio = 0.1 # Placeholder to show the split ratios

    train_size = int(len(dataset) * train_ratio)
    valid_size = int(len(dataset) * valid_ratio)
    test_size = len(dataset) - train_size - valid_size

    print("Train dataset size:", train_size)
    print("Validation dataset size:", valid_size)
    print("Test dataset size:", test_size)

    # Split the dataset into train, validation, and test sets using random_split
    train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, valid_size, test_size]
    )

    batch_size = round(train_size * 0.1)

    # Create data loaders for training, validation, and test
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, valid_loader, test_loader

def train(train_data, valid_data, criterion, device):
    model = ConcreteAutoencoder(input_dim, shared_dim, latent_dim, temperature).to(device)

    # Define optimizer (Adam)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)
    
    # Training loop
    num_epochs = 20
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        
        for data in train_data:
            data = data.to(device)
            optimizer.zero_grad()

            # Forward pass with fused modalities
            reconstructed = model(data)
            loss = criterion(reconstructed, data)

            # Backpropagate the gradients
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        average_loss = total_loss / len(train_data)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {average_loss:.4f}")

    # Validate the model
    val_loss = validate(model, valid_data, criterion, device)
    print(f"Validation Loss: {val_loss:.4f}")

    # Save the trained model
    torch.save(model.state_dict(), "./model/concrete_autoencoder.pt")


def validate(model, valid_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in valid_loader:
            batch = batch.to(device)
            reconstructions = model(batch)
            loss = criterion(reconstructions, batch)
            total_loss += loss.item()

    average_loss = total_loss / len(valid_loader)
    return average_loss


def test(test_data, criterion, device):
    model = ConcreteAutoencoder(input_dim, shared_dim, latent_dim, temperature).to(device)
    model.load_state_dict(torch.load("./model/concrete_autoencoder.pt", weights_only=True))
    model.to(device)
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in test_data:
            batch = batch.to(device)
            reconstructions = model(batch)
            loss = criterion(reconstructions, batch)
            total_loss += loss.item()
            
    average_loss = total_loss / len(test_data)
    print(f"Test Loss: {average_loss:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Multimodal Fusion Model")
    parser.add_argument(
        "--mode",
        choices=["process", "train", "test", "all"],
        help="Select mode: process, train, test, or all",
        required=True,
    )
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if args.mode == "process":
        process(device)
        
    if args.mode != "process":
        train_data, valid_data, test_data = load_data()
        criterion = nn.MSELoss()

    if args.mode == "train":
        train(train_data, valid_data, criterion, device)

    if args.mode == "test":
        test(test_data, criterion, device)
    
    if args.mode == "all":
        process(device)
        train(train_data, valid_data, criterion, device)
        test(test_data, criterion, device)


if __name__ == "__main__":
    main()