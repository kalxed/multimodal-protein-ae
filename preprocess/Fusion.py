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
from model.vgae import *
from model.PAE import *
import glob
from model.Attention import AttentionFusion
from downstreamtasks.utils import *

class MultimodalDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data_point = self.data[index]
        return data_point  # Returning as-is for now
    
# Function to process and adjust encoded graph data to a fixed size
def process_encoded_graph(encoded_graph, edge_index, fixed_size=640, feature_dim=10):
    num_nodes = encoded_graph.size(0)
    if num_nodes > fixed_size:
        ratio = fixed_size / num_nodes
        with torch.no_grad():
            pooling_layer = TopKPooling(in_channels=feature_dim, ratio=ratio)
            pooled_x, edge_index, edge_attr, batch, perm, score = pooling_layer(encoded_graph, edge_index)
        processed_encoded_graph = pooled_x
    else:
        padding_size = fixed_size - num_nodes
        zero_padding = torch.zeros(padding_size, feature_dim)
        processed_encoded_graph = torch.cat((encoded_graph, zero_padding), dim=0)

    return processed_encoded_graph
    
# Concrete autoencoder parameters
input_dim = 640 * 3  # Input dimension after fusion
modality_dim = 640  # Dimension of each modality
shared_dim = 640  # Shared dimension after fusion

# Input dimensions for the AttentionFusion model
input_dims = {"sequence": modality_dim, "graph": modality_dim, "point_cloud": modality_dim}  # Input feature dims for each modality
    
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
    vgae_model, pae_model, esm_model, _ = load_models()

    # Directories for graph, sequence, and point cloud data
    graph_dir = "./data/graphs/"
    pointcloud_dir = "./data/pointclouds/"
    sequence_dir = "./data/sequences/"

    # Get sorted file lists to ensure alignment
    graph_files = sorted(os.listdir(graph_dir))
    pointcloud_files = sorted(os.listdir(pointcloud_dir))
    sequence_files = sorted(os.listdir(sequence_dir))

    shared_dim = 640
    processed_data_list = []
    
    multimodal_files = sorted(os.listdir("./data/multimodal/"))

    for graph_file, pointcloud_file, sequence_file in tqdm(
        zip(graph_files, pointcloud_files, sequence_files),
        total=len(sequence_files),
        desc="Processing modalities"
    ):
        
        # Load graph, point cloud, and sequence data
        # Note: using graph file name for pointcloud because we do not want to process
        # proteins with non-standard amino acids.
        graph = torch.load(os.path.join(graph_dir, graph_file), weights_only=False)
        point_cloud = torch.load(os.path.join(pointcloud_dir, graph_file), weights_only=False)
        sequence = torch.load(os.path.join(sequence_dir, graph_file), weights_only=False)

        with torch.no_grad():
            sequence = sequence.to(device)
            encoded_sequence = esm_model(sequence, output_hidden_states=True)['hidden_states'][-1][0, -1].to("cpu")
            encoded_sequence = z_score_standardization(encoded_sequence)

        # Encode graph data using VGAE
        with torch.no_grad():
            graph = graph.to(device)
            encoded_graph = vgae_model.encode(graph.x, graph.edge_index).to("cpu")
            encoded_graph = process_encoded_graph(encoded_graph, graph.edge_index)
            encoded_graph = torch.mean(encoded_graph, dim=1)
            encoded_graph = z_score_standardization(encoded_graph)

        # Encode point cloud data using PAE
        with torch.no_grad():
            point_cloud = point_cloud.to(device)
            encoded_point_cloud = pae_model.encode(point_cloud[None, :]).squeeze().to("cpu")
            encoded_point_cloud = z_score_standardization(encoded_point_cloud)
        
        # input_dims = {"sequence": encoded_sequence.shape[0], "graph": encoded_graph.shape[0], "point_cloud": encoded_point_cloud.shape[0]}
        # attention_fusion = AttentionFusion(input_dims, shared_dim)
        
        # # Ensure all modalities are 2D tensors (batch_size, shared_dim) before passing to attention
        # encoded_sequence = encoded_sequence.unsqueeze(0) if encoded_sequence.dim() == 1 else encoded_sequence
        # encoded_graph = encoded_graph.unsqueeze(0) if encoded_graph.dim() == 1 else encoded_graph
        # encoded_point_cloud = encoded_point_cloud.unsqueeze(0) if encoded_point_cloud.dim() == 1 else encoded_point_cloud

        # fused_rep, _ = attention_fusion(encoded_sequence, encoded_graph, encoded_point_cloud)
        fused_rep = torch.cat((encoded_sequence, encoded_graph, encoded_point_cloud), dim=0)
        if fused_rep.size(0) == 1920:
        
        # Uncomment to save each fused data object individually
            fused_data_filename = f"./data/test-multimodal/{graph_file.split('.')[0]}.pt"
            torch.save(fused_rep, fused_data_filename)
        
        # Uncomment next three lines to save the processed data list as a pickle file
        # processed_data_list.append(projected_data)

    # with open(f"./data/pickles/fusion.pkl", "wb") as f:
    #     pickle.dump(processed_data_list, f)
    
    print("Attention Fusion Completed Successfully.")
        
def load_data():
    # Load the preprocessed data
    
    # Uncomment to load the fused data from a single pickle file
    # with open("./data/pickles/fusion.pkl", "rb") as f:
    #     print("Loading data ...")
    #     dataset = pickle.load(f)
    
    # Uncomment to load data from all .pt files
    pointcloud_files = glob.glob('./data/multimodal/*.pt')
    dataset = []

    for file in pointcloud_files:
        data = torch.load(file, weights_only=False)
        dataset.append(data)    

    dataset = MultimodalDataset(dataset) # Create a custom dataset from the loaded data
    print("Data loaded successfully.")

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
    model = AttentionFusion(input_dims, shared_dim).to(device)

    # Define optimizer (Adam)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)
    
    # Training loop
    num_epochs = 20
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        
        for batch in train_data:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            # Encode and decode
            fused_representation, attn_weights = model.encode(batch[:, :modality_dim], 
                                                              batch[:, modality_dim:2*modality_dim], 
                                                              batch[:, 2*modality_dim:])
            reconstructed_data = model.decode(fused_representation)
            
            # Calculate loss (comparing reconstructed data to original input)
            loss = criterion(reconstructed_data, batch)
            
            # Backpropagation
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # Log average loss
        average_loss = total_loss / len(train_data)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {average_loss:.4f}")
    
    # Validate the model after training
    val_loss = validate(model, valid_data, criterion, device)
    print(f"Validation Loss: {val_loss:.4f}")
    
    # Save the trained model
    torch.save(model.state_dict(), "./data/models/attention_fusion.pt")
    print("Model saved")


def validate(model, valid_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in valid_loader:
            batch = batch.to(device)
            fused_representation, _ = model.encode(batch[:, :modality_dim], 
                                                              batch[:, modality_dim:2*modality_dim], 
                                                              batch[:, 2*modality_dim:])
            reconstructed_data = model.decode(fused_representation)
            loss = criterion(reconstructed_data, batch)
            total_loss += loss.item()

    average_loss = total_loss / len(valid_loader)
    return average_loss


def test(test_data, criterion, device):
    model = AttentionFusion(input_dims, shared_dim).to(device)
    model.load_state_dict(torch.load("./data/models/attention_fusion.pt", weights_only=True))
    model.to(device)
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in test_data:
            batch = batch.to(device)
            fused_representation, _ = model.encode(batch[:, :modality_dim], 
                                                              batch[:, modality_dim:2*modality_dim], 
                                                              batch[:, 2*modality_dim:])
            reconstructed_data = model.decode(fused_representation)
            loss = criterion(reconstructed_data, batch)
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