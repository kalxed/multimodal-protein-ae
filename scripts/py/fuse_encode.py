import argparse
import os
import os.path as osp

import torch
import torch.nn as nn
import torch_geometric
import transformers
from torch_geometric.loader import DataLoader

from mpae.nn import *
from mpae.utils.data import SingleModeDataset

# Concrete autoencoder parameters
input_dim = 640 * 3  # Input dimension after fusion
shared_dim = 640  # Shared dimension after fusion
latent_dim = 64  # Latent space size
temperature = .5  # Concrete distribution temperature

torch.serialization.add_safe_globals([torch_geometric.data.data.DataEdgeAttr, 
                                      torch_geometric.data.data.DataTensorAttr,
                                      torch_geometric.data.storage.GlobalStorage])

# Function for Z-score standardization
def z_score_standardization(tensor):
    mean = tensor.mean()
    std = tensor.std()
    if std != 0:
        standardized_tensor = (tensor - mean) / std
    else:
        standardized_tensor = tensor  # Handle the case when std is 0
    return standardized_tensor

def fuse_attention(device: torch.device, models_dir: str, data_dir: str, protein_ids: list[str]):
    """perform attention-based fusion
    """
    
    vgae_model = torch.load(osp.join(models_dir, f"VGAE.pt"), map_location=device, weights_only=False)
    pae_model = torch.load(osp.join(models_dir, f"PAE.pt"), map_location=device, weights_only=False)
    model_token = "facebook/esm2_t30_150M_UR50D"
    esm_model = transformers.AutoModelForMaskedLM.from_pretrained(model_token)
    esm_model = esm_model.to(device)
    print("Pre-trained models loaded successfully.")

    graph_data = SingleModeDataset([osp.join(data_dir, "graphs", prot_id) for prot_id in protein_ids])
    seq_data = SingleModeDataset([osp.join(data_dir, "sequences", prot_id) for prot_id in protein_ids])
    cloud_data = SingleModeDataset([osp.join(data_dir, "pointclouds", prot_id) for prot_id in protein_ids])

    modality_dim = 640  # Dimension of each modality
    shared_dim = modality_dim * 3
    graph_loader = DataLoader(graph_data, batch_size=1, shuffle=False) 
    seq_loader   = DataLoader(seq_data, batch_size=1, shuffle=False)
    cloud_loader = DataLoader(cloud_data, batch_size=1, shuffle=False)
    res_dir = osp.join(data_dir, "fusion")
    os.makedirs(res_dir, exist_ok=True)
    for i, (graph, cloud, seq) in enumerate(zip(graph_loader, cloud_loader, seq_loader)):
        protein_id = protein_ids[i]
        # do this because they are in batches
        graph = graph[0]
        cloud = cloud[0]
        seq = seq[0]

        # Encode sequence data using ESM
        with torch.no_grad():
            seq = seq.to(device)
            encoded_sequence = esm_model(seq, output_hidden_states=True)["hidden_states"
            ][-1][0, -1]#.to("cpu")
            encoded_sequence = z_score_standardization(encoded_sequence)

        # Encode graph data using VGAE
        with torch.no_grad():
            graph = graph.to(device)
            encoded_graph = vgae_model.encode(graph.x, graph.edge_index)#.to("cpu")
            encoded_graph = torch.mean(encoded_graph, dim=1)
            encoded_graph = z_score_standardization(encoded_graph)

        # Encode point cloud data using PAE
        with torch.no_grad():
            cloud = cloud.to(device)
            encoded_point_cloud = pae_model.encode(cloud[None, :]).squeeze()#.to("cpu")
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
        )
        
        torch.save(fused_data, osp.join(res_dir, protein_id))

    print("Attention Fusion Completed Successfully.")

def load_fused_data(protein_list: list[str], data_dir: str, batch_size, train_size=0.7, val_size=0.1, test_size=0.2, ):
    # Load the preprocessed data
    dataset = SingleModeDataset([osp.join(data_dir, "fusion", p) for p in protein_list]) # Create a custom dataset from the loaded data

    # Split the dataset into train, validation, and test sets using random_split
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size], torch.Generator().manual_seed(1234)
    )

    # Create data loaders for training, validation, and test
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, valid_loader, test_loader

@torch.no_grad
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

@torch.no_grad
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

def train(train_loader, val_loader, criterion, device, num_epochs, model_dir):
    model = ConcreteAutoencoder(input_dim, shared_dim, latent_dim, temperature).to(device)

    # Define optimizer (Adam)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)
    # Training loop
    for epoch in range(1, num_epochs+1):
        model.train()
        total_loss = 0.0
        
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            # Forward pass with fused modalities
            reconstructed = model(batch)
            loss = criterion(reconstructed, batch)

            # Backpropagate the gradients
            loss.backward()
            optimizer.step()

        if epoch % 5 == 0:
            train_loss = total_loss / len(train_loader)
            val_loss = validate(model, val_loader, criterion, device)
            print(f"Epoch {epoch}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss}")

    # Save the trained model
    torch.save(model.state_dict(), osp.join(model_dir, "FUSE.pt"))

def main():
    parser = argparse.ArgumentParser(description="Multimodal Fusion Model")
    parser.add_argument(
        "--mode",
        choices=["process", "train", "test", "all"],
        help="Select mode: process, train, test, or all",
        required=True,
    )
    parser.add_argument("--model-dir", default="models", type=str,dest="model_dir",
                        help="directory containing the PAE and VGAE models and/or where to store the trained model")
    parser.add_argument("--id-file", default="proteins", help="file containing all the protein ids",type=str, dest="id_file")
    parser.add_argument("--data-dir", default="data/",help = "directory containing the graph files", type=str, dest="data_dir")
    parser.add_argument("--epochs", default=100, help="number of epochs to train for. only applicable when mode=train", type=int)
    parser.add_argument("--batch-size", default=64, help="batch size for training", type=int, dest="batch_size")

    args = parser.parse_args()
    model_dir = args.model_dir
    data_dir = args.data_dir
    id_file = args.id_file
    
    num_epochs = args.epochs
    batch_size = args.batch_size

    with open(id_file, 'r') as f:
        protein_ids = [pid.strip() for pid in f.readlines()]

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    
    if args.mode == "process":
        fuse_attention(device=device, models_dir=model_dir, data_dir=data_dir, protein_ids=protein_ids)

    if args.mode != "process":
        train_data, valid_data, test_data = load_fused_data(protein_ids, data_dir, batch_size)
        criterion = nn.MSELoss()

    if args.mode == "train":
        train(train_loader=train_data, val_loader=valid_data, criterion=criterion, device=device, num_epochs=num_epochs, model_dir=model_dir)

    if args.mode == "test":
        test(test_data, criterion, device)
    
    if args.mode == "all":
        fuse_attention(device=device, models_dir=model_dir, data_dir=data_dir, protein_ids=protein_ids)
        train(train_loader=train_data, val_loader=valid_data, criterion=criterion, device=device, num_epochs=num_epochs, model_dir=model_dir)
        test(test_data=test_data, criterion=criterion, device=device)


if __name__ == "__main__":
    main()