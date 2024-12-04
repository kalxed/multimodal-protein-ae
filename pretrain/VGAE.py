
import argparse
import os
import os.path as osp
import pickle
import random
import sys
import warnings

import torch
import torch_geometric
from torch.utils.data import random_split
from torch_geometric.data import DataLoader
from torch_geometric.loader import DataLoader
sys.path.append('.')
from model.vgae import *

from datasets import SingleModeDataset
# from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr

torch.serialization.add_safe_globals([torch_geometric.data.data.DataEdgeAttr, 
                                      torch_geometric.data.data.DataTensorAttr,
                                      torch_geometric.data.storage.GlobalStorage])

# Function to train the VGAE model
def train(model, train_loader, optimizer, device):
    model.train()
    total_loss = 0
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        z = model.encode(batch.x, batch.edge_index)
        loss = model.recon_loss(z, batch.edge_index, batch.neg_edge_index)
        loss.backward()
        optimizer.step()
        total_loss += loss
    return total_loss / len(train_loader)

# Function to perform validation on the VGAE model
@torch.no_grad
def validate_model(model, val_loader, device):
    model.eval()
    val_loss = 0
    for batch in val_loader:
        batch = batch.to(device)
        z = model.encode(batch.x, batch.edge_index)
        loss = model.recon_loss(z, batch.edge_index, batch.neg_edge_index)
        val_loss += loss.item()
    val_loss /= len(val_loader)
    return val_loss

# Function to test the VGAE model
@torch.no_grad
def test_model(model, test_loader, device):
    model.eval()
    AUC = 0
    AP = 0
    n=len(test_loader)
    for i, batch in enumerate(test_loader):
        batch = batch.to(device)
        z = model.encode(batch.x, batch.edge_index)
        auc, ap = model.test(z, batch.edge_index, batch.neg_edge_index)
        AUC += auc
        AP += ap
    return AUC/n, AP/n

def main():
    parser = argparse.ArgumentParser(description="Variational Graph Autoencoder (VGAE)")
    parser.add_argument("--mode", choices=["train", "test"], help="Select mode: train or test", required=True)
    parser.add_argument("--model-path", default="model.pt", type=str,dest="model_path",
                        help="where store the trained model, or where to load the model from for testing")
    parser.add_argument("--id-file", default="proteins", help="file containing all the protein ids",type=str, dest="id_file")
    parser.add_argument("--data-dir", default="data/graphs",help = "directory containing the graph files", type=str, dest="data_dir")
    parser.add_argument("--epochs", default=100, help="number of epochs to train for. only applicable when mode=train", type=int)
    parser.add_argument("--batch-size", default=64, help="batch size for training", type=int, dest="batch_size")

    args = parser.parse_args()
    model_path = args.model_path
    graph_dir = args.data_dir
    id_file = args.id_file

    num_epochs = args.epochs
    batch_size = args.batch_size
    # Define the output dimensions for the model
    out_channels = 10
    num_features = 20

    with open(id_file, "r") as f:
        fnames = [osp.join(graph_dir, fname.strip()) for fname in f.readlines()]

    graph_ds = SingleModeDataset(fnames=fnames)
    train_set, val_set, test_set = random_split(graph_ds, [0.0, 0.0, 1.0 ], torch.Generator().manual_seed(1234))

    # Create data loaders for train, test, and validation sets
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    

    # Create an instance of VGAE using the VariationalGCNEncoder
    vgae_model = VGAE(VariationalGCNEncoder(num_features, out_channels))
    # Check if GPU is available, and move the model to the appropriate device
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"using device {device}")
    vgae_model = vgae_model.to(device)

    # Create an optimizer for the VGAE (Variational Graph Autoencoder) model
    optimizer = torch.optim.Adam(vgae_model.parameters(), lr=0.001)
    
    
    if args.mode == "train":
        # Training mode
        for epoch in range(1, num_epochs + 1):
            train_loss = train(vgae_model, train_loader, optimizer, device)

            
            if epoch % 5 == 0:
                val_loss = validate_model(vgae_model, val_loader, device)
                print(f'Epoch [{epoch}/{num_epochs}], Train Loss: {train_loss:.4f}, Valid Loss: {val_loss:.4f}')

        # Save the VGAE model
        torch.save(vgae_model, model_path)
        print("Model saved")

    elif args.mode == "test":
        # Test mode
        # Load the saved model
        vgae_model = torch.load(model_path)

        # Test the model on the test dataset
        AUC, AP = test_model(vgae_model, test_loader, device)
        print(f"AUC: {AUC}, AP: {AP}")

if __name__ == "__main__":
    main()
