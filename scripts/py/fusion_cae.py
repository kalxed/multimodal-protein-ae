import argparse
import os
import os.path as osp

import torch
import torch.nn as nn
import torch_geometric
from torch_geometric.loader import DataLoader
from torch_geometric.nn import VGAE

from mpae.nn import ConcreteAutoencoder, VariationalGCNEncoder
from mpae.nn.pae import PointAutoencoder
from mpae.utils import fuse_with_attention
from mpae.utils.data import SingleModeDataset
import mpae
# Concrete autoencoder parameters
input_dim = 640 * 3  # Input dimension after fusion
shared_dim = 640  # Shared dimension after fusion
latent_dim = 64  # Latent space size
temperature = .5  # Concrete distribution temperature

torch.serialization.add_safe_globals([torch_geometric.data.data.DataEdgeAttr, 
                                      torch_geometric.data.data.DataTensorAttr,
                                      torch_geometric.data.storage.GlobalStorage])

# Function for Z-score standardization
def z_score_standardization(tensor: torch.Tensor):
    mean = tensor.mean()
    std = tensor.std()
    if std != 0:
        standardized_tensor = (tensor - mean) / std
    else:
        standardized_tensor = tensor  # Handle the case when std is 0
    return standardized_tensor

def fuse_proteins(device: torch.device, vgae_model_path: str, pae_model_path:str, data_dir: str, protein_ids: list[str]):
    """perform attention-based fusion

    For each protein, fuse the graph, point cloud, and sequence into a single representation
    """
    
    vgae_model = VGAE(VariationalGCNEncoder(20, 10)).to(device)
    vgae_model.load_state_dict(torch.load(vgae_model_path, map_location=device, weights_only=True))
    pae_model = PointAutoencoder(640, 2048).to(device)
    pae_model.load_state_dict(torch.load(pae_model_path, map_location=device, weights_only=True))
    print("Pre-trained models loaded successfully.")

    graph_data = SingleModeDataset([osp.join(data_dir, "graphs", prot_id) for prot_id in protein_ids], device=device)
    seq_data = SingleModeDataset([osp.join(data_dir, "sequences", prot_id) for prot_id in protein_ids], device=device)
    cloud_data = SingleModeDataset([osp.join(data_dir, "pointclouds", prot_id) for prot_id in protein_ids], device=device)

    modality_dim = 640  # Dimension of each modality
    shared_dim = modality_dim * 3
    graph_loader = DataLoader(graph_data, batch_size=1, shuffle=False) 
    seq_loader   = DataLoader(seq_data, batch_size=1, shuffle=False)
    cloud_loader = DataLoader(cloud_data, batch_size=1, shuffle=False)
    res_dir = osp.join(data_dir, "fusion")
    os.makedirs(res_dir, exist_ok=True)
    n_bad = 0
    for i, (graph, cloud, seq) in enumerate(zip(graph_loader, cloud_loader, seq_loader)):
        protein_id = protein_ids[i]
        # do this because they are in batches
        graph = graph[0]
        cloud = cloud[0]
        seq = seq[0]

        
        # Perform attention-based fusion using learned projections
        try:
            fused_data = fuse_with_attention(graph=graph, tokenized_seq=seq, pointcloud=cloud, vgae_model=vgae_model, 
                                             pae_model=pae_model, device=device, shared_dim=shared_dim)
        except torch.OutOfMemoryError:
            print(f"Failed with protein {protein_id}")
            n_bad += 1
            continue
        
        torch.save(fused_data, osp.join(res_dir, protein_id))
        if ((i+1) % 1000) == 0:
            print(f"Fused {i+1} files")
    
    if n_bad > 0:
        print(f"Skipped {n_bad} proteins due to memory issues")

    print("Attention Fusion Completed Successfully.")

def get_loaders(protein_list: list[str], data_dir: str, batch_size, train_size=0.7, val_size=0.1, test_size=0.2):
    # Load the preprocessed data
    print(f"loading data from {osp.join(data_dir, 'fusion')}")
    dataset = SingleModeDataset([osp.join(data_dir, "fusion", p) for p in protein_list]) # Create a custom dataset from the loaded data

    # Split the dataset into train, validation, and test sets using random_split
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size], torch.Generator().manual_seed(1234)
    )
    print(f"train data length: {len(train_dataset)}")
    print(f"val data length: {len(val_dataset)}")
    print(f"test data length: {len(test_dataset)}")

    # Create data loaders for training, validation, and test
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, valid_loader, test_loader

@torch.no_grad
def validate(model, val_loader, criterion, device):
    if len(val_loader) == 0:
        return torch.nan
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            reconstructions = model(batch)
            loss = criterion(reconstructions, batch)
            total_loss += loss.item()

    average_loss = total_loss / len(val_loader)
    return average_loss

@torch.no_grad
def test(test_loader: torch.utils.data.DataLoader, model_path, criterion, device):
    if len(test_loader) == 0:
        average_loss = torch.nan
    else:
        model = ConcreteAutoencoder(input_dim, shared_dim, latent_dim, temperature).to(device)
        model.load_state_dict(torch.load(model_path, weights_only=True))
        model.to(device)
        model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)
                reconstructions = model(batch)
                loss = criterion(reconstructions, batch)
                total_loss += loss.item()
                
        average_loss = total_loss / len(test_loader)

    print(f"Test Loss: {average_loss:.4f}")

def train(train_loader, val_loader, criterion, device, num_epochs, model_path):
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
            total_loss += loss.item()
            optimizer.step()

        if epoch % 5 == 0:
            train_loss = total_loss / len(train_loader)
            val_loss = validate(model, val_loader, criterion, device)
            print(f"Epoch {epoch}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss}")

    # Save the trained model
    torch.save(model.state_dict(), model_path)

def main():
    parser = argparse.ArgumentParser(description="Multimodal Fusion Model")
    parser.add_argument(
        "--mode",
        choices=["process", "train", "test", "all"],
        help="Select mode: process, train, test, or all",
        required=True,
    )
    parser.add_argument("--model-path", default="models/CAE.pt", type=str,dest="model_path",
                        help="where store the trained model, or where to load the model from for testing")
    parser.add_argument("--vgae-path", default="models/VGAE.pt", help="file containing trained VGAE model", type=str, dest="vgae_path")
    parser.add_argument("--pae-path", default="models/PAE.pt", help="file containing trained PAE model", type=str, dest="pae_path")
    parser.add_argument("--id-file", default="proteins", help="file containing all the protein ids",type=str, dest="id_file")
    parser.add_argument("--data-dir", default="data/",help = "directory containing the graph files", type=str, dest="data_dir")
    parser.add_argument("--epochs", default=100, help="number of epochs to train for. only applicable when mode=train", type=int)
    parser.add_argument("--batch-size", default=64, help="batch size for training", type=int, dest="batch_size")

    args = parser.parse_args()
    model_path = args.model_path
    vgae_path = args.vgae_path
    pae_path = args.pae_path
    data_dir = args.data_dir
    id_file = args.id_file
    
    num_epochs = args.epochs
    batch_size = args.batch_size

    with open(id_file, 'r') as f:
        protein_ids = [pid.strip() for pid in f.readlines()]

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"using device {device}")

    if args.mode == "process":
        if os.getenv("SLURM_ARRAY_TASK_COUNT"):
            ntasks = int(os.environ.get("SLURM_ARRAY_TASK_COUNT", "1"))
            task_idx = int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))

            # determine which files to process based on what task number we are
            files_to_process = len(protein_ids) // ntasks
            first_file = files_to_process * task_idx
            last_file = len(protein_ids) if task_idx == (ntasks - 1) else (first_file + files_to_process)
            protein_ids = protein_ids[first_file:last_file]
        fuse_proteins(device=device, vgae_model_path= vgae_path, pae_model_path=pae_path, data_dir=data_dir, protein_ids=protein_ids)

    if args.mode != "process":
        train_loader, val_loader, test_loader = get_loaders(protein_ids, data_dir, batch_size)
        criterion = nn.MSELoss()

    if args.mode == "train":
        train(train_loader=train_loader, val_loader=val_loader, criterion=criterion, device=device, num_epochs=num_epochs, model_path=model_path)

    if args.mode == "test":
        test(test_loader=test_loader, model_path=model_path, criterion=criterion, device=device)
    
    if args.mode == "all":
        fuse_proteins(device=device, vgae_model_path=vgae_path, pae_model_path=pae_path, data_dir=data_dir, protein_ids=protein_ids)
        train(train_loader=train_loader, val_loader=val_loader, criterion=criterion, device=device, num_epochs=num_epochs, model_path=model_path)
        test(test_loader=test_loader, model_path=model_path, criterion=criterion, device=device)


if __name__ == "__main__":
    main()
