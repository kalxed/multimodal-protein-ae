import argparse
import math
import os
import os.path as osp

import torch.optim as optim
from torch_geometric.loader import DataLoader

from mpae.nn.pae import *
from mpae.utils.data import SingleModeDataset


# Function to apply random rotations to input data
def random_rotation_matrix(batch_size):
    # Generate random angles for rotation
    angles = torch.rand((batch_size, 3)) * 2 * math.pi

    # Compute cosines and sines of the angles
    cosines = torch.cos(angles)
    sines = torch.sin(angles)

    # Create the rotation matrices for each axis
    Rx = torch.stack([torch.tensor([[1, 0, 0],
                                    [0, c, -s],
                                    [0, s, c]], dtype=torch.float32) for c, s in zip(cosines[:, 0], sines[:, 0])])

    Ry = torch.stack([torch.tensor([[c, 0, s],
                                    [0, 1, 0],
                                    [-s, 0, c]], dtype=torch.float32) for c, s in zip(cosines[:, 1], sines[:, 1])])

    Rz = torch.stack([torch.tensor([[c, -s, 0],
                                    [s, c, 0],
                                    [0, 0, 1]], dtype=torch.float32) for c, s in zip(cosines[:, 2], sines[:, 2])])

    # Multiply the rotation matrices to get the final rotation matrices
    rotation_matrices = torch.matmul(Rz, torch.matmul(Ry, Rx))

    return rotation_matrices

# Initialize the Chamfer Distance loss
chamfer_distance = ChamferDistance()

# Training function
def train(pae_model, train_loader, optimizer, device):
    pae_model.train()
    total_loss = 0.0

    for data in train_loader:
        data = data.to(device)
        rotation_matrix = random_rotation_matrix(data.size(0)).to(device)
        data = torch.matmul(rotation_matrix, data)

        optimizer.zero_grad()
        encoding = pae_model.encode(data)

        restoration = pae_model.decode(encoding)
        loss = chamfer_distance(data, restoration)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    average_loss = total_loss / len(train_loader)
    return average_loss

# Validation function
@torch.no_grad
def validation(pae_model, valid_loader, device):
    pae_model.eval()
    total_val_loss = 0.0

    with torch.no_grad():
        for data in valid_loader:
            data = data.to(device)
            encoding = pae_model.encode(data)
            restoration = pae_model.decode(encoding)

            # Calculate the Chamfer distance loss on the validation set
            val_loss = chamfer_distance(data, restoration)
            total_val_loss += val_loss.item()

    average_val_loss = total_val_loss / len(valid_loader)
    return average_val_loss

# Testing function
@torch.no_grad
def test(pae_model, test_loader, device):
    pae_model.eval()
    total_test_loss = 0.0

    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            encoding = pae_model.encode(data)
            restoration = pae_model.decode(encoding)

            # Calculate the Chamfer distance loss on the test set
            test_loss = chamfer_distance(data, restoration)
            total_test_loss += test_loss.item()

    average_test_loss = total_test_loss / len(test_loader)
    return average_test_loss

def main():
    parser = argparse.ArgumentParser(description="Point Cloud Auto-Encoder (PAE)")
    parser.add_argument("--mode", choices=["train", "test"], help="Select mode: train or test", required=True)
    parser.add_argument("--model-path", default="models/points.pt", type=str,dest="model_path",
                        help="where store the trained model, or where to load the model from for testing")
    parser.add_argument("--id-file", default="proteins", help="file containing all the protein ids",type=str, dest="id_file")
    parser.add_argument("--data-dir", default="data/pointclouds",help = "directory containing the pointcloud files", type=str, dest="data_dir")
    parser.add_argument("--epochs", default=100, help="number of epochs to train for. only applicable when mode=train", type=int)
    parser.add_argument("--batch-size", default=128, help="batch size for training", type=int, dest="batch_size")

    args = parser.parse_args()
    model_path = args.model_path
    pointcloud_dir = args.data_dir
    id_file = args.id_file

    num_epochs = args.epochs
    batch_size = args.batch_size

    os.makedirs(osp.dirname(model_path), exist_ok=True)

    # Define the dimension of the representation vector and the number of points
    k = 640
    num_points = 2048

    with open(id_file, "r") as f:
        fnames = [osp.join(pointcloud_dir, fname.strip()) for fname in f.readlines()]
    dataset = SingleModeDataset(fnames=fnames)
    # Split the dataset into train, validation, and test sets
    train_ratio = 0.7
    val = 0.1
    test_ratio = 0.2

    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_ratio, val, test_ratio], torch.Generator().manual_seed(1234))
    
    print(f"train data length: {len(train_dataset)}")
    print(f"val data length: {len(val_dataset)}")
    print(f"test data length: {len(test_dataset)}")

    # Create data loaders for train, validation, and test sets
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    pae_model = PointAutoencoder(k, num_points)

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"using device {device}")

    pae_model = pae_model.to(device)

    optimizer = optim.Adam(pae_model.parameters(), lr=0.001)

    if args.mode == "train":
        # Training mode
        for epoch in range(1, num_epochs + 1):
            train_loss = train(pae_model, train_loader, optimizer, device) 
            if epoch % 5 == 0:
                val_loss = validation(pae_model, val_loader, device)
                print(f'Epoch [{epoch}/{num_epochs}], Train Loss: {train_loss:.4f}, Valid Loss: {val_loss:.4f}')

        # Save the PAE model
        torch.save(pae_model.state_dict(), model_path)
        print("Model saved")

    elif args.mode == "test":
        # Test mode
        # Load the saved model
        pae_model = PointAutoencoder(k, num_points).to(device)
        pae_model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        # Evaluate the model on the test dataset
        test_loss = test(pae_model, test_loader, device)
        print(f"Average Chamfer Distance on Test Set: {test_loss:.4f}")

if __name__ == "__main__":
    main()
