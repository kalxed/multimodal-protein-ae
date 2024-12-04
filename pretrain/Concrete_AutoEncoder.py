import argparse
import pickle
import sys
import torch
import torch.nn as nn
import glob

from torch_geometric.loader import DataLoader
from torch.utils.data import Dataset

sys.path.append(".")
from model.Concrete_Autoencoder import ConcreteAutoencoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MultimodalDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # Assuming fused data is a dictionary, modify if needed
        data_point = self.data[index]
        return data_point  # Returning as-is for now

# Load the preprocessed data

# Uncomment to load the point cloud data from all .pt files
pointcloud_files = glob.glob('./data/multimodal/*.pt')
dataset = []

for file in pointcloud_files:
    data = torch.load(file, weights_only=False)
    dataset.append(data)

# Uncomment to load the fused data from a single pickle file
# with open("./data/pickles/fusion.pkl", "rb") as f:
#     print("Loading data ...")
#     dataset = pickle.load(f)
print("Data loaded successfully.")

# Create a custom dataset from the loaded data
dataset = MultimodalDataset(dataset)

# Split the dataset into train, validation, and test sets
train_ratio = 0.8
valid_ratio = 0.1
test_ratio = 0.1

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

# Initialize the model with the temperature parameter
input_dim = 640 * 3  # Input dimension after fusion
shared_dim = 640  # Shared dimension after fusion
latent_dim = 64  # Latent space size
temperature = .5  # Concrete distribution temperature

model = ConcreteAutoencoder(input_dim, shared_dim, latent_dim, temperature).to(device)

# Define loss function (MSE) and optimizer (Adam)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

def train(model, train_data, optimizer, criterion, device):
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
    return average_loss


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


def test(model, test_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            reconstructions = model(batch)
            loss = criterion(reconstructions, batch)
            total_loss += loss.item()
            
    average_loss = total_loss / len(test_loader)
    return average_loss


def eval(model, data, criterion, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in data:
            batch = batch.to(device)
            reconstructions = model(batch)
            loss = criterion(reconstructions, batch)
            total_loss += loss.item()
            
    average_loss = total_loss / len(data)
    return average_loss

def main():
    parser = argparse.ArgumentParser(description="Concrete Autoencoder")
    parser.add_argument(
        "--mode",
        choices=["train", "test"],
        help="Select mode: train or test",
        required=True,
    )
    args = parser.parse_args()

    if args.mode == "train":
        num_epochs = 20
        for epoch in range(num_epochs):
            average_loss = train(model, train_loader, optimizer, criterion, device)
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {average_loss:.4f}")

        # Validate the model
        val_loss = eval(model, valid_loader, criterion, device)
        print(f"Validation Loss: {val_loss:.4f}")

        # Save trained model
        torch.save(model.state_dict(), "./data/models/Concrete.pt")

    elif args.mode == "test":
        model.load_state_dict(torch.load("./data/models/Concrete.pt", weights_only=True))
        test_loss = eval(model, test_loader, criterion, device)
        print(f"Test Loss: {test_loss:.4f}")


if __name__ == "__main__":
    main()