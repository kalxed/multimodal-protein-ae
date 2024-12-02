import pickle
import os
import sys
from tqdm import tqdm

sys.path.append(".")

from model.ESM import esm_model
from model.VGAE import *
from model.PAE import *
from model.Attention import AttentionFusion

# Check if GPU is available
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using GPU:", torch.cuda.get_device_name(0))
else:
    device = torch.device("cpu")
    print("No GPU available, using CPU.")

# Load pre-trained models
os.path.dirname(os.path.abspath(__file__))
vgae_model = torch.load(f"./data/models/VGAE.pt", map_location=device, weights_only=False)
pae_model = torch.load(f"./data/models/PAE.pt", map_location=device, weights_only=False)
esm_model = esm_model.to(device)
print("Pre-trained models loaded successfully.")

data_folder = "./data/sequences/pickles"

# Load preprocessed data
with open(f"{data_folder}/graphs.pkl", "rb") as f:
    print("Loading graph data ...")
    graph_data = pickle.load(f)
print("Graph data loaded successfully.")

with open(f"{data_folder}/pointclouds.pkl", "rb") as f:
    print("Loading point cloud data ...")
    point_cloud_data = pickle.load(f)
print("Point Cloud data loaded successfully.")

with open(f"{data_folder}/sequences.pkl", "rb") as f:
    print("Loading sequence data ...")
    sequence_data = pickle.load(f)
print("Sequence data loaded successfully.")

# Function for Z-score standardization
def z_score_standardization(tensor):
    mean = tensor.mean()
    std = tensor.std()
    if std != 0:
        standardized_tensor = (tensor - mean) / std
    else:
        standardized_tensor = tensor  # Handle the case when std is 0
    return standardized_tensor

modality_dim = 640  # Dimension of each modality
shared_dim = modality_dim * 3
processed_data_list = []

# Assuming `processed_data_list` is already defined
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

with open(f"{data_folder}/fusion.pkl", "wb") as f:
    pickle.dump(processed_data_list, f)
