import torch
import torch.nn as nn

class AttentionFusion(nn.Module):
    def __init__(self, input_dims, shared_dim):
        super().__init__()
        self.projections = nn.ModuleDict({
            "sequence": nn.Linear(input_dims["sequence"], shared_dim),
            "graph": nn.Linear(input_dims["graph"], shared_dim),
            "point_cloud": nn.Linear(input_dims["point_cloud"], shared_dim),
        })
        self.attention = nn.MultiheadAttention(embed_dim=shared_dim, num_heads=4)
        self.fc_out = nn.Linear(shared_dim, shared_dim)

    def forward(self, sequence, graph, point_cloud):
        # Project all inputs to shared_dim
        seq_proj = self.projections["sequence"](sequence)
        graph_proj = self.projections["graph"](graph)
        pc_proj = self.projections["point_cloud"](point_cloud)

        # Stack the projected features
        tokens = torch.stack([seq_proj, graph_proj, pc_proj], dim=0)  # Shape: (3, batch_size, shared_dim)

        # Compute attention
        attn_output, _ = self.attention(tokens, tokens, tokens)  # Shape: (3, batch_size, shared_dim)

        # Aggregate modalities and produce fused output
        fusion = attn_output.mean(dim=0)  # Average over modalities
        fusion = self.fc_out(fusion)
        return fusion
