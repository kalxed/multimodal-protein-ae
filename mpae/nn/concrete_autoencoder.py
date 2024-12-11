import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


class ConcreteAutoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dim=640, dropout_rate=0.1, attention=False):
        super(ConcreteAutoencoder, self).__init__()
        
        # Use hidden_dim if provided, else default to input_dim
        hidden_dim = hidden_dim or input_dim

        # Encoder: Mapping input to latent space
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(dropout_rate),  # Add dropout after ReLU
            nn.Linear(hidden_dim, latent_dim)
        )

        # Decoder: Mapping latent space back to original input space
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(dropout_rate),  # Add dropout after ReLU
            nn.Linear(hidden_dim, input_dim)
        )

        self.use_attention = attention
        if (attention):
            self.attention = nn.MultiheadAttention(embed_dim=input_dim, num_heads=4)

        # Apply weight initialization
        self.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            init.kaiming_normal_(m.weight, a=0.01, mode='fan_in', nonlinearity='leaky_relu')
            if m.bias is not None:
                init.zeros_(m.bias)

    def encode(self, fused_rep):
        """
        Encodes the input into the latent space.
        """
        if self.use_attention:
            h, _ = self.attention(fused_rep, fused_rep, fused_rep)
        else:
            h = fused_rep
        encoded = self.encoder(h)
        return encoded

    def forward(self, fused_rep):
        """
        fused_rep: The representation coming from the attention fusion of multiple modalities
        """
        # Encoder produces logits
        encoded = self.encode(fused_rep)
 
        # Decoder reconstructs input from the latent representation
        reconstructed = self.decoder(encoded)

        return reconstructed
