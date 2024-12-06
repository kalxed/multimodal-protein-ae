import torch
import torch.nn as nn
import torch.nn.functional as F

# Concrete Distribution with Straight-Through Estimator
class ConcreteDistribution(nn.Module):
    def __init__(self, temperature=1.0, hard=False):
        super(ConcreteDistribution, self).__init__()
        self.temperature = temperature
        self.hard = hard

    def forward(self, logits):
        noise = torch.rand_like(logits).log().neg().log().neg()
        y = logits + noise
        y = F.softmax(y / self.temperature, dim=-1)

        # Hard version: Take the one-hot vector with max probability
        if self.hard:
            k = torch.argmax(y, dim=-1)
            y_hard = torch.zeros_like(y).scatter_(-1, k.unsqueeze(-1), 1.0)
            y = (y_hard - y).detach() + y  # Straight-through estimator
        return y

class ConcreteAutoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim, shared_dim=128, temperature=1.0, dropout_rate=0.3):
        super(ConcreteAutoencoder, self).__init__()

        # Encoder: Mapping input to latent space
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, shared_dim),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(dropout_rate),  # Add dropout after ReLU
            nn.Linear(shared_dim, latent_dim)
        )

        # Decoder: Mapping latent space back to original input space
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, shared_dim),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(dropout_rate),  # Add dropout after ReLU
            nn.Linear(shared_dim, input_dim)
        )
        
        # Concrete distribution parameters
        self.temperature = temperature
        self.concrete = ConcreteDistribution(temperature=self.temperature, hard=False)

    def forward(self, fused_rep):
        """
        fused_rep: The representation coming from the attention fusion of multiple modalities
        """
        # Encoder produces logits
        encoded = self.encoder(fused_rep)
        
        # Apply concrete distribution to get discrete latent representation
        concrete_rep = self.concrete(encoded)
        
        # Decoder reconstructs input from the latent representation
        reconstructed = self.decoder(concrete_rep)
        
        return reconstructed

