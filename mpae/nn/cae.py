import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


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

class ConcreteAutoEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dim=640, temperature=1.0, dropout_rate=0.1):
        super(ConcreteAutoEncoder, self).__init__()
        
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

        # Concrete distribution parameters
        self.temperature = temperature
        self.concrete = ConcreteDistribution(temperature=self.temperature, hard=False)
        
        # Apply weight initialization
        self.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            init.kaiming_normal_(m.weight, a=0.01, mode='fan_in', nonlinearity='leaky_relu')
            if m.bias is not None:
                init.zeros_(m.bias)

    def update_temp(self, new_temp):
        self.concrete.temperature = new_temp
                
    def encode(self, fused_rep):
        """
        Encodes the input into the latent space.
        """
        encoded = self.encoder(fused_rep)
        return encoded

    def forward(self, fused_rep):
        """
        fused_rep: The representation coming from the attention fusion of multiple modalities
        """
        # Encoder produces logits
        encoded = self.encode(fused_rep)
        
        # Apply concrete distribution to get discrete latent representation
        concrete_rep = self.concrete(encoded)
        
        # Decoder reconstructs input from the latent representation
        reconstructed = self.decoder(concrete_rep)
        
        return reconstructed

class AttentiveConcreteAutoEncoder(ConcreteAutoEncoder):
    def __init__(self, input_dim, latent_dim, hidden_dim=640, temperature=1, dropout_rate=0.1):
        super().__init__(input_dim=input_dim, latent_dim=latent_dim, hidden_dim=hidden_dim, temperature=temperature, dropout_rate=dropout_rate)
        self.attention = nn.MultiheadAttention(embed_dim=input_dim, num_heads=4)

    def encode(self, fused_rep):
        attentive_rep, _ = self.attention(fused_rep, fused_rep, fused_rep)
        return super().encode(attentive_rep)


class CementAutoEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dim=640, dropout_rate=0.1):
        super(CementAutoEncoder, self).__init__()
        
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
        encoded = self.encoder(fused_rep)
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

class AttentiveCementAutoEncoder(CementAutoEncoder):
    def __init__(self, input_dim, latent_dim, hidden_dim=640, dropout_rate=0.1):
        super().__init__(input_dim, latent_dim, hidden_dim, dropout_rate)
        self.attention = nn.MultiheadAttention(embed_dim=input_dim, num_heads=4)

    def encode(self, fused_rep):
        attentive_rep, _ = self.attention(fused_rep, fused_rep, fused_rep)
        return super().encode(attentive_rep)