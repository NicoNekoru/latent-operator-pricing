import torch
import torch.nn as nn
import torch.nn.functional as F

class NeuralOperator(nn.Module):
    def __init__(self, input_dim=6, latent_dim=3, output_dim=21):
        super(NeuralOperator, self).__init__()

        # Encoder: Maps history (30 days * 6 features) -> Latent Z
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(30 * input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, latent_dim)
        )

        # Decoder: Maps Latent Z -> Option Price Surface (21 values)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
            nn.Softplus() # Ensure positive prices
        )

    def forward(self, x):
        z = self.encoder(x)
        prices = self.decoder(z)
        return prices, z
