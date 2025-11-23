import torch
import torch.nn as nn
import torch.nn.functional as F

class MarketEncoder(nn.Module):
    """
    Encodes 30-day market history (Returns, Vol) into a latent vector Z using an MLP.
    Input Shape: (Batch, 30, 2) -> Flattened to (Batch, 60)
    Output Shape: (Batch, latent_dim)
    """
    def __init__(self, input_channels=2, seq_length=30, latent_dim=3):
        super(MarketEncoder, self).__init__()

        input_dim = input_channels * seq_length # 2 * 30 = 60

        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, latent_dim)
        )

    def forward(self, x):
        # x: (Batch, 30, 2)
        x = x.reshape(x.size(0), -1) # Flatten -> (Batch, 60)
        z = self.net(x)
        return z

class SurfaceDecoder(nn.Module):
    """
    Decodes latent vector Z into Option Price Surface.
    Input Shape: (Batch, latent_dim)
    Output Shape: (Batch, 21) -> (3 Maturities * 7 Strikes)
    """
    def __init__(self, latent_dim=3, output_dim=21):
        super(SurfaceDecoder, self).__init__()

        self.fc1 = nn.Linear(latent_dim, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, output_dim)

    def forward(self, z):
        x = F.relu(self.fc1(z))
        x = F.relu(self.fc2(x))
        # Use Softplus to ensure positive prices
        prices = F.softplus(self.fc3(x))
        return prices

class NeuralOperator(nn.Module):
    def __init__(self, latent_dim=3):
        super(NeuralOperator, self).__init__()
        self.encoder = MarketEncoder(latent_dim=latent_dim)
        self.decoder = SurfaceDecoder(latent_dim=latent_dim)

    def forward(self, x):
        z = self.encoder(x)
        prices = self.decoder(z)
        return prices, z
