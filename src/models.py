import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    """
    Simple Residual Block: x + Dropout(Linear(GELU(Linear(x))))
    """
    def __init__(self, features, dropout=0.1):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(features, features),
            nn.LayerNorm(features),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(features, features),
            nn.LayerNorm(features)
        )

    def forward(self, x):
        return x + self.block(x)

class DeepONet(nn.Module):
    """
    Deep Operator Network (DeepONet) for Option Pricing.

    Learns the operator G: u -> s, where:
    - u is the input function (Market History)
    - s is the solution function (Option Price Surface)

    Architecture (Optimized):
    - Branch Net: 1D CNN + ResNet MLP Head.
    - Trunk Net: ResNet MLP.
    - Activation: GELU.
    """
    def __init__(self, input_channels=6, seq_len=30, latent_dim=16, hidden_dim=128):
        super(DeepONet, self).__init__()

        # --- Branch Net ---
        # 1D CNN for feature extraction from time series
        self.branch_cnn = nn.Sequential(
            nn.Conv1d(in_channels=input_channels, out_channels=32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten()
        )

        # ResNet MLP Head for Branch
        self.branch_mlp = nn.Sequential(
            nn.Linear(64, hidden_dim),
            nn.GELU(),
            ResidualBlock(hidden_dim),
            ResidualBlock(hidden_dim),
            nn.Linear(hidden_dim, latent_dim)
        )

        # --- Trunk Net ---
        # ResNet MLP for Coordinate Processing
        self.trunk_net = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.GELU(),
            ResidualBlock(hidden_dim),
            ResidualBlock(hidden_dim),
            ResidualBlock(hidden_dim),
            nn.Linear(hidden_dim, latent_dim)
        )

        self.bias = nn.Parameter(torch.tensor(0.0))

    def forward(self, x, grid):
        """
        x: Market History (Batch, Seq_Len, Channels)
        grid: Query Coordinates (Batch, N_points, 2)
        """
        # Process Branch (Input Function)
        # Permute x to (Batch, Channels, Seq_Len) for Conv1d
        x = x.permute(0, 2, 1)

        # CNN Feature Extraction
        features = self.branch_cnn(x) # (Batch, 64)

        # MLP Encoding
        b = self.branch_mlp(features) # (Batch, latent_dim)

        # Process Trunk (Output Coordinates)
        t = self.trunk_net(grid) # (Batch, N_points, latent_dim)

        # Dot Product
        # b: (Batch, 1, latent_dim)
        # t: (Batch, N_points, latent_dim)
        # out: (Batch, N_points)
        b = b.unsqueeze(1)
        out = torch.sum(b * t, dim=-1) + self.bias

        # Enforce positivity (Option prices > 0)
        out = F.softplus(out)

        return out, b.squeeze(1) # Return prices and latent coefficients

def get_standard_grid(device):
    """
    Returns the standard (Maturity, Moneyness) grid tensor.
    Shape: (1, 21, 2)
    """
    # Maturities: 1, 3, 6 months -> 1/12, 3/12, 6/12 years
    maturities = [1/12, 3/12, 6/12]
    # Moneyness: 0.8 to 1.2
    moneyness = [0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2]

    grid = []
    for m in maturities:
        for k in moneyness:
            grid.append([k, m]) # Feature order: Moneyness, Maturity

    grid_tensor = torch.tensor(grid, dtype=torch.float32).to(device)
    return grid_tensor.unsqueeze(0) # Add batch dim: (1, 21, 2)
