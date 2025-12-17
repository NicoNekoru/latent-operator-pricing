import torch
import torch.nn as nn
import torch.nn.functional as F
from neuralop.layers.spectral_convolution import SpectralConv

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

class FourierFeatureMapping(nn.Module):
    def __init__(self, input_dim, mapping_size=64, scale=10.0):
        super().__init__()
        self.net = nn.Linear(input_dim, mapping_size)
        # Initialize frequencies from Gaussian
        nn.init.normal_(self.net.weight, mean=0.0, std=scale)
        nn.init.constant_(self.net.bias, 0.0)
        # Make untrainable (random features)
        for param in self.net.parameters():
            param.requires_grad = False

    def forward(self, x):
        x_proj = self.net(x)
        return torch.cat([torch.cos(x_proj), torch.sin(x_proj)], dim=-1)

class DifferentiableBSM(nn.Module):
    """
    Differentiable Black-Scholes-Merton Layer.
    Computes Call Price given Spot, Strike, Time, Rate, and Volatility.
    """
    def __init__(self):
        super().__init__()

    def forward(self, k, t, sigma, r=0.0):
        """
        Inputs:
            k: Moneyness (K/S) (Batch, N)
            t: Time to Maturity (Batch, N)
            sigma: Implied Volatility (Batch, N)
            r: Risk-free rate (scalar or Batch, N)
        """
        # Standardize inputs
        # S = 1.0 (Since k is Moneyness K/S)
        # K = k * S = k

        # Ensure numerical stability
        t = torch.clamp(t, min=1e-5)
        sigma = torch.clamp(sigma, min=1e-5)

        d1 = (torch.log(1.0 / k) + (r + 0.5 * sigma**2) * t) / (sigma * torch.sqrt(t))
        d2 = d1 - sigma * torch.sqrt(t)

        # CDF of Normal Distribution
        dist = torch.distributions.Normal(0, 1)

        # Call Price / S
        # C/S = N(d1) - K/S * e^(-rt) * N(d2)
        price = dist.cdf(d1) - k * torch.exp(-r * t) * dist.cdf(d2)
        return price

class DeepONet(nn.Module):
    """
    Deep Operator Network (DeepONet) for Option Pricing.

    Learns the operator G: u -> s, where:
    - u is the input function (Market History)
    - s is the solution function (Option Price Surface)

    Architecture (High-Capacity w/ Fourier Features):
    - Branch Net: 1D CNN + ResNet MLP Head.
    - Trunk Net: Fourier Features + ResNet MLP.
    - Output: IMPLIED VOLATILITY Surface.
    - Decoder: Differentiable BSM Layer (IV -> Price).
    """
    def __init__(self, input_channels=6, seq_len=30, latent_dim=64, hidden_dim=256):
        super(DeepONet, self).__init__()

        # --- Branch Net ---
        # 1D CNN for feature extraction from time series
        # Increased capacity (64 -> 128)
        self.branch_cnn = nn.Sequential(
            nn.BatchNorm1d(input_channels), # Normalize inputs (e.g. LogReturn)
            nn.Conv1d(in_channels=input_channels, out_channels=64, kernel_size=3, padding=1),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten()
        )

        # ResNet MLP Head for Branch
        self.branch_mlp = nn.Sequential(
            nn.Linear(128, hidden_dim),
            nn.GELU(),
            ResidualBlock(hidden_dim),
            ResidualBlock(hidden_dim),
            ResidualBlock(hidden_dim), # Added depth
            nn.Linear(hidden_dim, latent_dim)
        )

        # --- Trunk Net ---
        # Coordinate Processing with Fourier Features
        # Input grid is (2,) -> Mapped to (128,)
        self.fourier_map = FourierFeatureMapping(input_dim=2, mapping_size=hidden_dim // 2, scale=10.0)

        # ResNet MLP for Coordinates
        self.trunk_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), # Input matches Fourier output size
            nn.GELU(),
            ResidualBlock(hidden_dim),
            ResidualBlock(hidden_dim),
            ResidualBlock(hidden_dim),
            nn.Linear(hidden_dim, latent_dim)
        )


        self.bias = nn.Parameter(torch.tensor(-2.5)) # Initialize to ~17% Vol (Average of dataset)

        # Static Smile Network (Learns average Vol Surface)
        # Input: (T, K) -> Output: Log-Vol correction
        self.smile_net = nn.Sequential(
            nn.Linear(2, 32),
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )

        # Zero-initialize the last layer of Branch MLP to ensure it starts as "Zero Mean" contribution
        nn.init.constant_(self.branch_mlp[-1].weight, 0.0)
        nn.init.constant_(self.branch_mlp[-1].bias, 0.0)

        # Zero-initialize the last layer of Smile Net
        nn.init.constant_(self.smile_net[-1].weight, 0.0)
        nn.init.constant_(self.smile_net[-1].bias, 0.0)

        self.bsm = DifferentiableBSM()

    def forward(self, x, grid):
        """
        x: Market History (Batch, Seq_Len, Channels)
        grid: Query Coordinates (Batch, N_points, 2) [Moneyness, Maturity]
        """
        # Process Branch (Input Function)
        x = x.permute(0, 2, 1)
        features = self.branch_cnn(x)
        b = self.branch_mlp(features) # (Batch, latent_dim)

        # Process Trunk (Output Coordinates)
        grid_embed = self.fourier_map(grid)
        t = self.trunk_net(grid_embed) # (Batch, N, latent_dim)

        # Dynamic Component (Batch, N)
        b = b.unsqueeze(1)
        iv_dynamic = torch.sum(b * t, dim=-1)

        # Static Component (Batch, N, 1) -> (Batch, N)
        # Grid input for smile net: Normalize or raw?
        # Grid is ~1.0 for K and ~0.5 for T. Safe for Tanh.
        iv_static = self.smile_net(grid).squeeze(-1)

        # Total IV Pre-activation
        iv_raw = iv_dynamic + iv_static + self.bias

        # Force IV > 0. Using Sigmoid to bound Vol and keep gradients alive
        # Range: [0.01, 2.0] (1% to 200% Vol) - Wider range for crisis handling
        sigma = 0.01 + 2.0 * torch.sigmoid(iv_raw)

        # Decode to Price using BSM
        # grid[:, :, 0] = Moneyness (K/S)
        # grid[:, :, 1] = Maturity (T)
        k = grid[:, :, 0]
        t = grid[:, :, 1]

        # Extract Risk-Free Rate from Input Features
        # x is (Batch, Channels, Seq_Len) after permute in forward() line 140
        # Wait, line 140: x = x.permute(0, 2, 1) -> (Batch, Channels, Seq_Len)
        # Channel 4 is TNX. We want the rate at the current time (last step, -1).
        # r shape: (Batch, 1) -> Broadcast to (Batch, N)
        r = x[:, 4, -1].unsqueeze(1)

        price = self.bsm(k, t, sigma, r=r)

        return price, sigma, b.squeeze(1) # Return prices, sigma, and latent coefficients

class GradientReversalLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

def grad_reverse(x, alpha=1.0):
    return GradientReversalLayer.apply(x, alpha)

class SpectralDeepONet(nn.Module):
    def __init__(self, input_channels=6, latent_dim=64, hidden_dim=64, n_modes=12):
        super().__init__()

        # Branch Net: Spectral Convolution (Global processing)
        # Input: (Batch, Channels, Seq_Len)
        self.spec_conv1 = SpectralConv(
            in_channels=input_channels,
            out_channels=hidden_dim,
            n_modes=(n_modes,)
        )

        self.spec_conv2 = SpectralConv(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            n_modes=(n_modes,)
        )

        self.branch_pool = nn.AdaptiveAvgPool1d(1) # Latent Vector
        self.branch_projection = nn.Linear(hidden_dim, latent_dim)

        # Trunk Net: Fourier Features
        self.fourier_map = FourierFeatureMapping(input_dim=2, mapping_size=hidden_dim)
        self.trunk_net = nn.Sequential(
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, latent_dim)
        )

        # Bias & Smile Net
        self.bias = nn.Parameter(torch.tensor(-2.5))
        self.smile_net = nn.Sequential(
            nn.Linear(2, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )

        # --- Domain Adversarial Classifier ---
        # Predicts domain (0=Source, 1=Target) from latent features
        self.domain_classifier = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1) # Logits
        )

        # Initialize Output Heads
        nn.init.constant_(self.branch_projection.weight, 0.0)
        nn.init.constant_(self.branch_projection.bias, 0.0)
        nn.init.constant_(self.smile_net[-1].weight, 0.0)
        nn.init.constant_(self.smile_net[-1].bias, 0.0)

        self.bsm = DifferentiableBSM()

    def forward(self, x, grid, alpha=1.0):
        """
        x: (Batch, Seq_Len, Channels)
        grid: (Batch, N, 2)
        alpha: GRL scaling factor (0 -> 1 during training)
        """
        # Save Risk-Free Rate (Channel 4, Last Step)
        # x is (B, L, C). TNX is index 4.
        r = x[:, -1, 4].unsqueeze(1) # (B, 1)

        # --- Process Branch (Feature Extractor) ---
        x_in = x.permute(0, 2, 1) # (B, C, L)
        x_out = self.spec_conv1(x_in)
        x_out = F.gelu(x_out)
        x_out = self.spec_conv2(x_out)
        x_out = F.gelu(x_out)

        x_out = self.branch_pool(x_out).squeeze(-1) # (B, H)
        b = self.branch_projection(x_out) # (B, latent_dim) Features

        # --- Domain Classification (Adversarial Branch) ---
        reverse_b = grad_reverse(b, alpha)
        domain_pred = self.domain_classifier(reverse_b)

        # --- Process Trunk (Grid) ---
        grid_embed = self.fourier_map(grid)
        t = self.trunk_net(grid_embed)

        # --- Combine & Price ---
        b_expanded = b.unsqueeze(1)
        iv_dynamic = torch.sum(b_expanded * t, dim=-1)
        iv_static = self.smile_net(grid).squeeze(-1)

        iv_raw = iv_dynamic + iv_static + self.bias
        sigma = 0.01 + 2.0 * torch.sigmoid(iv_raw)

        # --- BSM Decode ---
        k = grid[:, :, 0]
        t_mat = grid[:, :, 1]

        price = self.bsm(k, t_mat, sigma, r=r)

        return price, sigma, b, domain_pred

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
