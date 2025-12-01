import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import os

from src.models import DeepONet, get_standard_grid
from src.dataset import OptionDataset

def manifold_walk():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load Data
    dataset_path = 'data/processed_dataset.parquet'
    dataset = OptionDataset(dataset_path, mode='test')

    # Load Model
    model = DeepONet(latent_dim=16).to(device)
    try:
        model.load_state_dict(torch.load('models/deeponet.pth', map_location=device))
    except FileNotFoundError:
        print("Model not found.")
        return
    model.eval()

    # Select Start and End Points
    # We want a "Calm" state (Low VIX) and a "Crisis" state (High VIX)
    # Let's search the dataset
    print("Searching for Calm and Crisis samples...")

    calm_idx = -1
    crisis_idx = -1
    min_vix = 100.0
    max_vix = 0.0

    # Scan first 1000 samples to find extremes
    for i in range(min(1000, len(dataset))):
        x, y = dataset[i]
        # x shape: (30, 6). VIX is index 2.
        # It's normalized, but let's just look at the last value
        vix = x[-1, 2].item()

        if vix < min_vix:
            min_vix = vix
            calm_idx = i
        if vix > max_vix:
            max_vix = vix
            crisis_idx = i

    print(f"Found Calm Sample (Idx {calm_idx}, VIX={min_vix:.4f})")
    print(f"Found Crisis Sample (Idx {crisis_idx}, VIX={max_vix:.4f})")

    x_calm, _ = dataset[calm_idx]
    x_crisis, _ = dataset[crisis_idx]

    x_calm = x_calm.unsqueeze(0).to(device)
    x_crisis = x_crisis.unsqueeze(0).to(device)

    # Get Latent Vectors
    # We need to manually call branch_cnn and branch_mlp
    # But wait, model.branch_net doesn't exist as a single module in my fix?
    # Let's check model definition again.
    # Ah, in models.py:
    # self.branch_cnn = ...
    # self.branch_mlp = ...
    # forward calls them.

    with torch.no_grad():
        # Encode
        # Permute for CNN
        z_calm_feat = model.branch_cnn(x_calm.permute(0, 2, 1))
        z_calm = model.branch_mlp(z_calm_feat) # (1, 16)

        z_crisis_feat = model.branch_cnn(x_crisis.permute(0, 2, 1))
        z_crisis = model.branch_mlp(z_crisis_feat) # (1, 16)

    # Interpolate
    steps = 5
    alphas = np.linspace(0, 1, steps)

    # Prepare Grid for Decoding
    grid = get_standard_grid(device) # (1, 21, 2)
    # Trunk output
    with torch.no_grad():
        t = model.trunk_net(grid) # (1, 21, 16)

    # Plotting
    fig = plt.figure(figsize=(20, 5))

    # Moneyness/Maturity for plotting
    # Grid is (Moneyness, Maturity)
    # Let's extract them
    grid_np = grid.cpu().numpy().squeeze()
    moneyness = grid_np[:, 0]
    maturity = grid_np[:, 1]

    # We want to plot surfaces. But our output is 1D (21 points).
    # We can plot them as lines or a 3D scatter/surface if we reshape.
    # But 21 points is sparse (3 mats * 7 strikes).
    # Let's plot as 3 lines (one for each maturity).

    unique_mats = np.unique(maturity)

    for i, alpha in enumerate(alphas):
        # Interpolate Latent
        z_interp = (1 - alpha) * z_calm + alpha * z_crisis

        # Decode
        # G(u)(y) = Softplus( sum(b * t) + bias )
        # b: (1, 1, 16)
        # t: (1, 21, 16)
        b = z_interp.unsqueeze(1)
        out = torch.sum(b * t, dim=-1) + model.bias
        prices = torch.nn.functional.softplus(out).cpu().detach().numpy().flatten()

        # Plot
        ax = fig.add_subplot(1, steps, i+1, projection='3d')

        # Create meshgrid for surface
        # We have irregular grid? No, it's structured.
        # 3 maturities, 7 strikes.
        X = moneyness.reshape(3, 7)
        Y = maturity.reshape(3, 7)
        Z = prices.reshape(3, 7)

        surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', alpha=0.8)

        ax.set_title(f"Interp $\\alpha={alpha:.2f}$")
        ax.set_xlabel('Moneyness')
        ax.set_ylabel('Maturity')
        ax.set_zlabel('Price')
        ax.set_zlim(0, 0.3) # Fixed scale

    plt.tight_layout()
    output_path = 'writeup/figures/manifold_walk.png'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    print(f"Saved manifold walk to {output_path}")

if __name__ == "__main__":
    manifold_walk()
