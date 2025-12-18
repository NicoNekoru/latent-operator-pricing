"""
Manifold Walk Visualization

Demonstrates that the learned latent space captures continuous geometry
by interpolating between two market regimes (Calm vs Crisis) and decoding
the intermediate latent vectors into volatility surfaces.

Output: analysis/plots/manifold_walk.png
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.append(PROJECT_ROOT)

from src.models import SpectralDeepONet, get_standard_grid
from src.dataset import OptionDataset
from torch.utils.data import DataLoader


def find_regime_samples(dataset, model, device, n_samples=500):
    """
    Find samples representing Calm and Crisis regimes based on
    the magnitude of the latent vector (proxy for volatility state).
    """
    loader = DataLoader(dataset, batch_size=64, shuffle=False)
    base_grid = get_standard_grid(device)

    all_latents = []
    all_inputs = []

    model.eval()
    with torch.no_grad():
        count = 0
        for x, _, _, _ in loader:
            x = x.to(device)
            bs = x.size(0)
            grid = base_grid.expand(bs, -1, -1)

            _, _, b, _ = model(x, grid, alpha=0.0)

            all_latents.append(b.cpu())
            all_inputs.append(x.cpu())

            count += bs
            if count >= n_samples:
                break

    latents = torch.cat(all_latents, dim=0)[:n_samples]
    inputs = torch.cat(all_inputs, dim=0)[:n_samples]

    # Use L2 norm of latent as "volatility magnitude"
    norms = torch.norm(latents, dim=1)

    # Find calmest (lowest norm) and most volatile (highest norm)
    calm_idx = torch.argmin(norms).item()
    crisis_idx = torch.argmax(norms).item()

    return {
        'calm': {'input': inputs[calm_idx], 'latent': latents[calm_idx]},
        'crisis': {'input': inputs[crisis_idx], 'latent': latents[crisis_idx]}
    }


def interpolate_latent(z_calm, z_crisis, n_steps=7):
    """
    Linear interpolation between two latent vectors.
    """
    alphas = np.linspace(0, 1, n_steps)
    z_interp = []
    for alpha in alphas:
        z = (1 - alpha) * z_calm + alpha * z_crisis
        z_interp.append(z)
    return z_interp, alphas


def decode_latent_to_surface(model, z, grid, device):
    """
    Decode a latent vector to a volatility surface.

    Note: This requires manually invoking the trunk net and BSM layer
    since we're bypassing the branch net.
    """
    model.eval()
    with torch.no_grad():
        # Trunk net processing
        grid_embed = model.fourier_map(grid)
        t = model.trunk_net(grid_embed)  # (1, N, latent_dim)

        # Combine with latent
        z_expanded = z.unsqueeze(0).unsqueeze(1).to(device)  # (1, 1, latent_dim)
        iv_dynamic = torch.sum(z_expanded * t, dim=-1)  # (1, N)

        # Static smile component
        iv_static = model.smile_net(grid).squeeze(-1)  # (1, N)

        # Total IV
        iv_raw = iv_dynamic + iv_static + model.bias
        sigma = 0.01 + 2.0 * torch.sigmoid(iv_raw)

        return sigma.squeeze(0).cpu().numpy()


def plot_manifold_walk(surfaces, alphas, grid):
    """
    Create a visualization of the manifold walk.
    Uses a 2-row layout for better ergonomics.
    """
    n_steps = len(surfaces)

    # Grid coordinates
    moneyness = grid[0, :, 0].numpy()
    maturity = grid[0, :, 1].numpy()

    # Reshape to surface (3 maturities x 7 strikes)
    n_mat, n_strike = 3, 7

    # Create 2x4 grid figure (7 plots + 1 for colorbar/legend)
    n_cols = 4
    n_rows = 2
    fig = plt.figure(figsize=(14, 8))

    # Moneyness and Maturity grids
    K = np.array([0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2])
    T = np.array([1/12, 3/12, 6/12])
    KK, TT = np.meshgrid(K, T)

    vmin = min(s.min() for s in surfaces)
    vmax = max(s.max() for s in surfaces)

    # Plot positions: top row [0,1,2,3], bottom row [4,5,6,colorbar]
    for i, (surf, alpha) in enumerate(zip(surfaces, alphas)):
        row = i // n_cols
        col = i % n_cols

        ax = fig.add_subplot(n_rows, n_cols, i + 1, projection='3d')

        # Reshape surface
        IV = surf.reshape(n_mat, n_strike)

        # Plot surface
        surf_plot = ax.plot_surface(KK, TT, IV, cmap='viridis', alpha=0.85,
                                     vmin=vmin, vmax=vmax)

        ax.set_xlabel('K/S', fontsize=9, labelpad=2)
        ax.set_ylabel('T', fontsize=9, labelpad=2)
        ax.set_zlabel(r'$\sigma$', fontsize=9, labelpad=2)

        # Label with regime interpretation
        if i == 0:
            label = fr'$\alpha={alpha:.1f}$ (Calm)'
        elif i == n_steps - 1:
            label = fr'$\alpha={alpha:.1f}$ (Crisis)'
        else:
            label = fr'$\alpha={alpha:.2f}$'
        ax.set_title(label, fontsize=11, fontweight='bold' if i in [0, n_steps-1] else 'normal')

        ax.set_zlim(vmin * 0.9, vmax * 1.1)
        ax.view_init(elev=25, azim=45)
        ax.tick_params(labelsize=7)

    # Add colorbar in the last subplot position
    cbar_ax = fig.add_subplot(n_rows, n_cols, 8)
    cbar_ax.axis('off')

    # Create a scalar mappable for colorbar
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=cbar_ax, shrink=0.8, aspect=15, pad=0.1)
    cbar.set_label(r'Implied Volatility ($\sigma$)', fontsize=10)

    # Add annotation in colorbar subplot
    cbar_ax.text(0.5, 0.3, r'Interpolation:' + '\n' + r'$z(\alpha) = (1-\alpha)z_{calm} + \alpha z_{crisis}$',
                 ha='center', va='center', fontsize=10,
                 transform=cbar_ax.transAxes,
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    fig.suptitle('Manifold Walk: Latent Space Interpolation from Calm to Crisis',
                 fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    return fig


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load model
    model = SpectralDeepONet(input_channels=6, latent_dim=64).to(device)
    model_path = os.path.join(PROJECT_ROOT, 'training/models/deeponet.pth')

    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("Model loaded successfully.")
    except FileNotFoundError:
        print(f"Model not found at {model_path}. Please train first.")
        return

    # Load dataset
    dataset_path = os.path.join(PROJECT_ROOT, 'data/processed_dataset.parquet')
    dataset = OptionDataset(dataset_path, mode='train')

    # Find regime samples
    print("Finding Calm and Crisis samples...")
    samples = find_regime_samples(dataset, model, device)

    z_calm = samples['calm']['latent']
    z_crisis = samples['crisis']['latent']

    print(f"Calm latent norm: {torch.norm(z_calm):.4f}")
    print(f"Crisis latent norm: {torch.norm(z_crisis):.4f}")

    # Interpolate
    print("Interpolating latent vectors...")
    z_interp, alphas = interpolate_latent(z_calm, z_crisis, n_steps=7)

    # Decode each to surface
    base_grid = get_standard_grid(device)

    print("Decoding surfaces...")
    surfaces = []
    for z in z_interp:
        surf = decode_latent_to_surface(model, z, base_grid, device)
        surfaces.append(surf)

    # Plot
    print("Creating visualization...")
    fig = plot_manifold_walk(surfaces, alphas, base_grid.cpu())

    # Save
    plot_dir = os.path.join(PROJECT_ROOT, 'analysis/plots')
    os.makedirs(plot_dir, exist_ok=True)
    save_path = os.path.join(plot_dir, 'manifold_walk.png')
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved to {save_path}")

    plt.close()


if __name__ == "__main__":
    main()
