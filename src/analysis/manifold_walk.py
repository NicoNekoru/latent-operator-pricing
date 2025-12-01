import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.models import ManifoldAutoencoder

def plot_surface(prices, title, filename):
    """
    Plot a single option surface (21 contracts).
    Assuming 3 maturities x 7 strikes.
    """
    # Reshape to (3, 7)
    # We don't have the exact grid (Maturities, Strikes) here, so we'll just plot indices.
    prices = prices.reshape(3, 7)

    X, Y = np.meshgrid(np.arange(7), np.arange(3))

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, prices, cmap='viridis')
    ax.set_xlabel('Strike Index')
    ax.set_ylabel('Maturity Index')
    ax.set_zlabel('Price')
    ax.set_title(title)
    plt.savefig(filename)
    plt.close()

def main():
    print("Generating Manifold Walk Visualization...")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ManifoldAutoencoder(input_dim=6, latent_dim=3).to(device)

    # Load weights
    model_path = 'models/neural_operator.pth'
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        print("Model not found. Using random weights.")

    model.eval()

    # Define Start (Calm) and End (Crisis) points
    # Based on previous observation of latent stats
    mean = np.array([153.6, -537.0, 222.0])
    std = np.array([9.0, 23.7, 9.4])

    # Let's assume "Calm" is near the mean and "Crisis" is far away (e.g. +3 sigma)
    z_calm = torch.tensor(mean).float().to(device).unsqueeze(0)
    z_crisis = torch.tensor(mean + 3 * std).float().to(device).unsqueeze(0)

    # Interpolate
    steps = 5
    alphas = np.linspace(0, 1, steps)

    os.makedirs('plots/manifold_walk', exist_ok=True)

    fig = plt.figure(figsize=(15, 5))

    for i, alpha in enumerate(alphas):
        z_interp = (1 - alpha) * z_calm + alpha * z_crisis

        with torch.no_grad():
            prices = model.decoder(z_interp).cpu().numpy().flatten()

        # Plot
        # We'll create a subplot for each step
        ax = fig.add_subplot(1, steps, i+1, projection='3d')
        prices_grid = prices.reshape(3, 7)
        X, Y = np.meshgrid(np.arange(7), np.arange(3))
        ax.plot_surface(X, Y, prices_grid, cmap='plasma')
        ax.set_title(f"$\\alpha={alpha:.2f}$")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])

    plt.suptitle("Manifold Walk: Interpolation from Calm to Crisis")
    plt.tight_layout()
    plt.savefig('plots/manifold_walk.png')
    print("Saved plot to plots/manifold_walk.png")

if __name__ == "__main__":
    main()
