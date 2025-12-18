"""
Debug script to understand strategy signal behavior.
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.append(PROJECT_ROOT)

from src.models import SpectralDeepONet, get_standard_grid
from src.dataset import OptionDataset
from torch.utils.data import DataLoader
import pandas as pd


def extract_model_outputs(model, dataset, device):
    loader = DataLoader(dataset, batch_size=64, shuffle=False)
    base_grid = get_standard_grid(device)

    all_latents = []
    all_surfaces = []
    all_returns = []

    model.eval()
    with torch.no_grad():
        for x, y_price, y_iv, domain in loader:
            x = x.to(device)
            bs = x.size(0)
            grid = base_grid.expand(bs, -1, -1)

            price_pred, sigma_pred, latent, _ = model(x, grid, alpha=0.0)

            all_latents.append(latent.cpu().numpy())
            all_surfaces.append(sigma_pred.cpu().numpy())
            returns = x[:, -1, 0].cpu().numpy()
            all_returns.append(returns)

    return (np.concatenate(all_latents),
            np.concatenate(all_surfaces),
            np.concatenate(all_returns))


def main():
    device = torch.device('cpu')

    # Load model
    model = SpectralDeepONet(input_channels=6, latent_dim=64).to(device)
    model_path = os.path.join(PROJECT_ROOT, 'training/models/deeponet.pth')
    model.load_state_dict(torch.load(model_path, map_location=device))

    # Load data
    dataset_path = os.path.join(PROJECT_ROOT, 'data/processed_dataset.parquet')
    train_dataset = OptionDataset(dataset_path, mode='train')
    val_dataset = OptionDataset(dataset_path, mode='val')

    train_latents, train_surfaces, train_returns = extract_model_outputs(model, train_dataset, device)
    val_latents, val_surfaces, val_returns = extract_model_outputs(model, val_dataset, device)

    # Analyze latent space
    print("=== Latent Space Analysis ===")
    train_norms = np.linalg.norm(train_latents, axis=1)
    val_norms = np.linalg.norm(val_latents, axis=1)

    print(f"Train latent norms: mean={train_norms.mean():.4f}, std={train_norms.std():.4f}, "
          f"min={train_norms.min():.4f}, max={train_norms.max():.4f}")
    print(f"Val latent norms:   mean={val_norms.mean():.4f}, std={val_norms.std():.4f}, "
          f"min={val_norms.min():.4f}, max={val_norms.max():.4f}")

    print(f"\nTrain norm percentiles: 25th={np.percentile(train_norms, 25):.4f}, "
          f"50th={np.percentile(train_norms, 50):.4f}, 75th={np.percentile(train_norms, 75):.4f}")
    print(f"Val norm percentiles:   25th={np.percentile(val_norms, 25):.4f}, "
          f"50th={np.percentile(val_norms, 50):.4f}, 75th={np.percentile(val_norms, 75):.4f}")

    # Test the Regime Distance logic
    print("\n=== Regime Distance Debug ===")

    # Compute calm centroid from train data
    median_norm = np.median(train_norms)
    calm_mask = train_norms < median_norm
    calm_centroid = np.mean(train_latents[calm_mask], axis=0)

    print(f"Calm centroid computed from {calm_mask.sum()} samples (norm < {median_norm:.4f})")
    print(f"Calm centroid norm: {np.linalg.norm(calm_centroid):.4f}")

    # Distance from calm for all samples
    train_distances = np.linalg.norm(train_latents - calm_centroid, axis=1)
    val_distances = np.linalg.norm(val_latents - calm_centroid, axis=1)

    print(f"\nTrain distances from calm: mean={train_distances.mean():.4f}, std={train_distances.std():.4f}")
    print(f"Val distances from calm:   mean={val_distances.mean():.4f}, std={val_distances.std():.4f}")

    # What threshold would separate train calm from val crisis?
    train_95th = np.percentile(train_distances, 95)
    print(f"\nTrain 95th percentile distance: {train_95th:.4f}")
    print(f"% of val samples above this threshold: {(val_distances > train_95th).mean()*100:.1f}%")

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Latent norms over time
    ax1 = axes[0, 0]
    ax1.plot(train_norms, label='Train', alpha=0.7)
    ax1.axhline(train_norms.mean(), color='blue', linestyle='--', label='Train mean')
    ax1.set_title('Train: Latent Norms Over Time')
    ax1.set_ylabel('||z||')
    ax1.legend()

    ax2 = axes[0, 1]
    ax2.plot(val_norms, label='Val (2022)', color='red', alpha=0.7)
    ax2.axhline(val_norms.mean(), color='red', linestyle='--', label='Val mean')
    ax2.axhline(train_norms.mean(), color='blue', linestyle='--', alpha=0.5, label='Train mean')
    ax2.set_title('Validation: Latent Norms Over Time')
    ax2.set_ylabel('||z||')
    ax2.legend()

    # Distance from calm
    ax3 = axes[1, 0]
    ax3.hist(train_distances, bins=50, alpha=0.7, label='Train', density=True)
    ax3.hist(val_distances, bins=50, alpha=0.7, label='Val', density=True)
    ax3.axvline(train_95th, color='red', linestyle='--', label='Train 95th %ile')
    ax3.set_title('Distance from Calm Centroid')
    ax3.set_xlabel('Distance')
    ax3.legend()

    # What signal would be generated?
    # Using the current Regime Distance logic
    ax4 = axes[1, 1]

    # Simulate signals
    signals = []
    for i in range(len(val_latents)):
        if i < 60:
            signals.append(1.0)
            continue

        # Use val history only
        history = val_latents[max(0, i-60):i]
        norms = np.linalg.norm(history, axis=1)
        median_n = np.median(norms)
        calm_mask = norms < median_n
        if calm_mask.sum() < 5:
            local_centroid = np.mean(history, axis=0)
        else:
            local_centroid = np.mean(history[calm_mask], axis=0)

        current_dist = np.linalg.norm(val_latents[i] - local_centroid)
        hist_dists = np.linalg.norm(history - local_centroid, axis=1)
        dist_std = np.std(hist_dists) + 1e-6
        normalized_dist = current_dist / dist_std

        # Signal with current params
        signal = 1.0 - np.tanh(5.0 * (normalized_dist - 0.3))
        signal = np.clip(signal, 0.0, 1.0)
        signals.append(signal)

    signals = np.array(signals)
    ax4.plot(signals, label='Signal (0=Cash, 1=Long)')
    ax4.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
    ax4.set_title(f'Regime Distance Signals on Val\n(Mean signal: {signals.mean():.3f})')
    ax4.set_ylabel('Position')
    ax4.set_ylim(-0.1, 1.1)
    ax4.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(PROJECT_ROOT, 'analysis/plots/strategy_debug.png'), dpi=150)
    print(f"\nSaved debug plot to analysis/plots/strategy_debug.png")


if __name__ == "__main__":
    main()
