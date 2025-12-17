import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.dataset import OptionDataset
from src.models import SpectralDeepONet, get_standard_grid

def viz_surface_comparison():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load Model
    model = SpectralDeepONet(input_channels=6, latent_dim=64).to(device)
    model_path = '../../training/models/deeponet.pth'
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
    except:
        print(f"No weights at {model_path}")
        return

    model.eval()

    # Load Real Data (Validation 2022)
    dataset_path = '../../data/processed_dataset.parquet'
    dataset = OptionDataset(dataset_path, mode='val')
    loader = DataLoader(dataset, batch_size=1, shuffle=True)

    base_grid = get_standard_grid(device)

    # Find a good sample (non-empty)
    found = False
    with torch.no_grad():
        for x, y_price, y_iv, domain in loader:
            x = x.to(device)
            # Check for valid IVs
            if y_iv.max() > 0.1 and y_iv.max() < 1.0:
                # Predict
                grid = base_grid.expand(1, -1, -1)
                _, sigma_pred, _, _ = model(x, grid, alpha=0.0)

                iv_pred = sigma_pred.squeeze().cpu().numpy()
                iv_true = y_iv.squeeze().cpu().numpy()

                found = True
                break

    if not found:
        print("No suitable sample found")
        return

    iv_pred_mat = iv_pred.reshape(3, 7)
    iv_true_mat = iv_true.reshape(3, 7)

    # Calculate Error
    err = np.abs(iv_pred_mat - iv_true_mat)

    os.makedirs('../../analysis/plots', exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    sns.heatmap(iv_true_mat, ax=axes[0], cmap='viridis', annot=True, fmt='.2f')
    axes[0].set_title("True Volatility Surface (2022)")
    axes[0].set_xlabel("Moneyness Index")
    axes[0].set_ylabel("Maturity Index")

    sns.heatmap(iv_pred_mat, ax=axes[1], cmap='viridis', annot=True, fmt='.2f')
    axes[1].set_title("Predicted Surface (Neural Operator)")
    axes[1].set_xlabel("Moneyness Index")
    axes[1].set_yticks([])

    sns.heatmap(err, ax=axes[2], cmap='rocket', annot=True, fmt='.3f')
    axes[2].set_title("Absolute Error")
    axes[2].set_xlabel("Moneyness Index")
    axes[2].set_yticks([])

    plt.tight_layout()
    plt.savefig('../../analysis/plots/surface_comparison.png')
    print("Saved surface plot to analysis/plots/surface_comparison.png")

if __name__ == "__main__":
    viz_surface_comparison()
