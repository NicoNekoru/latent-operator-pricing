import torch
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.models import SpectralDeepONet, get_standard_grid
from src.dataset import OptionDataset
from torch.utils.data import DataLoader

def inspect_model():
    device = torch.device("cpu")
    model = SpectralDeepONet(input_channels=6).to(device)

    # Load Weights if available
    try:
        model.load_state_dict(torch.load('../../training/models/deeponet.pth', map_location=device))
        print("Loaded model weights.")
    except:
        print("No weights found at ../../training/models/deeponet.pth, using random init (Expect garbage).")

    model.eval()

    dataset = OptionDataset('../../data/processed_dataset.parquet', mode='val')
    loader = DataLoader(dataset, batch_size=1, shuffle=True)

    base_grid = get_standard_grid(device)

    with torch.no_grad():
        for x, y_price, y_iv, domain in loader:
            x, y_iv = x.to(device), y_iv.to(device)

            # Predict
            price_pred, sigma_pred, _, _ = model(x, base_grid.expand(1, -1, -1))

            # Convert to numpy
            iv_pred = sigma_pred.squeeze().numpy()
            iv_true = y_iv.squeeze().numpy()

            # Print Stats
            print(f"\nSample Date: Domain {domain.item()}")
            print(f"Pred IV: Min {iv_pred.min():.4f}, Max {iv_pred.max():.4f}, Mean {iv_pred.mean():.4f}")
            print(f"True IV: Min {iv_true.min():.4f}, Max {iv_true.max():.4f}, Mean {iv_true.mean():.4f}")

            err = np.abs(iv_pred - iv_true)
            print(f"Mean Abs Error (MAE): {err.mean():.4f}")

            # Price Check
            p_pred = price_pred.squeeze().numpy()
            p_true = y_price.squeeze().numpy()
            mape = np.mean(np.abs((p_pred - p_true) / (p_true + 1e-6))) * 100
            print(f"Price MAPE: {mape:.2f}%")

            if mape > 20.0:
                print(">>> HIGH ERROR DETECTED")
                break

if __name__ == "__main__":
    inspect_model()
