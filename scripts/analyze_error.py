import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.models import DeepONet, get_standard_grid
from src.dataset import OptionDataset
from torch.utils.data import DataLoader

def analyze_error():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load Model
    model = DeepONet(input_channels=6, latent_dim=16).to(device)
    try:
        model.load_state_dict(torch.load('models/deeponet.pth', map_location=device))
    except FileNotFoundError:
        print("Model not found.")
        return
    model.eval()

    # Load Validation Data
    dataset = OptionDataset('data/processed_dataset.parquet', mode='val')
    loader = DataLoader(dataset, batch_size=64, shuffle=False)

    base_grid = get_standard_grid(device)

    # Moneyness levels corresponding to the grid (flattened)
    # Grid order: 3 maturities x 7 moneyness
    # Moneyness: [0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2] repeated 3 times
    moneyness_levels = [0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2] * 3
    moneyness_levels = np.array(moneyness_levels)

    errors_by_moneyness = {k: [] for k in np.unique(moneyness_levels)}
    prices_by_moneyness = {k: [] for k in np.unique(moneyness_levels)}

    print("Analyzing error distribution...")

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            grid = base_grid.expand(x.size(0), -1, -1)

            y_pred, _ = model(x, grid)

            # y_pred and y are (Batch, 21)
            # Calculate absolute percentage error
            # Avoid division by zero
            mask = y > 1e-5

            ape = torch.abs(y_pred - y) / (y + 1e-9) * 100.0

            y_np = y.cpu().numpy()
            ape_np = ape.cpu().numpy()

            for i in range(21):
                m = moneyness_levels[i]
                # Filter out valid points
                valid_mask = mask[:, i].cpu().numpy()
                if valid_mask.sum() > 0:
                    errors_by_moneyness[m].extend(ape_np[valid_mask, i])
                    prices_by_moneyness[m].extend(y_np[valid_mask, i])

    print("\nMAPE by Moneyness:")
    print(f"{'Moneyness':<10} | {'MAPE (%)':<10} | {'Avg Price':<10} | {'Count':<10}")
    print("-" * 50)

    for m in sorted(errors_by_moneyness.keys()):
        mape = np.mean(errors_by_moneyness[m])
        avg_price = np.mean(prices_by_moneyness[m])
        count = len(errors_by_moneyness[m])
        print(f"{m:<10.2f} | {mape:<10.2f} | {avg_price:<10.5f} | {count:<10}")

if __name__ == "__main__":
    analyze_error()
