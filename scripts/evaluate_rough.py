import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import os

from src.models import DeepONet, get_standard_grid
from src.utils import calculate_metrics

def evaluate_rough():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load Data
    data_path = 'data/rough_heston_test.parquet'
    if not os.path.exists(data_path):
        print("Data not found. Run generate_rough_experiment.py first.")
        return

    df = pd.read_parquet(data_path)
    print(f"Loaded {len(df)} samples from {data_path}")

    # Prepare Tensors
    # Input: (N, 30, 6)
    inputs = np.stack(df['Input_Features'].apply(lambda x: x.reshape(30, 6)).values)
    inputs = torch.tensor(inputs, dtype=torch.float32).to(device)

    # Target: (N, 21)
    targets = np.stack(df['Target_Prices'].values)
    targets = torch.tensor(targets, dtype=torch.float32).to(device)

    # Hurst
    hurst_values = df['Hurst'].values

    # Load Model
    model = DeepONet(latent_dim=16).to(device)
    try:
        model.load_state_dict(torch.load('models/deeponet.pth', map_location=device))
    except FileNotFoundError:
        print("Model not found.")
        return
    model.eval()

    # Inference
    batch_size = 32
    n_samples = len(df)

    all_preds = []
    all_latents = []

    base_grid = get_standard_grid(device)

    with torch.no_grad():
        for i in range(0, n_samples, batch_size):
            x_batch = inputs[i:i+batch_size]

            # Get Predictions and Latents
            current_batch = x_batch.size(0)
            grid = base_grid.expand(current_batch, -1, -1)
            y_pred, latents = model(x_batch, grid)

            all_latents.append(latents.cpu().numpy())
            all_preds.append(y_pred.cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_latents = np.concatenate(all_latents, axis=0)
    all_preds_tensor = torch.tensor(all_preds, device=device)

    # Calculate Metrics
    mape, dollar = calculate_metrics(all_preds_tensor, targets)
    print(f"Rough Heston MAPE: {mape:.2f}%")
    print(f"Rough Heston Dollar Error: ${dollar:.4f}")

    # Latent Space Analysis (PCA)
    pca = PCA(n_components=2)
    z_pca = pca.fit_transform(all_latents)

    # Plotting
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 8))

    scatter = plt.scatter(z_pca[:, 0], z_pca[:, 1], c=hurst_values, cmap='viridis', alpha=0.8)
    plt.colorbar(scatter, label='Hurst Exponent (H)')
    plt.title('Latent Space PCA colored by Roughness (H)')
    plt.xlabel('PC1')
    plt.ylabel('PC2')

    output_plot = 'writeup/figures/rough_latent_space.png'
    os.makedirs(os.path.dirname(output_plot), exist_ok=True)
    plt.savefig(output_plot)
    print(f"Saved latent plot to {output_plot}")

    # Correlation Analysis
    # Check correlation between PC1/PC2 and H
    corr_pc1 = np.corrcoef(z_pca[:, 0], hurst_values)[0, 1]
    corr_pc2 = np.corrcoef(z_pca[:, 1], hurst_values)[0, 1]

    print(f"Correlation (PC1, H): {corr_pc1:.4f}")
    print(f"Correlation (PC2, H): {corr_pc2:.4f}")

if __name__ == "__main__":
    evaluate_rough()
