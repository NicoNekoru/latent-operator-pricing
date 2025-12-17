import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import pandas as pd
from torch.utils.data import DataLoader
import sys
import os

# Add root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.dataset import OptionDataset
from src.models import SpectralDeepONet, get_standard_grid

def analyze_latent_space():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. Load Data
    dataset_path = '../../data/processed_dataset.parquet'

    try:
        val_dataset = OptionDataset(dataset_path, mode='val') # This contains 2022 data (Out of Sample)
        train_dataset = OptionDataset(dataset_path, mode='train') # 2019-2021
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # Use a subset for visualization to avoid clutter
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=True)

    # 2. Load Model
    model = SpectralDeepONet(input_channels=6, latent_dim=64).to(device)
    model_path = '../../training/models/deeponet.pth'

    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("Model weights loaded successfully.")
    except FileNotFoundError:
        print(f"Model weights not found at {model_path}! Please train the model first.")
        return

    model.eval()

    # 3. Extract Latent Vectors
    latents = []
    domains = []
    phases = [] # 'Train' or 'Test'

    base_grid = get_standard_grid(device)

    print("Extracting Latent Features...")

    # Extract from Train (Subset 500 samples)
    count = 0
    with torch.no_grad():
        for x, _, _, domain in train_loader:
            x = x.to(device)
            bs = x.size(0)
            grid = base_grid.expand(bs, -1, -1)

            # Forward pass
            _, _, b, _ = model(x, grid, alpha=0.0) # b is (Batch, LatentDim)

            latents.append(b.cpu().numpy())
            domains.append(domain.numpy())
            phases.extend(['Train'] * bs)

            count += bs
            if count > 1000: break

    # Extract from Val (Subset 500 samples)
    count = 0
    with torch.no_grad():
        for x, _, _, domain in val_loader:
            x = x.to(device)
            bs = x.size(0)
            grid = base_grid.expand(bs, -1, -1)

            _, _, b, _ = model(x, grid, alpha=0.0)

            latents.append(b.cpu().numpy())
            domains.append(domain.numpy())
            phases.extend(['Test (2022)'] * bs)

            count += bs
            if count > 1000: break

    # Concatenate
    latents = np.concatenate(latents, axis=0) # (N, 32)
    domains = np.concatenate(domains, axis=0) # (N,)
    domain_labels = ['Post-2020' if d == 1.0 else 'Pre-2020' for d in domains]

    df_latent = pd.DataFrame(latents)

    # 4. Dimensionality Reduction (PCA)
    print("Running PCA...")
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(latents)

    df_viz = pd.DataFrame({
        'PCA_1': pca_result[:, 0],
        'PCA_2': pca_result[:, 1],
        'Regime': domain_labels,
        'Phase': phases
    })

    # 5. t-SNE (Standard for Manifold Viz)
    print("Running t-SNE...")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    tsne_result = tsne.fit_transform(latents)

    df_viz['tSNE_1'] = tsne_result[:, 0]
    df_viz['tSNE_2'] = tsne_result[:, 1]

    # 6. Plotting
    os.makedirs('../../analysis/plots', exist_ok=True)

    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        data=df_viz, x='PCA_1', y='PCA_2',
        hue='Regime', style='Phase',
        palette='viridis', alpha=0.7
    )
    plt.title('Spectral Operator Latent Space (PCA)')
    plt.savefig('../../analysis/plots/latent_space_pca.png')
    plt.close()

    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        data=df_viz, x='tSNE_1', y='tSNE_2',
        hue='Regime', style='Phase',
        palette='rocket', alpha=0.7
    )
    plt.title('Spectral Operator Latent Space (t-SNE)')
    plt.savefig('../../analysis/plots/latent_space_tsne.png')
    plt.close()

    print("Analysis Complete. Plots saved to analysis/plots/.")

if __name__ == "__main__":
    analyze_latent_space()
