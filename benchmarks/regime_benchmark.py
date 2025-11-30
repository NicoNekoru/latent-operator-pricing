import torch
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import os

from src.models import NeuralOperator

def load_data(path='data/processed_dataset.parquet'):
    print(f"Loading data from {path}...")
    df = pd.read_parquet(path)

    # Extract features
    # Input_Features is a column of arrays. Stack them.
    features_list = df['Input_Features'].tolist()
    X = np.stack(features_list) # Shape (N, 180)

    # Reshape to (N, 30, 6) to easily extract specific features for baseline
    X_reshaped = X.reshape(-1, 30, 6)

    return df, X, X_reshaped

def load_model(path='models/neural_operator.pth', input_dim=6, latent_dim=3):
    print(f"Loading model from {path}...")
    model = NeuralOperator(input_dim=input_dim, latent_dim=latent_dim)

    if os.path.exists(path):
        model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
    else:
        print("Warning: Model file not found. Using untrained model for demonstration.")

    model.eval()
    return model

def get_latent_vectors(model, X):
    print("Generating latent vectors...")
    X_tensor = torch.FloatTensor(X)
    with torch.no_grad():
        # The model's encoder expects flattened input or (N, 30, 6) which it flattens.
        # Our X is (N, 180), so it's already flattened.
        z = model.encoder(X_tensor)
    return z.numpy()

def plot_clusters(z, labels, title, filename):
    pca = PCA(n_components=2)
    z_pca = pca.fit_transform(z)

    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(z_pca[:, 0], z_pca[:, 1], c=labels, cmap='viridis', alpha=0.6)
    plt.colorbar(scatter, label='Cluster')
    plt.title(title)
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.grid(True, alpha=0.3)

    os.makedirs('plots', exist_ok=True)
    plt.savefig(f'plots/{filename}')
    print(f"Saved plot to plots/{filename}")
    plt.close()

def benchmark_regimes(df, z, X_reshaped):
    print("\n--- Starting Regime Benchmark ---\n")

    # 1. Prepare Data

    # A. Neural Latent Space
    scaler_z = StandardScaler()
    z_scaled = scaler_z.fit_transform(z)

    # B. Baseline Features (RealizedVol, VIX)
    # Take the most recent value in the window (index -1)
    # Feature indices: 1 = RealizedVol, 2 = VIX
    baseline_features = X_reshaped[:, -1, [1, 2]]
    scaler_b = StandardScaler()
    b_scaled = scaler_b.fit_transform(baseline_features)

    # 2. Clustering
    n_clusters = 3
    print(f"Clustering with K={n_clusters}...")

    kmeans_z = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels_z = kmeans_z.fit_predict(z_scaled)

    kmeans_b = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels_b = kmeans_b.fit_predict(b_scaled)

    # Plotting
    plot_clusters(z_scaled, labels_z, "Neural Latent Space Clusters", "neural_clusters.png")
    plot_clusters(b_scaled, labels_b, "Baseline (Vol+VIX) Clusters", "baseline_clusters.png")

    # 3. Evaluation

    # Metric: Average Realized Volatility per Cluster
    # We use the actual RealizedVol from the data (feature index 1)
    realized_vol = X_reshaped[:, -1, 1]

    results = []

    for name, labels in [("Neural Latent", labels_z), ("Baseline (Vol+VIX)", labels_b)]:
        sil_score = silhouette_score(z_scaled if name == "Neural Latent" else b_scaled, labels)

        # Calculate stats per cluster
        cluster_stats = []
        for k in range(n_clusters):
            mask = labels == k
            avg_vol = realized_vol[mask].mean()
            count = mask.sum()
            cluster_stats.append({'cluster': k, 'avg_vol': avg_vol, 'count': count})

        # Sort clusters by avg_vol to align them (Low, Med, High)
        cluster_stats.sort(key=lambda x: x['avg_vol'])

        # Calculate "Separation" (Ratio of High Vol / Low Vol)
        separation = cluster_stats[-1]['avg_vol'] / (cluster_stats[0]['avg_vol'] + 1e-9)

        results.append({
            'Method': name,
            'Silhouette': sil_score,
            'Low Vol': cluster_stats[0]['avg_vol'],
            'Med Vol': cluster_stats[1]['avg_vol'],
            'High Vol': cluster_stats[2]['avg_vol'],
            'Separation': separation
        })

    # 4. Print Report
    results_df = pd.DataFrame(results)
    print("\nResults Summary:")
    print(results_df.to_string(index=False))

    print("\nDetailed Interpretation:")
    for i, row in results_df.iterrows():
        print(f"\n{row['Method']}:")
        print(f"  Silhouette Score: {row['Silhouette']:.4f} (Higher is better cluster definition)")
        print(f"  Regime Separation (High/Low Vol): {row['Separation']:.2f}x")
        print(f"  Avg Vol by Regime: Low={row['Low Vol']:.4f}, Med={row['Med Vol']:.4f}, High={row['High Vol']:.4f}")

if __name__ == "__main__":
    try:
        df, X, X_reshaped = load_data()
        model = load_model()
        z = get_latent_vectors(model, X)
        benchmark_regimes(df, z, X_reshaped)
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
