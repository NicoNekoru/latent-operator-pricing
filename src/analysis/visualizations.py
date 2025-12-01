import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from matplotlib.colors import ListedColormap
import os

def visualize_trajectories(z_arr, dates, ticker_arr, vol_arr, save_path='plots/crisis_trajectories.png'):
    """
    Plots the specific trajectories of crisis periods in the Latent Space.
    """
    print("Plotting Crisis Trajectories...")

    # Define Crisis Periods
    periods = {
        '2008 Crisis': ('2008-09-01', '2009-03-01'),
        '2020 COVID': ('2020-02-01', '2020-04-01'),
        '2022 Inflation': ('2022-01-01', '2022-06-01')
    }

    fig = plt.figure(figsize=(15, 5))

    for i, (name, (start, end)) in enumerate(periods.items()):
        ax = fig.add_subplot(1, 3, i+1, projection='3d')

        # Plot Background (All Data)
        ax.scatter(z_arr[:,0], z_arr[:,1], z_arr[:,2], c='lightgray', alpha=0.1, s=1)

        # Filter for Period and Ticker (e.g. GSPC)
        mask_ticker = ticker_arr == '^GSPC'
        mask_date = (pd.to_datetime(dates) >= start) & (pd.to_datetime(dates) <= end)
        mask = mask_ticker & mask_date

        z_seg = z_arr[mask]
        vol_seg = vol_arr[mask]

        if len(z_seg) > 0:
            # Plot Trajectory Line
            ax.plot(z_seg[:,0], z_seg[:,1], z_seg[:,2], color='black', linewidth=1.5, alpha=0.7)

            # Plot Points colored by Volatility
            sc = ax.scatter(z_seg[:,0], z_seg[:,1], z_seg[:,2], c=vol_seg, cmap='inferno', s=20)

            # Mark Start and End
            ax.scatter(z_seg[0,0], z_seg[0,1], z_seg[0,2], c='green', s=100, marker='^', label='Start')
            ax.scatter(z_seg[-1,0], z_seg[-1,1], z_seg[-1,2], c='red', s=100, marker='v', label='End')

        ax.set_title(f"{name} ({start} - {end})")
        ax.set_xlabel('Z1')
        ax.set_ylabel('Z2')
        ax.set_zlabel('Z3')

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()

def visualize_clusters(z_arr, n_clusters=3, save_path='plots/latent_clusters.png'):
    """
    Performs K-Means clustering on the Latent Space and visualizes regimes.
    """
    print(f"Performing K-Means Clustering (k={n_clusters})...")

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(z_arr)

    # 3D Scatter of Clusters
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Create a custom colormap for regimes
    cmap = ListedColormap(['green', 'orange', 'red'])

    sc = ax.scatter(z_arr[:,0], z_arr[:,1], z_arr[:,2], c=labels, cmap=cmap, alpha=0.6, s=10)

    # Add Legend manually
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='green', label='Regime 0 (Calm?)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', label='Regime 1 (Transition?)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red', label='Regime 2 (Crisis?)')
    ]
    ax.legend(handles=legend_elements)

    ax.set_title(f"Latent Market Regimes (K-Means, k={n_clusters})")
    ax.set_xlabel('Z1')
    ax.set_ylabel('Z2')
    ax.set_zlabel('Z3')

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()

    return labels

def visualize_velocity_field(z_arr, vol_arr, save_path='plots/latent_velocity.png'):
    """
    Plots the velocity field (quiver) of the latent space to show flow dynamics.
    """
    print("Generating Latent Velocity Field...")

    # Calculate Velocity (dZ)
    dz = np.diff(z_arr, axis=0)
    # Pad last element to match shape
    dz = np.vstack([dz, dz[-1]])

    # Subsample for clarity (too many arrows is messy)
    stride = 10
    z_sub = z_arr[::stride]
    dz_sub = dz[::stride]
    vol_sub = vol_arr[::stride]

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Quiver Plot
    # Length of arrow proportional to speed
    ax.quiver(z_sub[:,0], z_sub[:,1], z_sub[:,2],
              dz_sub[:,0], dz_sub[:,1], dz_sub[:,2],
              length=0.5, normalize=True, alpha=0.4, color='gray')

    # Scatter on top for context
    sc = ax.scatter(z_sub[:,0], z_sub[:,1], z_sub[:,2], c=vol_sub, cmap='coolwarm', s=5)

    ax.set_title("Latent Space Velocity Field (Market Flow)")
    ax.set_xlabel('Z1')
    ax.set_ylabel('Z2')
    ax.set_zlabel('Z3')

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()

def visualize_error_heatmap(avg_mae_grid, save_path='plots/error_heatmap.png'):
    """
    Plots Mean Absolute Error (MAE) as a 2D heatmap (Moneyness vs. Maturity).
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(avg_mae_grid, annot=True, fmt=".4f", cmap='Reds',
                xticklabels=['0.8', '0.9', '0.95', '1.0', '1.05', '1.1', '1.2'],
                yticklabels=['1M', '3M', '6M'])
    plt.title("Reconstruction Error Heatmap (MAE)")
    plt.xlabel("Moneyness (K/S)")
    plt.ylabel("Maturity")
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()

def visualize_velocity_drawdown(velocity, future_returns, save_path='plots/velocity_drawdown.png'):
    """
    Scatter plot of Latent Velocity vs. Next 5-Day Return.
    """
    plt.figure(figsize=(8, 6))
    plt.scatter(velocity, future_returns, alpha=0.3, s=10)
    plt.axhline(0, color='black', linestyle='--', linewidth=0.8)
    plt.title("Latent Velocity vs. Future 5-Day Return")
    plt.xlabel("Latent Velocity $||v_t||$")
    plt.ylabel("Next 5-Day Return")

    # Add trend line
    z = np.polyfit(velocity, future_returns, 1)
    p = np.poly1d(z)
    plt.plot(velocity, p(velocity), "r--", alpha=0.8)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_pca_projection(z_arr, vol_arr, save_path='plots/pca_projection.png'):
    print("Generating PCA Projection...")
    pca = PCA(n_components=2)
    z_pca = pca.fit_transform(z_arr)

    plt.figure(figsize=(10, 8))
    sc = plt.scatter(z_pca[:,0], z_pca[:,1], c=vol_arr, cmap='RdBu_r', alpha=0.6, s=2)
    plt.colorbar(sc, label='Realized Volatility')
    plt.title(f'PCA Projection of Latent Space (Explained Var: {pca.explained_variance_ratio_.sum():.2f})')
    plt.xlabel('PC 1')
    plt.ylabel('PC 2')
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_correlation_heatmap(z_arr, vol_arr, save_path='plots/latent_correlation.png'):
    print("Generating Correlation Heatmap...")

    corr_df = pd.DataFrame(z_arr, columns=[f'Latent_{i+1}' for i in range(z_arr.shape[1])])
    corr_df['Realized_Vol'] = vol_arr

    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_df.corr(), annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Latent Space Feature Correlation')
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_index_comparison(results, save_path='plots/index_comparison.png'):
    print("Generating Comparison Plot...")
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 16), sharex=True)
    colors = {'^GSPC': 'blue', '^NDX': 'orange', '^RUT': 'green', '^DJI': 'red'}

    for ticker in results:
        c = colors.get(ticker, 'black')
        ax1.plot(results[ticker]['dates'], results[ticker]['true'], label=f'{ticker} Truth', color=c, alpha=0.6)
        ax1.plot(results[ticker]['dates'], results[ticker]['pred'], label=f'{ticker} Pred', color=c, linestyle='--')
        ax2.plot(results[ticker]['dates'], results[ticker]['mae'], label=f'{ticker} MAE', color=c)
        ax3.plot(results[ticker]['dates'], results[ticker]['mre'], label=f'{ticker} MRE', color=c)

    ax1.set_title("Model Generalization: Multi-Index Comparison")
    ax1.set_ylabel("Normalized Price (ATM 3-Month)")
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)

    ax2.set_title("Mean Absolute Error (MAE)")
    ax2.set_ylabel("MAE Loss")
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(True, alpha=0.3)

    ax3.set_title("Mean Relative Error (MRE)")
    ax3.set_ylabel("MRE (%)")
    ax3.set_xlabel("Date")
    ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
