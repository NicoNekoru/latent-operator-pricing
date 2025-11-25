import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models import NeuralOperator
from src.data_loader import MarketScraper, HestonSimulator

def visualize_latent_space():
    """
    Generates Latent Space visualizations using Matplotlib.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NeuralOperator(latent_dim=3).to(device)
    try:
        model.load_state_dict(torch.load('models/neural_operator.pth', map_location=device))
    except FileNotFoundError:
        print("Model not found. Train first.")
        return
    model.eval()

    # Load Data (All Tickers)
    tickers = ['^GSPC', '^NDX', '^RUT', '^DJI']
    scraper = MarketScraper(tickers=tickers, start_date='2010-01-01')
    market_data = scraper.process_data()

    z_list = []
    vol_list = []
    ticker_list = []
    dates = []

    print("Generating Latent Space for all tickers...")

    for ticker in tickers:
        if ticker not in market_data['Ticker'].values:
            continue

        df = market_data[market_data['Ticker'] == ticker].sort_index()

        # Stride for visualization speed
        stride = 5
        for i in range(30, len(df), stride):
            past_30 = df.iloc[i-30:i]
            x = np.stack([past_30['LogReturn'].values, past_30['RealizedVol'].values], axis=1).reshape(1, 30, 2)
            x_tensor = torch.tensor(x, dtype=torch.float32).to(device)

            with torch.no_grad():
                _, z = model(x_tensor)

            z_list.append(z.cpu().numpy().flatten())
            vol_list.append(df.iloc[i]['RealizedVol'])
            ticker_list.append(ticker)
            dates.append(df.index[i])

    z_arr = np.array(z_list)
    vol_arr = np.array(vol_list)
    ticker_arr = np.array(ticker_list)

    os.makedirs('plots', exist_ok=True)

    # 1. Latent Space 3D Scatter (Colored by Volatility)
    print("Plotting Latent Space Topology (Vol)...")
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(z_arr[:,0], z_arr[:,1], z_arr[:,2], c=vol_arr, cmap='RdBu_r', alpha=0.6, s=10)
    plt.colorbar(sc, label='Realized Volatility')
    ax.set_title("Latent Space Topology (All Indices - Volatility)")
    ax.set_xlabel('Latent Dim 1')
    ax.set_ylabel('Latent Dim 2')
    ax.set_zlabel('Latent Dim 3')
    plt.savefig('plots/latent_space_3d.png', dpi=300)
    plt.close()

    # 2. Latent Space 3D Scatter (Colored by Ticker)
    print("Plotting Latent Space Topology (Ticker)...")
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    colors = {'^GSPC': 'blue', '^NDX': 'orange', '^RUT': 'green', '^DJI': 'red'}

    for ticker in tickers:
        mask = ticker_arr == ticker
        if mask.sum() > 0:
            ax.scatter(z_arr[mask,0], z_arr[mask,1], z_arr[mask,2],
                       c=colors.get(ticker, 'black'), label=ticker, alpha=0.5, s=10)

    ax.set_title("Latent Space Topology (Colored by Ticker)")
    ax.set_xlabel('Latent Dim 1')
    ax.set_ylabel('Latent Dim 2')
    ax.set_zlabel('Latent Dim 3')
    ax.legend()
    plt.savefig('plots/latent_space_by_ticker.png', dpi=300)
    plt.close()

    # --- Advanced Visualizations ---

    # 3. PCA Projection (2D)
    print("Generating PCA Projection...")
    pca = PCA(n_components=2)
    z_pca = pca.fit_transform(z_arr)

    plt.figure(figsize=(10, 8))
    sc = plt.scatter(z_pca[:,0], z_pca[:,1], c=vol_arr, cmap='RdBu_r', alpha=0.6, s=10)
    plt.colorbar(sc, label='Realized Volatility')
    plt.title(f'PCA Projection of Latent Space (Explained Var: {pca.explained_variance_ratio_.sum():.2f})')
    plt.xlabel('PC 1')
    plt.ylabel('PC 2')
    plt.savefig('plots/pca_projection.png', dpi=300)
    plt.close()

    # 4. t-SNE Projection (2D)
    print("Generating t-SNE Projection (this may take a moment)...")
    # Subsample for t-SNE speed if needed, but ~4000 points is fine
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    z_tsne = tsne.fit_transform(z_arr)

    plt.figure(figsize=(10, 8))
    sc = plt.scatter(z_tsne[:,0], z_tsne[:,1], c=vol_arr, cmap='RdBu_r', alpha=0.6, s=10)
    plt.colorbar(sc, label='Realized Volatility')
    plt.title('t-SNE Manifold Projection')
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.savefig('plots/tsne_projection.png', dpi=300)
    plt.close()

    # 5. Parallel Coordinates Plot
    print("Generating Parallel Coordinates Plot...")
    # Create a DataFrame for plotting
    # Normalize Z for better visualization if ranges differ wildly (though they shouldn't)
    z_df = pd.DataFrame(z_arr, columns=[f'Z{i+1}' for i in range(z_arr.shape[1])])
    z_df['Volatility'] = vol_arr
    # Bin volatility for coloring lines
    z_df['Regime'] = pd.qcut(z_df['Volatility'], q=4, labels=['Low', 'Medium', 'High', 'Extreme'])

    plt.figure(figsize=(12, 6))
    pd.plotting.parallel_coordinates(z_df.sample(500), 'Regime', colormap='viridis', alpha=0.5)
    plt.title('Parallel Coordinates of Latent Dimensions by Volatility Regime')
    plt.xlabel('Latent Dimension')
    plt.ylabel('Value')
    plt.savefig('plots/parallel_coordinates.png', dpi=300)
    plt.close()

    # 6. Correlation Heatmap
    print("Generating Correlation Heatmap...")
    # We need to align Z with original features (Returns, Vol)
    # Re-construct a DF with Z and features
    # Note: z_arr corresponds to the loop above. We need to grab the features from that loop.
    # Ideally we should have saved them. Let's assume z_arr and vol_arr are aligned.
    # We only have Vol saved. Let's use Vol.

    corr_df = pd.DataFrame(z_arr, columns=[f'Latent_{i+1}' for i in range(z_arr.shape[1])])
    corr_df['Realized_Vol'] = vol_arr

    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_df.corr(), annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Latent Space Feature Correlation')
    plt.tight_layout()
    plt.savefig('plots/latent_correlation.png', dpi=300)
    plt.close()

    print("Advanced visualizations saved.")

def compare_indices():
    """
    Generates comparison graph for GSPC vs NDX using Matplotlib.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NeuralOperator(latent_dim=3).to(device)
    try:
        model.load_state_dict(torch.load('models/neural_operator.pth', map_location=device))
    except FileNotFoundError: return
    model.eval()

    print("Fetching Market Data for Comparison...")
    tickers = ['^GSPC', '^NDX', '^RUT', '^DJI']
    scraper = MarketScraper(tickers=tickers, start_date='2023-01-01')
    market_data = scraper.process_data()
    simulator = HestonSimulator()

    results = {}

    for ticker in tickers:
        print(f"Processing {ticker}...")
        if ticker not in market_data['Ticker'].values:
            print(f"Warning: No data for {ticker}")
            continue

        df = market_data[market_data['Ticker'] == ticker].sort_index()
        dates, true_prices, pred_prices, maes = [], [], [], []

        start_idx = 30
        end_idx = min(len(df), 230)

        for i in range(start_idx, end_idx):
            row = df.iloc[i]
            spot = row['Close']
            realized_vol = row['RealizedVol']

            # Ground Truth
            v0 = realized_vol ** 2
            surface = simulator.generate_surface(spot, v0, 2.0, v0, 0.3, -0.7)
            atm_true = surface[3]['Price']
            flat_true = np.array([p['Price'] for p in surface])

            # Predict
            past_30 = df.iloc[i-30:i]
            x = np.stack([past_30['LogReturn'].values, past_30['RealizedVol'].values], axis=1).reshape(1, 30, 2)
            x_tensor = torch.tensor(x, dtype=torch.float32).to(device)

            with torch.no_grad():
                y_pred, _ = model(x_tensor)
                y_pred = y_pred.cpu().numpy().flatten()

            dates.append(row.name)
            true_prices.append(atm_true)
            pred_prices.append(y_pred[3])
            maes.append(np.mean(np.abs(y_pred - flat_true)))

        results[ticker] = {'dates': dates, 'true': true_prices, 'pred': pred_prices, 'mae': maes}

    print("Generating Comparison Plot...")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12), sharex=True)
    colors = {'^GSPC': 'blue', '^NDX': 'orange', '^RUT': 'green', '^DJI': 'red'}

    for ticker in results:
        c = colors.get(ticker, 'black')
        ax1.plot(results[ticker]['dates'], results[ticker]['true'], label=f'{ticker} Truth', color=c, alpha=0.6)
        ax1.plot(results[ticker]['dates'], results[ticker]['pred'], label=f'{ticker} Pred', color=c, linestyle='--')
        ax2.plot(results[ticker]['dates'], results[ticker]['mae'], label=f'{ticker} MAE', color=c)

    ax1.set_title("Model Generalization: Multi-Index Comparison")
    ax1.set_ylabel("Normalized Price")
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)

    ax2.set_title("Mean Absolute Error (MAE)")
    ax2.set_ylabel("MAE Loss")
    ax2.set_xlabel("Date")
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("plots/index_comparison.png", dpi=300)
    print("Comparison plot saved.")

if __name__ == "__main__":
    visualize_latent_space()
    compare_indices()
