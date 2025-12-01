import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

from src.models import DeepONet, get_standard_grid
from src.data_loader import MarketData, MacroData, HestonGenerator
from src.analysis.visualizations import (
    visualize_trajectories, visualize_clusters, visualize_velocity_field,
    visualize_error_heatmap, visualize_velocity_drawdown,
    plot_pca_projection, plot_correlation_heatmap, plot_index_comparison
)

def visualize_latent_space():
    """
    Generates Latent Space visualizations using Matplotlib.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DeepONet(input_channels=6, latent_dim=16).to(device)
    base_grid = get_standard_grid(device)
    try:
        model.load_state_dict(torch.load('models/deeponet.pth', map_location=device))
    except FileNotFoundError:
        print("Model not found. Train first.")
        return
    model.eval()

    # Load Data (All Tickers)
    tickers = ['^GSPC', '^NDX', '^RUT', '^DJI']
    print("Fetching Market Data...")
    market = MarketData(tickers=tickers, start_date='2006-01-01').fetch() # Expanded range for 2008
    macro = MacroData(start_date='2006-01-01').fetch()
    merged_data = market.join(macro, how='left').ffill().dropna()

    z_list = []
    vol_list = []
    ticker_list = []
    dates = []

    print("Generating Latent Space for all tickers...")

    for ticker in tickers:
        if ticker not in merged_data['Ticker'].values:
            continue

        df = merged_data[merged_data['Ticker'] == ticker].sort_index()

        # Stride for visualization speed (keep low for trajectory accuracy)
        stride = 1

        # Pre-calculate Volume Feature
        vol_feature = np.log(df['Volume'] + 1) / 20.0

        for i in range(30, len(df), stride):
            window = df.iloc[i-30:i]
            window_vol = vol_feature.iloc[i-30:i]

            # Construct 6-channel input
            features = np.stack([
                window['LogReturn'].values,
                window['RealizedVol'].values,
                window['VIX'].values,
                window_vol.values,
                window['TNX'].values,
                window['Buffett_Ind'].values
            ], axis=1)

            x = features.reshape(1, 30, 6)
            x_tensor = torch.tensor(x, dtype=torch.float32).to(device)

            with torch.no_grad():
                _, z = model(x_tensor, base_grid.expand(1, -1, -1))

            z_list.append(z.cpu().numpy().flatten())
            vol_list.append(df.iloc[i]['RealizedVol'])
            ticker_list.append(ticker)
            dates.append(df.index[i])

    z_arr = np.array(z_list)
    vol_arr = np.array(vol_list)
    ticker_arr = np.array(ticker_list)
    dates = np.array(dates)

    os.makedirs('plots', exist_ok=True)

    # --- Standard Visualizations ---

    # 1. Latent Space 3D Scatter (Colored by Volatility)
    print("Plotting Latent Space Topology (Vol)...")
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(z_arr[:,0], z_arr[:,1], z_arr[:,2], c=vol_arr, cmap='RdBu_r', alpha=0.6, s=2)
    plt.colorbar(sc, label='Realized Volatility')
    ax.set_title("Latent Space Topology (All Indices - Volatility)")
    ax.set_xlabel('Latent Dim 1')
    ax.set_ylabel('Latent Dim 2')
    ax.set_zlabel('Latent Dim 3')
    plt.tight_layout()
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
            # Downsample for ticker plot to avoid clutter
            ax.scatter(z_arr[mask][::5,0], z_arr[mask][::5,1], z_arr[mask][::5,2],
                       c=colors.get(ticker, 'black'), label=ticker, alpha=0.5, s=2)

    ax.set_title("Latent Space Topology (Colored by Ticker)")
    ax.set_xlabel('Latent Dim 1')
    ax.set_ylabel('Latent Dim 2')
    ax.set_zlabel('Latent Dim 3')
    ax.legend()
    plt.tight_layout()
    plt.savefig('plots/latent_space_by_ticker.png', dpi=300)
    plt.close()

    # --- Advanced Visualizations ---

    # 3. Crisis Trajectories
    visualize_trajectories(z_arr, dates, ticker_arr, vol_arr)

    # 4. K-Means Clusters
    visualize_clusters(z_arr, n_clusters=3)

    # 5. Velocity Field
    visualize_velocity_field(z_arr, vol_arr)

    # 6. PCA Projection (2D)
    plot_pca_projection(z_arr, vol_arr)

    # 7. Correlation Heatmap
    plot_correlation_heatmap(z_arr, vol_arr)

    # 8. Error Heatmap
    generate_error_heatmap_data(model, device)

    # 9. Velocity vs Drawdown
    generate_velocity_drawdown_data(z_arr, dates, ticker_arr)

    print("Advanced visualizations saved.")

def generate_error_heatmap_data(model, device):
    """
    Generates data for Error Heatmap and calls visualization.
    """
    print("Generating Error Heatmap...")
    tickers = ['^GSPC']
    market = MarketData(tickers=tickers, start_date='2010-01-01').fetch()
    macro = MacroData(start_date='2010-01-01').fetch()
    merged_data = market.join(macro, how='left').ffill().dropna()

    base_grid = get_standard_grid(device)
    simulator = HestonGenerator()

    # Accumulate errors per grid point (21 points)
    total_mae = np.zeros(21)
    count = 0

    df = merged_data[merged_data['Ticker'] == '^GSPC']
    vol_feature = np.log(df['Volume'] + 1) / 20.0

    # Sample random points to save time
    indices = np.random.choice(range(30, len(df)), size=500, replace=False)

    for i in indices:
        row = df.iloc[i]
        spot = row['Close']
        realized_vol = row['RealizedVol']

        # Ground Truth
        v0 = realized_vol ** 2
        simulator.setup_engine(spot, v0, 2.0, v0, 0.3, -0.7)
        surface = simulator.generate(spot)

        # Predict
        window = df.iloc[i-30:i]
        window_vol = vol_feature.iloc[i-30:i]

        features = np.stack([
            window['LogReturn'].values,
            window['RealizedVol'].values,
            window['VIX'].values,
            window_vol.values,
            window['TNX'].values,
            window['Buffett_Ind'].values
        ], axis=1)

        x = features.reshape(1, 30, 6)
        x_tensor = torch.tensor(x, dtype=torch.float32).to(device)

        with torch.no_grad():
            y_pred, _ = model(x_tensor, base_grid.expand(1, -1, -1))
            y_pred = y_pred.cpu().numpy().flatten()

        total_mae += np.abs(y_pred - surface)
        count += 1

    avg_mae = total_mae / count
    avg_mae_grid = avg_mae.reshape(3, 7)

    visualize_error_heatmap(avg_mae_grid)

def generate_velocity_drawdown_data(z_arr, dates, ticker_arr):
    """
    Generates data for Velocity vs Drawdown and calls visualization.
    """
    print("Generating Velocity vs Drawdown Scatter...")

    # Filter for GSPC
    mask = ticker_arr == '^GSPC'
    z_gspc = z_arr[mask]
    dates_gspc = pd.to_datetime(dates[mask])

    # Calculate Velocity
    dz = np.diff(z_gspc, axis=0)
    velocity = np.linalg.norm(dz, axis=1)

    # Calculate Future Returns (5-day)
    market = MarketData(tickers=['^GSPC'], start_date='2006-01-01').fetch()
    prices = market[market['Ticker'] == '^GSPC']['Close']

    future_returns = []
    aligned_velocity = []

    for i in range(len(dates_gspc)-6):
        date = dates_gspc[i]
        if date in prices.index:
            try:
                idx = prices.index.get_loc(date)
                p_t = prices.iloc[idx]
                p_t5 = prices.iloc[min(idx+5, len(prices)-1)]
                ret = (p_t5 - p_t) / p_t

                future_returns.append(ret)
                aligned_velocity.append(velocity[i])
            except KeyError:
                continue

    visualize_velocity_drawdown(aligned_velocity, future_returns)

def compare_indices():
    """
    Generates comparison graph for GSPC vs NDX using Matplotlib.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DeepONet(input_channels=6, latent_dim=16).to(device)
    base_grid = get_standard_grid(device)
    try:
        model.load_state_dict(torch.load('models/deeponet.pth', map_location=device))
    except FileNotFoundError: return
    model.eval()

    print("Fetching Market Data for Comparison...")
    tickers = ['^GSPC', '^NDX', '^RUT', '^DJI']
    market = MarketData(tickers=tickers, start_date='2023-01-01').fetch()
    macro = MacroData(start_date='2023-01-01').fetch()
    merged_data = market.join(macro, how='left').ffill().dropna()

    simulator = HestonGenerator()

    results = {}

    for ticker in tickers:
        print(f"Processing {ticker}...")
        if ticker not in merged_data['Ticker'].values:
            print(f"Warning: No data for {ticker}")
            continue

        df = merged_data[merged_data['Ticker'] == ticker].sort_index()
        dates, true_prices, pred_prices, maes, mres = [], [], [], [], []

        start_idx = 30
        end_idx = min(len(df), 230)

        vol_feature = np.log(df['Volume'] + 1) / 20.0

        for i in range(start_idx, end_idx):
            row = df.iloc[i]
            spot = row['Close']
            realized_vol = row['RealizedVol']

            # Ground Truth
            v0 = realized_vol ** 2
            simulator.setup_engine(spot, v0, 2.0, v0, 0.3, -0.7)
            surface = simulator.generate(spot)
            # ATM 3-month is index 10 (1*7 + 3)
            atm_true = surface[10]

            # Predict
            window = df.iloc[i-30:i]
            window_vol = vol_feature.iloc[i-30:i]

            features = np.stack([
                window['LogReturn'].values,
                window['RealizedVol'].values,
                window['VIX'].values,
                window_vol.values,
                window['TNX'].values,
                window['Buffett_Ind'].values
            ], axis=1)

            x = features.reshape(1, 30, 6)
            x_tensor = torch.tensor(x, dtype=torch.float32).to(device)

            with torch.no_grad():
                y_pred, _ = model(x_tensor, base_grid.expand(1, -1, -1))
                y_pred = y_pred.cpu().numpy().flatten()

            dates.append(row.name)
            true_prices.append(atm_true)
            pred_prices.append(y_pred[10]) # ATM 3-month
            maes.append(np.mean(np.abs(y_pred - surface)))

            # MRE: Mean Relative Error (avoid div by zero)
            mres.append(np.mean(np.abs((y_pred - surface) / (surface + 1e-9))))

        results[ticker] = {'dates': dates, 'true': true_prices, 'pred': pred_prices, 'mae': maes, 'mre': mres}

    plot_index_comparison(results)

if __name__ == "__main__":
    visualize_latent_space()
    compare_indices()
