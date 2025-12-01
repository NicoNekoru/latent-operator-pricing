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

from src.models import DeepONet, get_standard_grid
from src.data_loader import MarketData, MacroData, HestonGenerator

def visualize_trajectories(z_arr, dates, ticker_arr, vol_arr):
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

    # Background: All points (faint)
    # We use the first 3 dimensions of Z

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
    plt.savefig('plots/crisis_trajectories.png', dpi=300)
    plt.close()

def visualize_clusters(z_arr, n_clusters=3):
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
    plt.savefig('plots/latent_clusters.png', dpi=300)
    plt.close()

    return labels

def visualize_latent_space():
    """
    Generates Latent Space visualizations using Matplotlib.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ManifoldAutoencoder(input_dim=6, latent_dim=3).to(device)
    try:
        model.load_state_dict(torch.load('models/neural_operator.pth', map_location=device))
    except FileNotFoundError:
        print("Model not found. Train first.")
        return
    model.eval()

    # Load Data (All Tickers)
    tickers = ['^GSPC', '^NDX', '^RUT', '^DJI']
    print("Fetching Market Data...")
    market = MarketData(tickers=tickers, start_date='2010-01-01').fetch()
    macro = MacroData(start_date='2010-01-01').fetch()
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

        # Stride for visualization speed
        stride = 5

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
            ax.scatter(z_arr[mask,0], z_arr[mask,1], z_arr[mask,2],
                       c=colors.get(ticker, 'black'), label=ticker, alpha=0.5, s=10)

    ax.set_title("Latent Space Topology (Colored by Ticker)")
    ax.set_xlabel('Latent Dim 1')
    ax.set_ylabel('Latent Dim 2')
    ax.set_zlabel('Latent Dim 3')
    ax.legend()
    plt.tight_layout()
    plt.savefig('plots/latent_space_by_ticker.png', dpi=300)
    plt.close()

    return labels

def visualize_velocity_field(z_arr, vol_arr):
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
    plt.savefig('plots/latent_velocity.png', dpi=300)
    plt.close()

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
    plt.savefig('plots/pca_projection.png', dpi=300)
    plt.close()

    # 7. Correlation Heatmap
    print("Generating Correlation Heatmap...")

    corr_df = pd.DataFrame(z_arr, columns=[f'Latent_{i+1}' for i in range(z_arr.shape[1])])
    corr_df['Realized_Vol'] = vol_arr

    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_df.corr(), annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Latent Space Feature Correlation')
    plt.tight_layout()
    plt.savefig('plots/latent_correlation.png', dpi=300)
    plt.close()

    # 8. Error Heatmap
    visualize_error_heatmap(model, device)

    # 9. Velocity vs Drawdown
    visualize_velocity_drawdown(z_arr, dates, ticker_arr)

    print("Advanced visualizations saved.")

def visualize_error_heatmap(model, device):
    """
    Plots Mean Absolute Error (MAE) as a 2D heatmap (Moneyness vs. Maturity).
    """
    print("Generating Error Heatmap...")
    # Generate a synthetic test set covering the grid
    # Moneyness: 0.8 to 1.2 (7 points)
    # Maturity: 0.1, 0.5, 1.0 (3 points) - indices 0, 1, 2

    # We need to simulate many Heston surfaces and compare Model(Encoder(Surface)) vs Surface?
    # No, Model(Encoder(History)) vs Surface.
    # We can use the validation set logic.

    # For simplicity, we'll use the "HestonGenerator" to generate random valid surfaces
    # and feed them into the Decoder directly?
    # No, the model is History -> Price.

    # We will use the 'compare_indices' logic but aggregate errors by (Maturity, Strike).

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

    plt.figure(figsize=(8, 6))
    sns.heatmap(avg_mae_grid, annot=True, fmt=".4f", cmap='Reds',
                xticklabels=['0.8', '0.9', '0.95', '1.0', '1.05', '1.1', '1.2'],
                yticklabels=['1M', '3M', '6M'])
    plt.title("Reconstruction Error Heatmap (MAE)")
    plt.xlabel("Moneyness (K/S)")
    plt.ylabel("Maturity")
    plt.tight_layout()
    plt.savefig('plots/error_heatmap.png', dpi=300)
    plt.close()

def visualize_velocity_drawdown(z_arr, dates, ticker_arr):
    """
    Scatter plot of Latent Velocity vs. Next 5-Day Return.
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
    # We need the original price data.
    # We can approximate using the dates if we re-fetch, but we have dates.
    # Let's fetch just GSPC prices to align.

    market = MarketData(tickers=['^GSPC'], start_date='2006-01-01').fetch()
    prices = market[market['Ticker'] == '^GSPC']['Close']

    # Align dates
    # velocity[i] corresponds to transition from date[i] to date[i+1]
    # We want to compare velocity at t with return from t to t+5.

    future_returns = []
    aligned_velocity = []

    for i in range(len(dates_gspc)-6):
        date = dates_gspc[i]
        if date in prices.index:
            # Find price at t and t+5 (approx)
            # Since dates_gspc might be strided, we look up in prices
            try:
                idx = prices.index.get_loc(date)
                p_t = prices.iloc[idx]
                p_t5 = prices.iloc[min(idx+5, len(prices)-1)]
                ret = (p_t5 - p_t) / p_t

                future_returns.append(ret)
                aligned_velocity.append(velocity[i])
            except KeyError:
                continue

    plt.figure(figsize=(8, 6))
    plt.scatter(aligned_velocity, future_returns, alpha=0.3, s=10)
    plt.axhline(0, color='black', linestyle='--', linewidth=0.8)
    plt.title("Latent Velocity vs. Future 5-Day Return")
    plt.xlabel("Latent Velocity $||v_t||$")
    plt.ylabel("Next 5-Day Return")

    # Add trend line
    z = np.polyfit(aligned_velocity, future_returns, 1)
    p = np.poly1d(z)
    plt.plot(aligned_velocity, p(aligned_velocity), "r--", alpha=0.8)

    plt.tight_layout()
    plt.savefig('plots/velocity_drawdown.png', dpi=300)
    plt.close()



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
            atm_true = surface[3] # 3rd element is ATM? Need to check generate order.
            # generate returns array of 21 prices.
            # maturities [1,3,6] * moneyness [0.8...1.2] (7)
            # 3 months is index 1. ATM is index 3 in moneyness.
            # So index = 1*7 + 3 = 10.
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
    ax3.set_ylabel("MRE (%)") # It's a ratio, but often interpreted as %
    ax3.set_xlabel("Date")
    ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("plots/index_comparison.png", dpi=300)
    print("Comparison plot saved.")

if __name__ == "__main__":
    visualize_latent_space()
    compare_indices()
