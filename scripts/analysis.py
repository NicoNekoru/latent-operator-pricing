import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
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

    # Load Data (Sample)
    scraper = MarketScraper(tickers=['^GSPC'], start_date='2010-01-01')
    market_data = scraper.process_data()
    df = market_data[market_data['Ticker'] == '^GSPC']

    # Generate Z
    z_list = []
    vol_list = []
    dates = []

    print("Generating Latent Space...")
    for i in range(30, len(df), 5): # Stride 5 for speed
        past_30 = df.iloc[i-30:i]
        x = np.stack([past_30['LogReturn'].values, past_30['RealizedVol'].values], axis=1).reshape(1, 30, 2)
        x_tensor = torch.tensor(x, dtype=torch.float32).to(device)

        with torch.no_grad():
            _, z = model(x_tensor)

        z_list.append(z.cpu().numpy().flatten())
        vol_list.append(df.iloc[i]['RealizedVol'])
        dates.append(df.index[i])

    z_arr = np.array(z_list)
    vol_arr = np.array(vol_list)

    os.makedirs('plots', exist_ok=True)

    # 1. Latent Space 3D Scatter
    print("Plotting Latent Space Topology...")
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(z_arr[:,0], z_arr[:,1], z_arr[:,2], c=vol_arr, cmap='RdBu_r', alpha=0.6)
    plt.colorbar(sc, label='Realized Volatility')
    ax.set_title("Latent Space Topology (Colored by Volatility)")
    ax.set_xlabel('Latent Dim 1')
    ax.set_ylabel('Latent Dim 2')
    ax.set_zlabel('Latent Dim 3')
    plt.savefig('plots/latent_space_3d.png', dpi=300)
    plt.close()

    # 2. Trajectory (2023)
    print("Plotting Market Trajectory...")
    mask_2023 = [str(d).startswith('2023') for d in dates]
    z_2023 = z_arr[mask_2023]

    if len(z_2023) > 0:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        # Color by time index
        time_idx = np.arange(len(z_2023))
        # Plot line segments
        for i in range(len(z_2023)-1):
            ax.plot(z_2023[i:i+2,0], z_2023[i:i+2,1], z_2023[i:i+2,2],
                    color=plt.cm.viridis(i/len(z_2023)))

        ax.set_title("Market Trajectory (2023)")
        ax.set_xlabel('Latent Dim 1')
        ax.set_ylabel('Latent Dim 2')
        ax.set_zlabel('Latent Dim 3')
        # Add colorbar for time
        sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=0, vmax=len(z_2023)))
        plt.colorbar(sm, label='Trading Days in 2023')
        plt.savefig('plots/trajectory_2023.png', dpi=300)
        plt.close()

    # 3. Latent Landscape (Heatmap)
    print("Generating Landscape...")
    grid_size = 50
    z1 = np.linspace(z_arr[:,0].min(), z_arr[:,0].max(), grid_size)
    z2 = np.linspace(z_arr[:,1].min(), z_arr[:,1].max(), grid_size)
    z3_mean = z_arr[:,2].mean()

    Z1, Z2 = np.meshgrid(z1, z2)
    prices = np.zeros((grid_size, grid_size))

    for i in range(grid_size):
        for j in range(grid_size):
            z_vec = torch.tensor([[Z1[i,j], Z2[i,j], z3_mean]], dtype=torch.float32).to(device)
            with torch.no_grad():
                p = model.decoder(z_vec).cpu().numpy().flatten()
            prices[i,j] = p[3] # ATM Price

    plt.figure(figsize=(10, 8))
    plt.contourf(Z1, Z2, prices, levels=50, cmap='viridis')
    plt.colorbar(label='ATM Option Price')
    plt.title('Latent Physics Landscape (ATM Price Surface)')
    plt.xlabel('Latent Dim 1')
    plt.ylabel('Latent Dim 2')
    plt.savefig('plots/latent_landscape.png', dpi=300)
    plt.close()

    # 4. Velocity Field
    print("Generating Velocity Field...")
    dz = np.diff(z_arr, axis=0)
    z_start = z_arr[:-1]
    stride = 10

    plt.figure(figsize=(10, 8))
    plt.quiver(z_start[::stride,0], z_start[::stride,1], dz[::stride,0], dz[::stride,1],
               angles='xy', scale_units='xy', scale=1, color='blue', alpha=0.6)
    plt.title('Latent Velocity Field (Market Flow)')
    plt.xlabel('Latent Dim 1')
    plt.ylabel('Latent Dim 2')
    plt.grid(True, alpha=0.3)
    plt.savefig('plots/latent_velocity.png', dpi=300)
    plt.close()

    print("Latent space visualizations saved.")

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
    scraper = MarketScraper(tickers=['^GSPC', '^NDX'], start_date='2023-01-01')
    market_data = scraper.process_data()
    simulator = HestonSimulator()

    results = {}

    for ticker in ['^GSPC', '^NDX']:
        print(f"Processing {ticker}...")
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
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    colors = {'^GSPC': 'blue', '^NDX': 'orange'}

    for ticker in ['^GSPC', '^NDX']:
        c = colors[ticker]
        ax1.plot(results[ticker]['dates'], results[ticker]['true'], label=f'{ticker} Truth', color=c)
        ax1.plot(results[ticker]['dates'], results[ticker]['pred'], label=f'{ticker} Pred', color=c, linestyle='--')
        ax2.plot(results[ticker]['dates'], results[ticker]['mae'], label=f'{ticker} MAE', color=c)
        ax2.fill_between(results[ticker]['dates'], results[ticker]['mae'], color=c, alpha=0.3)

    ax1.set_title("Model Generalization: S&P 500 vs Nasdaq 100")
    ax1.set_ylabel("Normalized Price")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.set_title("Mean Absolute Error (MAE)")
    ax2.set_ylabel("MAE Loss")
    ax2.set_xlabel("Date")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("plots/index_comparison.png", dpi=300)
    print("Comparison plot saved.")

if __name__ == "__main__":
    visualize_latent_space()
    compare_indices()
