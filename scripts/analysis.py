
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from torch.utils.data import DataLoader
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models import NeuralOperator
from src.train import OptionDataset

def visualize_latent_space():
    """
    Generates 3D Latent Space visualizations.
    """
    from src.data_loader import MarketScraper

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NeuralOperator(latent_dim=3).to(device)
    model.load_state_dict(torch.load('models/neural_operator.pth', map_location=device))
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

    # 1. Latent Space 3D (Plotly - Interactive)
    fig = px.scatter_3d(
        x=z_arr[:,0], y=z_arr[:,1], z=z_arr[:,2],
        color=vol_list,
        color_continuous_scale='RdBu_r',
        title="Latent Space Topology (Colored by Volatility)",
        labels={'color': 'Realized Vol'}
    )
    fig.write_html("plots/latent_space_3d.html")
    # Try saving PNG, but don't fail if kaleido is broken
    try:
        fig.write_image("plots/latent_space_3d.png", scale=2)
    except: pass

    # 2. Trajectory (2023) (Plotly - Interactive)
    mask_2023 = [str(d).startswith('2023') for d in dates]
    z_2023 = z_arr[mask_2023]

    if len(z_2023) > 0:
        fig2 = px.line_3d(
            x=z_2023[:,0], y=z_2023[:,1], z=z_2023[:,2],
            title="Market Trajectory (2023)",
        )
        fig2.write_html("plots/trajectory_2023.html")
        try:
            fig2.write_image("plots/trajectory_2023.png", scale=2)
        except: pass

    # 3. Latent Landscape (Heatmap) - Matplotlib
    # Grid search over Z1, Z2 (Z3 fixed at mean)
    grid_size = 50
    z1 = np.linspace(z_arr[:,0].min(), z_arr[:,0].max(), grid_size)
    z2 = np.linspace(z_arr[:,1].min(), z_arr[:,1].max(), grid_size)
    z3_mean = z_arr[:,2].mean()

    Z1, Z2 = np.meshgrid(z1, z2)
    prices = np.zeros((grid_size, grid_size))

    print("Generating Landscape...")
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
    plt.savefig('plots/latent_landscape.png')
    plt.close()

    # Save interactive version too for consistency
    fig3 = go.Figure(data=[go.Surface(z=prices, x=z1, y=z2)])
    fig3.update_layout(title="Latent Physics Landscape (ATM Price Surface)", scene=dict(xaxis_title='Z1', yaxis_title='Z2', zaxis_title='Price'))
    fig3.write_html("plots/latent_landscape.html")


    # 4. Velocity Field - Matplotlib
    # Calculate delta Z
    dz = np.diff(z_arr, axis=0)
    z_start = z_arr[:-1]

    # Downsample for quiver plot
    stride = 10

    plt.figure(figsize=(10, 8))
    plt.quiver(z_start[::stride,0], z_start[::stride,1], dz[::stride,0], dz[::stride,1],
               angles='xy', scale_units='xy', scale=1, color='blue', alpha=0.6)
    plt.title('Latent Velocity Field (Market Flow)')
    plt.xlabel('Latent Dim 1')
    plt.ylabel('Latent Dim 2')
    plt.grid(True, alpha=0.3)
    plt.savefig('plots/latent_velocity.png')
    plt.close()

    # Save interactive version too
    fig4 = go.Figure(data=go.Cone(
        x=z_start[::stride,0], y=z_start[::stride,1], z=z_start[::stride,2],
        u=dz[::stride,0], v=dz[::stride,1], w=dz[::stride,2],
        sizemode="absolute", sizeref=2, anchor="tail"
    ))
    fig4.update_layout(title="Latent Velocity Field (Market Flow)")
    fig4.write_html("plots/latent_velocity.html")

    print("All visualizations saved to plots/")

def compare_indices():
    """
    Generates a comparison graph for GSPC (Training Index) vs NDX (Test Index).
    Plots:
    1. Time Series of ATM Call Prices (Model vs Ground Truth).
    2. MAE Loss over time.
    """
    from src.data_loader import MarketScraper, HestonSimulator

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load Model
    model = NeuralOperator(latent_dim=3).to(device)
    model.load_state_dict(torch.load('models/neural_operator.pth', map_location=device))
    model.eval()

    # Fetch Data for both indices
    print("Fetching Market Data for GSPC and NDX...")
    scraper = MarketScraper(tickers=['^GSPC', '^NDX'], start_date='2023-01-01') # Focus on 2023+
    market_data = scraper.process_data()

    simulator = HestonSimulator()

    results = {}

    for ticker in ['^GSPC', '^NDX']:
        print(f"Processing {ticker}...")
        df = market_data[market_data['Ticker'] == ticker].sort_index()

        dates = []
        true_prices = []
        pred_prices = []
        maes = []

        # Process last 200 days
        # Need to ensure we have 30 days history for the first day
        start_idx = 30
        end_idx = min(len(df), 230) # 200 days

        for i in range(start_idx, end_idx):
            row = df.iloc[i]
            date_val = row.name
            spot = row['Close']
            realized_vol = row['RealizedVol']

            # 1. Generate Ground Truth
            v0 = realized_vol ** 2
            kappa = 2.0
            theta = v0
            sigma = 0.3
            rho = -0.7

            surface = simulator.generate_surface(spot, v0, kappa, theta, sigma, rho)
            # Extract ATM Price (Maturity 1m, Moneyness 1.0 -> Index 3 in our list)
            # Surface list order: M1_K1...K7, M2...
            # M1 is first 7 items. K=1.0 is 4th item (index 3).
            atm_true = surface[3]['Price']

            # Flatten for full surface comparison
            flat_true = np.array([p['Price'] for p in surface])

            # 2. Prepare Input for Model
            past_30 = df.iloc[i-30:i]
            input_returns = past_30['LogReturn'].values
            input_vols = past_30['RealizedVol'].values

            # Shape: (1, 30, 2)
            x = np.stack([input_returns, input_vols], axis=1).reshape(1, 30, 2)
            x_tensor = torch.tensor(x, dtype=torch.float32).to(device)

            # 3. Predict
            with torch.no_grad():
                y_pred, _ = model(x_tensor)
                y_pred = y_pred.cpu().numpy().flatten()

            atm_pred = y_pred[3]

            # 4. Metrics
            mae = np.mean(np.abs(y_pred - flat_true))

            dates.append(date_val)
            true_prices.append(atm_true)
            pred_prices.append(atm_pred)
            maes.append(mae)

        results[ticker] = {
            'dates': dates,
            'true': true_prices,
            'pred': pred_prices,
            'mae': maes
        }

    # Plotting - Matplotlib
    print("Generating Comparison Plot...")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    colors = {'^GSPC': 'blue', '^NDX': 'orange'}

    for ticker in ['^GSPC', '^NDX']:
        color = colors[ticker]

        # Price Plot (Top)
        ax1.plot(results[ticker]['dates'], results[ticker]['true'], label=f'{ticker} Truth', color=color)
        ax1.plot(results[ticker]['dates'], results[ticker]['pred'], label=f'{ticker} Pred', color=color, linestyle='--')

        # MAE Plot (Bottom)
        ax2.plot(results[ticker]['dates'], results[ticker]['mae'], label=f'{ticker} MAE', color=color)
        ax2.fill_between(results[ticker]['dates'], results[ticker]['mae'], color=color, alpha=0.3)

    ax1.set_title("Model Generalization: S&P 500 vs Nasdaq 100")
    ax1.set_ylabel("Normalized Price")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.set_title("Mean Absolute Error (MAE) over Time")
    ax2.set_ylabel("MAE Loss")
    ax2.set_xlabel("Date")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("plots/index_comparison.png")
    print("Comparison plot saved to plots/index_comparison.png")

    # Save interactive version too
    import plotly.subplots as sp
    fig = sp.make_subplots(rows=2, cols=1, shared_xaxes=True,
                           vertical_spacing=0.1,
                           subplot_titles=("ATM Call Price Comparison (Model vs Truth)", "Mean Absolute Error (MAE) over Time"))

    for ticker in ['^GSPC', '^NDX']:
        color = colors[ticker]
        fig.add_trace(go.Scatter(x=results[ticker]['dates'], y=results[ticker]['true'], mode='lines', name=f'{ticker} Truth', line=dict(color=color, width=2)), row=1, col=1)
        fig.add_trace(go.Scatter(x=results[ticker]['dates'], y=results[ticker]['pred'], mode='lines', name=f'{ticker} Pred', line=dict(color=color, width=2, dash='dash')), row=1, col=1)
        fig.add_trace(go.Scatter(x=results[ticker]['dates'], y=results[ticker]['mae'], mode='lines', name=f'{ticker} MAE', line=dict(color=color, width=1.5), fill='tozeroy', opacity=0.3), row=2, col=1)

    fig.update_layout(height=800, title_text="Model Generalization: S&P 500 vs Nasdaq 100")
    fig.write_html("plots/index_comparison.html")

if __name__ == "__main__":
    visualize_latent_space()
    compare_indices()
