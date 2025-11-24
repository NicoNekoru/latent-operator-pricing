
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

    # Plotting
    print("Generating Comparison Plot...")
    import plotly.subplots as sp

    fig = sp.make_subplots(rows=2, cols=1, shared_xaxes=True,
                           vertical_spacing=0.1,
                           subplot_titles=("ATM Call Price Comparison (Model vs Truth)", "Mean Absolute Error (MAE) over Time"))

    colors = {'^GSPC': 'blue', '^NDX': 'orange'}

    for ticker in ['^GSPC', '^NDX']:
        color = colors[ticker]

        # Price Plot (Top)
        fig.add_trace(go.Scatter(
            x=results[ticker]['dates'], y=results[ticker]['true'],
            mode='lines', name=f'{ticker} Truth',
            line=dict(color=color, width=2)
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=results[ticker]['dates'], y=results[ticker]['pred'],
            mode='lines', name=f'{ticker} Pred',
            line=dict(color=color, width=2, dash='dash')
        ), row=1, col=1)

        # MAE Plot (Bottom)
        fig.add_trace(go.Scatter(
            x=results[ticker]['dates'], y=results[ticker]['mae'],
            mode='lines', name=f'{ticker} MAE',
            line=dict(color=color, width=1.5),
            fill='tozeroy', # Fill area under curve
            opacity=0.3
        ), row=2, col=1)

    fig.update_layout(height=800, title_text="Model Generalization: S&P 500 vs Nasdaq 100")
    fig.update_yaxes(title_text="Normalized Price", row=1, col=1)
    fig.update_yaxes(title_text="MAE Loss", row=2, col=1)

    fig.write_html("plots/index_comparison.html")
    print("Comparison plot saved to plots/index_comparison.html")

if __name__ == "__main__":
    # visualize_latent_space()
    compare_indices()
