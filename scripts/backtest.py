import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models import NeuralOperator
from src.data_loader import MarketScraper
from src.strategies.benchmark import BenchmarkStrategy
from src.strategies.regime import RegimeStrategy
from src.strategies.momentum import MomentumStrategy
from src.strategies.mean_reversion import MeanReversionStrategy

def run_backtest():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load Model
    model = NeuralOperator(latent_dim=3).to(device)
    try:
        model.load_state_dict(torch.load('models/neural_operator.pth', map_location=device))
    except FileNotFoundError:
        print("Model not found. Please train first.")
        return
    model.eval()

    # 2. Load Data (All Tickers)
    tickers = ['^GSPC', '^NDX', '^RUT', '^DJI']
    print(f"Fetching Market Data for {tickers}...")
    scraper = MarketScraper(tickers=tickers, start_date='2023-01-01')
    market_data = scraper.process_data()

    # Prepare Plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()

    for idx, ticker in enumerate(tickers):
        print(f"Backtesting {ticker}...")
        ax = axes[idx]

        if ticker not in market_data['Ticker'].values:
            print(f"Warning: No data for {ticker}")
            continue

        df = market_data[market_data['Ticker'] == ticker].sort_index()
        df = df[df.index >= '2023-01-01']

        # 3. Generate Latent Trajectory
        z_history = []
        dates = []
        returns = []

        start_idx = 30

        for i in range(start_idx, len(df)):
            past_30 = df.iloc[i-30:i]
            input_returns = past_30['LogReturn'].values
            input_vols = past_30['RealizedVol'].values

            x = np.stack([input_returns, input_vols], axis=1).reshape(1, 30, 2)
            x_tensor = torch.tensor(x, dtype=torch.float32).to(device)

            with torch.no_grad():
                _, z = model(x_tensor)
                z_np = z.cpu().numpy().flatten()

            z_history.append(z_np)
            dates.append(df.index[i])
            returns.append(df.iloc[i]['LogReturn'])

        z_history = np.array(z_history)
        returns = np.array(returns)

        # 4. Run Strategies
        strategies = [
            BenchmarkStrategy(),
            RegimeStrategy(threshold_percentile=80),
            MomentumStrategy(lookback=5),
            MeanReversionStrategy(z_score_threshold=2.0)
        ]

        results = {}

        for strat in strategies:
            signals = strat.generate_signals(z_history)
            strat_returns = signals.shift(1).fillna(0).values * returns
            cum_ret = np.cumprod(1 + strat_returns) - 1

            total_ret = cum_ret[-1]
            sharpe = np.mean(strat_returns) / (np.std(strat_returns) + 1e-9) * np.sqrt(252)

            results[strat.name] = {'cum_ret': cum_ret, 'sharpe': sharpe}

        # 5. Plot on Subplot
        for name, res in results.items():
            ax.plot(dates, res['cum_ret'], label=f"{name} (SR: {res['sharpe']:.2f})")

        ax.set_title(f"{ticker} Strategy Performance")
        ax.set_xlabel("Date")
        ax.set_ylabel("Cumulative Return")
        ax.legend(fontsize='small')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs('plots', exist_ok=True)
    plt.savefig("plots/multi_index_strategy_comparison.png", dpi=300)
    print("Saved plots/multi_index_strategy_comparison.png")

if __name__ == "__main__":
    run_backtest()
