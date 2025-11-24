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

    # 2. Load Data (2023-Present)
    print("Fetching Market Data...")
    scraper = MarketScraper(tickers=['^GSPC'], start_date='2023-01-01')
    market_data = scraper.process_data()
    df = market_data[market_data['Ticker'] == '^GSPC'].sort_index()

    # 3. Generate Latent Trajectory
    print("Generating Latent Trajectory...")
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

    print("Running Strategies...")
    for strat in strategies:
        signals = strat.generate_signals(z_history)
        strat_returns = signals.shift(1).fillna(0).values * returns
        cum_ret = np.cumprod(1 + strat_returns) - 1

        total_ret = cum_ret[-1]
        sharpe = np.mean(strat_returns) / (np.std(strat_returns) + 1e-9) * np.sqrt(252)

        cum_ret_series = pd.Series(cum_ret)
        drawdown = cum_ret_series - cum_ret_series.cummax()
        max_dd = drawdown.min()

        results[strat.name] = {
            'cum_ret': cum_ret,
            'total_ret': total_ret,
            'sharpe': sharpe,
            'max_dd': max_dd
        }
        print(f"{strat.name}: Return={total_ret*100:.2f}%, Sharpe={sharpe:.2f}, MaxDD={max_dd*100:.2f}%")

    # 5. Visualization - Matplotlib
    print("Generating Strategy Comparison Plot...")
    plt.figure(figsize=(12, 8))

    for name, res in results.items():
        plt.plot(dates, res['cum_ret'], label=f"{name} (SR: {res['sharpe']:.2f})")

    plt.title("Strategy Performance Benchmark (2023-Present)")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Return")
    plt.legend()
    plt.grid(True, alpha=0.3)

    os.makedirs('plots', exist_ok=True)
    plt.savefig("plots/strategy_comparison.png", dpi=300)
    print("Saved plots/strategy_comparison.png")

if __name__ == "__main__":
    run_backtest()
