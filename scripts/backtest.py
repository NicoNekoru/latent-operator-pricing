import torch
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

from src.models import NeuralOperator
from src.data_loader import MarketScraper
from src.strategies.regime import LatentRegimeStrategy
from src.strategies.momentum import LatentMomentumStrategy
from src.strategies.mean_reversion import LatentMeanReversionStrategy
from src.strategies.benchmark import BenchmarkStrategy

def run_backtest():
    # 1. Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Load Model
    print("Loading Neural Operator...")
    model = NeuralOperator(latent_dim=3).to(device)
    model.load_state_dict(torch.load('models/neural_operator.pth', map_location=device))
    model.eval()

    # 3. Load Data
    print("Fetching Market Data...")
    # Use a longer history to allow for warm-up of rolling windows
    scraper = MarketScraper(tickers=['^GSPC'], start_date='2022-01-01')
    market_data = scraper.process_data()
    df = market_data[market_data['Ticker'] == '^GSPC'].sort_index()

    # 4. Generate Latent Trajectory
    print("Generating Latent Trajectory...")
    z_history = []
    valid_dates = []
    returns_history = []

    # Need 30 days for input
    start_idx = 30

    for i in range(start_idx, len(df)):
        # Prepare Input
        past_30 = df.iloc[i-30:i]
        input_returns = past_30['LogReturn'].values
        input_vols = past_30['RealizedVol'].values

        x = np.stack([input_returns, input_vols], axis=1).reshape(1, 30, 2)
        x_tensor = torch.tensor(x, dtype=torch.float32).to(device)

        with torch.no_grad():
            _, z = model(x_tensor)
            z_np = z.cpu().numpy().flatten()

        z_history.append(z_np)
        valid_dates.append(df.index[i])
        returns_history.append(df.iloc[i]['LogReturn']) # Return for the day we just predicted for?
        # Actually, strategy signal at T uses info up to T.
        # Return realized at T+1.
        # Let's align: Signal[T] * Return[T+1].

    z_history = np.array(z_history)
    returns_series = pd.Series(returns_history, index=valid_dates)

    # Shift returns by -1 to align Signal[T] with Return[T+1]
    # We want to trade TOMORROW based on TODAY's signal.
    future_returns = returns_series.shift(-1).dropna()

    # Align Z data to the same length (drop last point as we don't have future return)
    z_aligned = z_history[:len(future_returns)]
    dates_aligned = valid_dates[:len(future_returns)]

    # 5. Run Strategies
    strategies = [
        BenchmarkStrategy(),
        LatentRegimeStrategy(threshold_percentile=80),
        LatentMomentumStrategy(lookback=5),
        LatentMeanReversionStrategy(window=20, z_score_buy=2.0, z_score_sell=-1.0)
    ]

    results = {}

    print("Running Backtests...")
    for strat in strategies:
        print(f"  Evaluating {strat.name()}...")
        signals = strat.generate_signals(z_aligned)

        # Align signals with returns
        signals = signals.values

        # Calculate Strategy Returns
        # If Signal=1 (Long), Return = Market Return
        # If Signal=0 (Cash), Return = 0 (assuming 0% risk-free for simplicity)
        strat_returns = signals * future_returns.values

        # Cumulative Return
        cum_ret = np.cumsum(strat_returns)

        # Metrics
        total_return = cum_ret[-1]
        daily_std = np.std(strat_returns)
        sharpe = (np.mean(strat_returns) / daily_std) * np.sqrt(252) if daily_std > 0 else 0

        # Max Drawdown
        peak = np.maximum.accumulate(cum_ret)
        drawdown = peak - cum_ret
        max_drawdown = np.max(drawdown)

        results[strat.name()] = {
            'cum_ret': cum_ret,
            'sharpe': sharpe,
            'max_dd': max_drawdown,
            'total_ret': total_return
        }

    # 6. Visualization
    # 6. Visualization - Matplotlib
    print("Generating Performance Plot...")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 8))

    colors = ['gray', 'blue', 'orange', 'green']

    for i, strat in enumerate(strategies):
        name = strat.name()
        res = results[name]

        plt.plot(dates_aligned, res['cum_ret'], label=f"{name} (Sharpe: {res['sharpe']:.2f})", color=colors[i], linewidth=2)

    plt.title("Latent Strategy Performance Comparison (2023-Present)")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Log Return")
    plt.legend(loc='upper left')
    plt.grid(True, alpha=0.3)

    plt.savefig("plots/strategy_comparison.png")
    print("Saved plots/strategy_comparison.png")

    # Print Summary Table
    print("\nPerformance Summary:")
    print(f"{'Strategy':<25} | {'Sharpe':<8} | {'Max DD':<8} | {'Total Ret':<8}")
    print("-" * 60)
    for strat in strategies:
        name = strat.name()
        res = results[name]
        print(f"{name:<25} | {res['sharpe']:.2f}     | {res['max_dd']:.2f}     | {res['total_ret']:.2f}")

if __name__ == "__main__":
    run_backtest()
