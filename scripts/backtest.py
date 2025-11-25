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
from src.strategies.skew import SkewStrategy

def run_backtest_period(model, market_data, tickers, start_date, end_date, title_suffix, filename):
    device = next(model.parameters()).device
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()

    print(f"\n--- Running Backtest: {title_suffix} ({start_date} to {end_date}) ---")

    for idx, ticker in enumerate(tickers):
        print(f"Processing {ticker}...")
        ax = axes[idx]

        if ticker not in market_data['Ticker'].values:
            print(f"Warning: No data for {ticker}")
            continue

        df = market_data[market_data['Ticker'] == ticker].sort_index()
        # Filter by date range
        mask = (df.index >= start_date) & (df.index < end_date)
        df = df[mask]

        if len(df) < 60:
            print(f"Insufficient data for {ticker} in this period.")
            continue

        # 3. Generate Latent Trajectory & Decode Prices
        z_history = []
        price_history = []
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
                # Get both Z and the decoded Prices
                y_pred, z = model(x_tensor)
                z_np = z.cpu().numpy().flatten()
                prices_np = y_pred.cpu().numpy().flatten()

            z_history.append(z_np)
            price_history.append(prices_np)
            dates.append(df.index[i])
            returns.append(df.iloc[i]['LogReturn'])

        z_history = np.array(z_history)
        price_history = np.array(price_history)
        returns = np.array(returns)

        # 4. Run Strategies
        strategies = [
            BenchmarkStrategy(ticker=ticker),
            RegimeStrategy(threshold_percentile=80),
            MomentumStrategy(lookback=5),
            MeanReversionStrategy(z_score_threshold=2.0),
            SkewStrategy(skew_threshold=0.05)
        ]

        results = {}

        for strat in strategies:
            # Pass decoded prices to strategies (SkewStrategy needs it)
            signals = strat.generate_signals(z_history, prices=price_history)
            strat_returns = signals.shift(1).fillna(0).values * returns
            cum_ret = np.cumprod(1 + strat_returns) - 1

            total_ret = cum_ret[-1]
            sharpe = np.mean(strat_returns) / (np.std(strat_returns) + 1e-9) * np.sqrt(252)

            results[strat.name] = {'cum_ret': cum_ret, 'sharpe': sharpe}

        # 5. Plot on Subplot
        for name, res in results.items():
            # Ensure Benchmark is visible (higher zorder or distinct style if needed)
            if "Buy & Hold" in name:
                ax.plot(dates, res['cum_ret'], label=f"{name} (SR: {res['sharpe']:.2f})", linewidth=2, linestyle='--', color='black', alpha=0.7)
            else:
                ax.plot(dates, res['cum_ret'], label=f"{name} (SR: {res['sharpe']:.2f})")

        ax.set_title(f"{ticker} - {title_suffix}")
        ax.set_xlabel("Date")
        ax.set_ylabel("Cumulative Return")
        ax.legend(fontsize='small')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs('plots', exist_ok=True)
    plt.savefig(f"plots/{filename}", dpi=300)
    print(f"Saved plots/{filename}")

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

    # 2. Load Data (All Tickers, Full History)
    tickers = ['^GSPC', '^NDX', '^RUT', '^DJI']
    print(f"Fetching Market Data for {tickers}...")
    # Load from 2006 to get full training history including 2008 Crisis
    scraper = MarketScraper(tickers=tickers, start_date='2006-01-01')
    market_data = scraper.process_data()

    # 3. Run Backtests
    # Train Period (In-Sample): 2006-01-01 to 2023-01-01
    run_backtest_period(model, market_data, tickers, '2006-01-01', '2023-01-01', "Train Set (In-Sample)", "strategy_performance_train.png")

    # Test Period (Out-of-Sample): 2023-01-01 to Present
    run_backtest_period(model, market_data, tickers, '2023-01-01', '2025-12-31', "Test Set (Out-of-Sample)", "strategy_performance_test.png")

if __name__ == "__main__":
    run_backtest()
