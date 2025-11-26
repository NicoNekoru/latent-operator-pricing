import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from src.models import NeuralOperator
from src.data_loader import MarketData, MacroData
from src.strategies.benchmark import BenchmarkStrategy
from src.strategies.neural_surfer import NeuralSurferStrategy
from src.strategies.skew import NeuralSkewStrategy
from src.strategies.neural_arbitrage import NeuralArbitrageStrategy
from src.strategies.neural_velocity import NeuralVelocityStrategy
from src.strategies.neural_projection import NeuralProjectionStrategy

def run_backtest_period(model, merged_data, tickers, start_date, end_date, title_suffix, filename):
    device = next(model.parameters()).device
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()

    print(f"\n--- Running Backtest: {title_suffix} ({start_date} to {end_date}) ---")

    for idx, ticker in enumerate(tickers):
        print(f"Processing {ticker}...")
        ax = axes[idx]

        if ticker not in merged_data['Ticker'].values:
            print(f"Warning: No data for {ticker}")
            continue

        df = merged_data[merged_data['Ticker'] == ticker].sort_index()
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

        # Pre-calculate Volume Feature to match training
        # Log(Volume + 1) / 20.0
        vol_feature = np.log(df['Volume'] + 1) / 20.0

        for i in range(start_idx, len(df)):
            window = df.iloc[i-30:i]
            window_vol = vol_feature.iloc[i-30:i]

            # Construct 6-channel input
            # [LogReturn, RealizedVol, VIX, Volume, TNX, Buffett]
            features = np.stack([
                window['LogReturn'].values,
                window['RealizedVol'].values,
                window['VIX'].values,
                window_vol.values,
                window['TNX'].values,
                window['Buffett_Ind'].values
            ], axis=1)

            # Shape: (1, 30, 6)
            x = features.reshape(1, 30, 6)
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
            NeuralSurferStrategy(velocity_threshold_percentile=80),
            NeuralSkewStrategy(),
            NeuralArbitrageStrategy(),
            NeuralProjectionStrategy(aggressive=False), # Naive Pricing
            NeuralVelocityStrategy(aggressive=True)     # Latent Flow (Shorting)
        ]

        results = {}

        for strat in strategies:
            # Pass decoded prices AND model to strategies (VelocityStrategy needs model)
            signals = strat.generate_signals(z_history, prices=price_history, model=model)
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
    # Note: Input dim is now 6
    model = NeuralOperator(input_dim=6, latent_dim=3).to(device)
    try:
        model.load_state_dict(torch.load('models/neural_operator.pth', map_location=device))
    except FileNotFoundError:
        print("Model not found. Please train first.")
        return
    model.eval()

    # 2. Load Data (All Tickers, Full History)
    tickers = ['^GSPC', '^NDX', '^RUT', '^DJI']
    print(f"Fetching Market Data for {tickers}...")

    # Use new Modular Loaders
    market = MarketData(tickers=tickers, start_date='2006-01-01').fetch()
    macro = MacroData(start_date='2006-01-01').fetch()

    # Merge
    merged_data = market.join(macro, how='left').ffill().dropna()

    # 3. Run Backtests
    # Train Period (In-Sample): 2010-01-01 to 2022-12-31
    run_backtest_period(model, merged_data, tickers, '2010-01-01', '2022-12-31', "Train Set (In-Sample)", "strategy_performance_train_enriched.png")

    # Test Set 1: Crisis Period (Out-of-Sample) 2006-01-01 to 2009-12-31
    run_backtest_period(model, merged_data, tickers, '2006-01-01', '2009-12-31', "Test Set 1 (Crisis)", "strategy_performance_crisis_enriched.png")

    # Test Set 2: Recent Period (Out-of-Sample) 2023-01-01 to Present
    run_backtest_period(model, merged_data, tickers, '2023-01-01', '2025-12-31', "Test Set 2 (Recent)", "strategy_performance_test_enriched.png")

if __name__ == "__main__":
    run_backtest()
