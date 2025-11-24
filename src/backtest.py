import torch
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from src.models import NeuralOperator
from src.data_loader import MarketScraper
from src.strategies.benchmark import BenchmarkStrategy
from src.strategies.regime import RegimeStrategy
from src.strategies.momentum import MomentumStrategy
from src.strategies.mean_reversion import MeanReversionStrategy
import os

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

    # Need 30 days history for first inference
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
        returns.append(df.iloc[i]['LogReturn']) # Return for the *current* day (t)

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

        # Align signals with returns
        # Signal calculated at t-1 applies to return at t
        # z_history[i] is z_t. We use z_0...z_{t-1} to predict for t?
        # Wait, z_history[i] comes from data[i-30:i]. This is available at time i (before close? or after?)
        # Let's assume we trade at Close of day i based on Z_i.
        # So Signal_i * Return_{i+1}.

        # Shift signals by 1 to avoid lookahead
        # signal[i] is based on z[i]. It trades for return[i+1].

        strat_returns = signals.shift(1).fillna(0).values * returns

        # Cumulative Return
        cum_ret = np.cumprod(1 + strat_returns) - 1

        # Metrics
        total_ret = cum_ret[-1]
        sharpe = np.mean(strat_returns) / (np.std(strat_returns) + 1e-9) * np.sqrt(252)

        # Max Drawdown
        cum_ret_series = pd.Series(cum_ret)
        roll_max = cum_ret_series.cummax()
        drawdown = cum_ret_series - roll_max
        max_dd = drawdown.min()

        results[strat.name] = {
            'cum_ret': cum_ret,
            'total_ret': total_ret,
            'sharpe': sharpe,
            'max_dd': max_dd
        }
        print(f"{strat.name}: Return={total_ret*100:.2f}%, Sharpe={sharpe:.2f}, MaxDD={max_dd*100:.2f}%")

    # 5. Visualization
    print("Generating Strategy Comparison Plot...")
    fig = go.Figure()

    for name, res in results.items():
        fig.add_trace(go.Scatter(
            x=dates, y=res['cum_ret'],
            mode='lines', name=f"{name} (SR: {res['sharpe']:.2f})"
        ))

    fig.update_layout(
        title="Strategy Performance Benchmark (2023-Present)",
        xaxis_title="Date",
        yaxis_title="Cumulative Return",
        template="plotly_dark"
    )

    os.makedirs('plots', exist_ok=True)
    fig.write_html("plots/strategy_comparison.html")
    print("Saved plots/strategy_comparison.html")

if __name__ == "__main__":
    run_backtest()
