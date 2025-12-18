"""
Strategy Backtesting Script for SpectralDeepONet

Runs the new manifold-aware strategies and generates paper-ready outputs.
"""
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.append(PROJECT_ROOT)

from src.models import SpectralDeepONet, get_standard_grid
from src.dataset import OptionDataset
from src.strategies import (
    BuyAndHold,
    ManifoldMomentumStrategy,
    VolatilityRegimeStrategy,
    AdaptiveRiskStrategy,
    VolSurpriseStrategy,
    PredictionConfidenceStrategy,
    LegacySurferStrategy,
    LegacySkewStrategy,
)
from torch.utils.data import DataLoader


def extract_model_outputs(model, dataset, device, max_samples=None):
    """
    Extract latent vectors and predicted surfaces from the model.

    Returns:
        latents: (T, latent_dim) array
        surfaces: (T, 21) array of IV predictions
        returns: (T,) array of log returns
        dates: DatetimeIndex
    """
    loader = DataLoader(dataset, batch_size=64, shuffle=False)
    base_grid = get_standard_grid(device)

    all_latents = []
    all_surfaces = []
    all_returns = []

    model.eval()
    with torch.no_grad():
        for batch_idx, (x, y_price, y_iv, domain) in enumerate(loader):
            x = x.to(device)
            bs = x.size(0)
            grid = base_grid.expand(bs, -1, -1)

            # Forward pass
            price_pred, sigma_pred, latent, _ = model(x, grid, alpha=0.0)

            all_latents.append(latent.cpu().numpy())
            all_surfaces.append(sigma_pred.cpu().numpy())

            # Extract returns from input (LogReturn is channel 0, last timestep)
            returns = x[:, -1, 0].cpu().numpy()
            all_returns.append(returns)

            if max_samples and (batch_idx + 1) * 64 >= max_samples:
                break

    latents = np.concatenate(all_latents, axis=0)
    surfaces = np.concatenate(all_surfaces, axis=0)
    returns = np.concatenate(all_returns, axis=0)

    # Truncate to max_samples if specified
    if max_samples:
        latents = latents[:max_samples]
        surfaces = surfaces[:max_samples]
        returns = returns[:max_samples]

    # Create synthetic dates for now (dataset doesn't expose dates directly)
    dates = pd.date_range(start='2019-01-01', periods=len(latents), freq='B')

    return latents, surfaces, returns, dates


def run_backtest(model, dataset, device, period_name: str):
    """
    Run all strategies on a dataset period.

    Returns:
        results: Dict mapping strategy name to performance metrics
    """
    print(f"\n=== Running Backtest: {period_name} ===")

    # Extract model outputs
    latents, surfaces, returns, dates = extract_model_outputs(model, dataset, device)
    print(f"Data shape: {len(latents)} samples, latent_dim={latents.shape[1]}")

    # Initialize strategies with shorter lookback for small datasets
    lookback = min(60, len(latents) // 3)  # Adaptive based on dataset size

    strategies = [
        BuyAndHold(),
        # New strategies based on surface signals
        VolSurpriseStrategy(z_threshold=1.5, lookback=lookback),
        PredictionConfidenceStrategy(skew_percentile=80, lookback=lookback),
        ManifoldMomentumStrategy(z_threshold=1.5, momentum_window=5, lookback=lookback),
        AdaptiveRiskStrategy(ema_span=20, risk_threshold=1.0, lookback=lookback),
        # Legacy for comparison
        LegacySurferStrategy(percentile=80, lookback=lookback),
    ]

    results = {}

    for strat in strategies:
        print(f"  Running {strat.name}...")
        signals = strat.generate_signals(latents, surfaces, returns, dates)
        metrics = strat.compute_metrics(signals, returns)
        results[strat.name] = {
            'signals': signals,
            'metrics': metrics
        }
        print(f"    Sharpe: {metrics['sharpe']:.3f}, Cum Return: {metrics['cum_return']:.2%}")

    return results, dates, returns


def plot_results(results, dates, returns, period_name: str, save_path: str):
    """Generate comparison plot."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # Top: Cumulative returns
    ax1 = axes[0]

    # Benchmark: Buy & Hold
    bh_cum = np.cumprod(1 + returns) - 1
    ax1.plot(dates, bh_cum, label='Buy & Hold', color='gray', linewidth=2, alpha=0.7)

    colors = sns.color_palette("husl", len(results) - 1)
    color_idx = 0

    for name, data in results.items():
        if name == 'Buy & Hold':
            continue
        metrics = data['metrics']
        strat_cum = np.cumprod(1 + metrics['strategy_returns']) - 1
        ax1.plot(dates, strat_cum,
                label=f"{name} (SR: {metrics['sharpe']:.2f})",
                color=colors[color_idx], linewidth=1.5)
        color_idx += 1

    ax1.set_title(f'Strategy Performance: {period_name}', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Cumulative Return')
    ax1.legend(loc='upper left', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

    # Bottom: Signals over time (for one representative strategy)
    ax2 = axes[1]

    # Show Vol Surprise signals (or first non-B&H strategy)
    signal_strat = None
    for name in ['Vol Surprise', 'Pred Confidence', 'Manifold Momentum']:
        if name in results:
            signal_strat = name
            break

    if signal_strat:
        signals = results[signal_strat]['signals']
        ax2.fill_between(dates, 0, signals, alpha=0.5, label=f'{signal_strat} Signal')
        ax2.set_ylim(-0.1, 1.1)
        ax2.set_ylabel('Position (0=Cash, 1=Long)')
        ax2.set_xlabel('Date')
        ax2.set_title(f'{signal_strat} Strategy Signals', fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved to {save_path}")
    plt.close()


def generate_summary_table(all_results: dict) -> pd.DataFrame:
    """Generate summary table for paper."""
    rows = []

    for period, results in all_results.items():
        for name, data in results.items():
            m = data['metrics']
            rows.append({
                'Period': period,
                'Strategy': name,
                'Sharpe': m['sharpe'],
                'Cum Return': m['cum_return'],
                'Max Drawdown': m['max_drawdown'],
                'Volatility': m['volatility'],
                'Win Rate': m['win_rate']
            })

    df = pd.DataFrame(rows)
    return df


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load model
    model = SpectralDeepONet(input_channels=6, latent_dim=64).to(device)
    model_path = os.path.join(PROJECT_ROOT, 'training/models/deeponet.pth')

    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("Model loaded successfully.")
    except FileNotFoundError:
        print(f"Model not found at {model_path}. Please train first.")
        return

    # Load datasets
    dataset_path = os.path.join(PROJECT_ROOT, 'data/processed_dataset.parquet')

    train_dataset = OptionDataset(dataset_path, mode='train')
    val_dataset = OptionDataset(dataset_path, mode='val')

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")

    # Run backtests
    all_results = {}

    # Training period (In-Sample)
    train_results, train_dates, train_returns = run_backtest(
        model, train_dataset, device, "Train (2019-2021)"
    )
    all_results['Train'] = train_results

    # Validation period (Out-of-Sample, 2022 Crisis)
    val_results, val_dates, val_returns = run_backtest(
        model, val_dataset, device, "Validation (2022 Crisis)"
    )
    all_results['Val'] = val_results

    # Generate plots
    plot_dir = os.path.join(PROJECT_ROOT, 'analysis/plots')
    os.makedirs(plot_dir, exist_ok=True)

    plot_results(train_results, train_dates, train_returns,
                 "Train (2019-2021)", os.path.join(plot_dir, 'strategy_train.png'))
    plot_results(val_results, val_dates, val_returns,
                 "Validation (2022 Crisis)", os.path.join(plot_dir, 'strategy_val.png'))

    # Generate summary table
    summary = generate_summary_table(all_results)
    print("\n=== Summary Table ===")
    print(summary.to_string(index=False))

    # Save table
    summary.to_csv(os.path.join(plot_dir, 'strategy_summary.csv'), index=False)
    print(f"\nSaved summary to {plot_dir}/strategy_summary.csv")


if __name__ == "__main__":
    main()
