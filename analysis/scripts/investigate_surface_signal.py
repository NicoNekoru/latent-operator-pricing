"""
Deep investigation of predicted IV surface for trading signals.

The question: Does the model's predicted surface contain information
that could be used for trading, even if the latent space doesn't
strongly separate regimes?
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.append(PROJECT_ROOT)

from src.models import SpectralDeepONet, get_standard_grid
from src.dataset import OptionDataset
from torch.utils.data import DataLoader


def extract_all_outputs(model, dataset, device):
    """Extract model predictions and compare to actuals."""
    loader = DataLoader(dataset, batch_size=64, shuffle=False)
    base_grid = get_standard_grid(device)

    all_latents = []
    all_pred_surfaces = []
    all_actual_surfaces = []
    all_returns = []

    model.eval()
    with torch.no_grad():
        for x, y_price, y_iv, domain in loader:
            x = x.to(device)
            y_iv = y_iv.to(device)
            bs = x.size(0)
            grid = base_grid.expand(bs, -1, -1)

            price_pred, sigma_pred, latent, _ = model(x, grid, alpha=0.0)

            all_latents.append(latent.cpu().numpy())
            all_pred_surfaces.append(sigma_pred.cpu().numpy())
            all_actual_surfaces.append(y_iv.cpu().numpy())

            # Return is channel 0, last timestep
            returns = x[:, -1, 0].cpu().numpy()
            all_returns.append(returns)

    return {
        'latents': np.concatenate(all_latents),
        'pred_surfaces': np.concatenate(all_pred_surfaces),
        'actual_surfaces': np.concatenate(all_actual_surfaces),
        'returns': np.concatenate(all_returns)
    }


def analyze_prediction_error(train_data, val_data):
    """
    Key hypothesis: Prediction errors might signal regime mismatch.
    If the model was trained mostly on calm regimes, it should make
    larger errors during crisis; this error itself could be a signal.
    """
    print("=== Prediction Error Analysis ===\n")

    # Compute MAPE for each sample, excluding floor values (0.01)
    def compute_mape(pred, actual):
        """Compute MAPE excluding floor values."""
        mapes = []
        for i in range(len(pred)):
            valid_mask = actual[i] > 0.02  # Exclude 0.01 floors
            if valid_mask.sum() > 0:
                mape = np.mean(np.abs(pred[i, valid_mask] - actual[i, valid_mask]) /
                               actual[i, valid_mask])
                mapes.append(mape)
            else:
                mapes.append(np.nan)
        return np.array(mapes)

    train_mape = compute_mape(train_data['pred_surfaces'], train_data['actual_surfaces'])
    val_mape = compute_mape(val_data['pred_surfaces'], val_data['actual_surfaces'])

    # Remove NaN
    train_mape = train_mape[~np.isnan(train_mape)]
    val_mape = val_mape[~np.isnan(val_mape)]

    print(f"Train MAPE: mean={train_mape.mean()*100:.2f}%, std={train_mape.std()*100:.2f}%")
    print(f"Val MAPE:   mean={val_mape.mean()*100:.2f}%, std={val_mape.std()*100:.2f}%")
    print(f"\nVal MAPE percentiles: 50th={np.percentile(val_mape, 50)*100:.2f}%, "
          f"75th={np.percentile(val_mape, 75)*100:.2f}%, 95th={np.percentile(val_mape, 95)*100:.2f}%")

    return train_mape, val_mape


def analyze_surface_features(train_data, val_data):
    """
    Look at surface-level features that might differ between regimes:
    1. ATM vol level
    2. Skew (OTM put vs OTM call)
    3. Term structure (1M vs 6M)
    4. Smile curvature
    """
    print("\n=== Surface Feature Analysis ===\n")

    # Grid: 7 strikes x 3 maturities = 21 points
    # Strikes: [0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2]
    # Maturities: [1M, 3M, 6M] (indices 0-6, 7-13, 14-20)

    features = {}

    for name, data in [('Train', train_data), ('Val', val_data)]:
        pred = data['pred_surfaces']
        actual = data['actual_surfaces']

        # ATM vol (1M, strike=1.0, index 3)
        atm_pred = pred[:, 3]
        atm_actual = actual[:, 3]

        # Skew (1M): OTM Put (0.8) - OTM Call (1.2)
        skew_pred = pred[:, 0] - pred[:, 6]
        skew_actual = actual[:, 0] - actual[:, 6]

        # Term structure: 6M ATM - 1M ATM
        term_pred = pred[:, 17] - pred[:, 3]  # 6M ATM is index 14+3=17
        term_actual = actual[:, 17] - actual[:, 3]

        # Curvature (1M): second derivative at ATM
        curv_pred = pred[:, 4] - 2*pred[:, 3] + pred[:, 2]
        curv_actual = actual[:, 4] - 2*actual[:, 3] + actual[:, 2]

        features[name] = {
            'atm_pred': atm_pred, 'atm_actual': atm_actual,
            'skew_pred': skew_pred, 'skew_actual': skew_actual,
            'term_pred': term_pred, 'term_actual': term_actual,
            'curv_pred': curv_pred, 'curv_actual': curv_actual,
        }

        print(f"{name}:")
        print(f"  ATM Vol - Pred: {atm_pred.mean():.4f}+/-{atm_pred.std():.4f}, "
              f"Actual: {atm_actual.mean():.4f}+/-{atm_actual.std():.4f}")
        print(f"  Skew - Pred: {skew_pred.mean():.4f}+/-{skew_pred.std():.4f}, "
              f"Actual: {skew_actual.mean():.4f}+/-{skew_actual.std():.4f}")
        print(f"  Term Struct - Pred: {term_pred.mean():.4f}+/-{term_pred.std():.4f}, "
              f"Actual: {term_actual.mean():.4f}+/-{term_actual.std():.4f}")
        print()

    return features


def analyze_error_as_signal(data, returns, mape, threshold_percentile=75):
    """
    Test: Can prediction error be used as a crisis signal?
    High error = model struggling = unusual regime = reduce exposure
    """
    print("\n=== Error-Based Trading Signal ===\n")

    # Simple strategy: go to cash when error is above threshold
    threshold = np.percentile(mape, threshold_percentile)
    print(f"Error threshold (75th percentile): {threshold*100:.2f}%")

    # Generate signals
    signals = np.where(mape > threshold, 0.0, 1.0)

    # Lag signals and compute returns
    strategy_returns = np.zeros_like(returns)
    strategy_returns[1:] = signals[:-1] * returns[1:]

    if np.std(strategy_returns) > 1e-9:
        sharpe = np.mean(strategy_returns) / np.std(strategy_returns) * np.sqrt(252)
    else:
        sharpe = 0.0

    cum_ret = np.prod(1 + strategy_returns) - 1
    bh_cum = np.prod(1 + returns) - 1

    print(f"Strategy Sharpe: {sharpe:.3f}")
    print(f"Strategy Cumulative Return: {cum_ret:.2%}")
    print(f"Buy & Hold Cumulative Return: {bh_cum:.2%}")
    print(f"Cash ratio: {1 - signals.mean():.1%}")

    return signals, strategy_returns


def analyze_surprise_signal(data, returns):
    """
    Test: Use the difference between predicted and actual as surprise signal.
    If model predicts low vol but actual is high = crisis surprise = reduce exposure
    """
    print("\n=== Surprise-Based Trading Signal ===\n")

    # ATM vol surprise (actual - predicted)
    pred_atm = data['pred_surfaces'][:, 3]
    actual_atm = data['actual_surfaces'][:, 3]
    surprise = actual_atm - pred_atm

    print(f"Surprise (actual - pred): mean={surprise.mean():.4f}, std={surprise.std():.4f}")
    print(f"  Positive surprise (actual > pred): {(surprise > 0).mean():.1%}")

    # Rolling z-score of surprise
    window = 20
    signals = np.ones(len(surprise))

    for i in range(window, len(surprise)):
        history = surprise[max(0, i-window):i]
        mean_s = np.mean(history)
        std_s = np.std(history) + 1e-6
        z = (surprise[i] - mean_s) / std_s

        # High positive surprise (actual >> pred) = reduce exposure
        if z > 1.5:
            signals[i] = 0.5
        elif z > 2.0:
            signals[i] = 0.0

    # Compute strategy performance
    strategy_returns = np.zeros_like(returns)
    strategy_returns[1:] = signals[:-1] * returns[1:]

    if np.std(strategy_returns) > 1e-9:
        sharpe = np.mean(strategy_returns) / np.std(strategy_returns) * np.sqrt(252)
    else:
        sharpe = 0.0

    cum_ret = np.prod(1 + strategy_returns) - 1

    print(f"Surprise Strategy Sharpe: {sharpe:.3f}")
    print(f"Surprise Strategy Cum Return: {cum_ret:.2%}")

    return signals, surprise


def main():
    device = torch.device('cpu')

    # Load model
    model = SpectralDeepONet(input_channels=6, latent_dim=64).to(device)
    model_path = os.path.join(PROJECT_ROOT, 'training/models/deeponet.pth')
    model.load_state_dict(torch.load(model_path, map_location=device))

    # Load data
    dataset_path = os.path.join(PROJECT_ROOT, 'data/processed_dataset.parquet')
    train_dataset = OptionDataset(dataset_path, mode='train')
    val_dataset = OptionDataset(dataset_path, mode='val')

    print("Extracting model outputs...")
    train_data = extract_all_outputs(model, train_dataset, device)
    val_data = extract_all_outputs(model, val_dataset, device)

    # 1. Prediction Error Analysis
    train_mape, val_mape = analyze_prediction_error(train_data, val_data)

    # 2. Surface Features
    features = analyze_surface_features(train_data, val_data)

    # 3. Error-Based Trading (on Val)
    print("\n" + "="*50)
    print("VALIDATION SET (2022 Crisis)")
    print("="*50)
    error_signals, error_returns = analyze_error_as_signal(
        val_data, val_data['returns'], val_mape
    )

    # 4. Surprise-Based Trading (on Val)
    surprise_signals, surprise = analyze_surprise_signal(val_data, val_data['returns'])

    # Plot
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # MAPE distribution (multiply by 100 for display)
    ax = axes[0, 0]
    ax.hist(train_mape * 100, bins=50, alpha=0.7, label='Train', density=True)
    ax.hist(val_mape * 100, bins=50, alpha=0.7, label='Val', density=True)
    ax.set_xlabel('MAPE (%)')
    ax.set_title('Prediction Error Distribution')
    ax.legend()

    # ATM vol comparison
    ax = axes[0, 1]
    ax.scatter(features['Val']['atm_actual'], features['Val']['atm_pred'],
               alpha=0.5, s=10, label='Val')
    ax.plot([0.1, 0.5], [0.1, 0.5], 'r--', label='Perfect')
    ax.set_xlabel('Actual ATM Vol')
    ax.set_ylabel('Predicted ATM Vol')
    ax.set_title('ATM Vol: Predicted vs Actual (Val)')
    ax.legend()

    # Skew over time
    ax = axes[0, 2]
    ax.plot(features['Val']['skew_actual'], label='Actual', alpha=0.7)
    ax.plot(features['Val']['skew_pred'], label='Predicted', alpha=0.7)
    ax.set_xlabel('Sample')
    ax.set_ylabel('Skew (OTM Put - OTM Call)')
    ax.set_title('Skew Over Time (Val)')
    ax.legend()

    # Error over time with signals (multiply by 100 for display)
    ax = axes[1, 0]
    ax.plot(val_mape * 100, label='MAPE', alpha=0.7)
    ax.axhline(np.percentile(val_mape, 75) * 100, color='r', linestyle='--', label='75th %ile')
    ax.set_xlabel('Sample')
    ax.set_ylabel('MAPE (%)')
    ax.set_title('Prediction Error Over Time (Val)')
    ax.legend()

    # Surprise over time
    ax = axes[1, 1]
    ax.plot(surprise, label='Surprise (Actual - Pred)')
    ax.axhline(0, color='gray', linestyle='-', alpha=0.5)
    ax.set_xlabel('Sample')
    ax.set_ylabel('Vol Surprise')
    ax.set_title('ATM Vol Surprise Over Time (Val)')
    ax.legend()

    # Cumulative returns for strategies
    ax = axes[1, 2]
    bh_cum = np.cumprod(1 + val_data['returns']) - 1
    err_cum = np.cumprod(1 + error_returns) - 1
    ax.plot(bh_cum, label=f'Buy & Hold ({bh_cum[-1]:.1%})', linewidth=2)
    ax.plot(err_cum, label=f'Error Signal ({err_cum[-1]:.1%})', linewidth=2)
    ax.axhline(0, color='gray', linestyle='-', alpha=0.5)
    ax.set_xlabel('Sample')
    ax.set_ylabel('Cumulative Return')
    ax.set_title('Strategy Comparison (Val)')
    ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(PROJECT_ROOT, 'analysis/plots/surface_signal_analysis.png'), dpi=150)
    print(f"\nSaved analysis to analysis/plots/surface_signal_analysis.png")


if __name__ == "__main__":
    main()
