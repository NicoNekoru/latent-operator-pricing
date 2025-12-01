import torch
import numpy as np
import pandas as pd
from sklearn.manifold import trustworthiness
from sklearn.neighbors import NearestNeighbors
from src.models import DeepONet, get_standard_grid
from src.data_loader import MarketData, MacroData, HestonGenerator
from src.utils import calculate_metrics
import scipy.stats as si

def calculate_continuity(X, X_embedded, n_neighbors=5):
    """
    Calculates Continuity metric (Venna & Kaski).
    Continuity is Trustworthiness with X and X_embedded swapped.
    """
    return trustworthiness(X_embedded, X, n_neighbors=n_neighbors)

def bsm_price(S, K, T, r, sigma, type='call'):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if type == 'call':
        price = S * si.norm.cdf(d1, 0.0, 1.0) - K * np.exp(-r * T) * si.norm.cdf(d2, 0.0, 1.0)
    else:
        price = K * np.exp(-r * T) * si.norm.cdf(-d2, 0.0, 1.0) - S * si.norm.cdf(-d1, 0.0, 1.0)

    return price

def bsm_implied_vol(price, S, K, T, r, type='call'):
    """
    Simple bisection method to find IV.
    """
    sigma_low = 0.001
    sigma_high = 5.0

    for i in range(50):
        sigma = (sigma_low + sigma_high) / 2
        p = bsm_price(S, K, T, r, sigma, type)

        if abs(p - price) < 1e-5:
            return sigma

        if p < price:
            sigma_low = sigma
        else:
            sigma_high = sigma

    return sigma

def run_analysis():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load Model
    model = DeepONet(input_channels=6, latent_dim=16).to(device)
    base_grid = get_standard_grid(device)
    try:
        model.load_state_dict(torch.load('models/deeponet.pth', map_location=device))
    except FileNotFoundError:
        print("Model not found.")
        return
    model.eval()

    # Load Data (Validation Set 2022 for consistency with paper)
    print("Fetching Data...")
    tickers = ['^GSPC']
    market = MarketData(tickers=tickers, start_date='2022-01-01', end_date='2022-12-31').fetch()
    macro = MacroData(start_date='2022-01-01').fetch()
    merged_data = market.join(macro, how='left').ffill().dropna()

    df = merged_data[merged_data['Ticker'] == '^GSPC']
    vol_feature = np.log(df['Volume'] + 1) / 20.0

    # Prepare Data
    X_high_dim = []
    Z_latent = []
    Baseline_features = []

    # For BSM Comparison
    deeponet_errors = {'ATM': [], 'Deep_OTM': []}
    bsm_errors = {'ATM': [], 'Deep_OTM': []}

    simulator = HestonGenerator()

    # Subsample for manifold metrics (computationally expensive)
    indices = range(30, len(df), 5)

    print(f"Processing {len(indices)} samples...")

    for i in indices:
        row = df.iloc[i]
        spot = row['Close']
        realized_vol = row['RealizedVol']
        vix = row['VIX']
        log_ret = row['LogReturn']

        # 1. Input High-Dim (Flattened Window)
        window = df.iloc[i-30:i]
        window_vol = vol_feature.iloc[i-30:i]

        features = np.stack([
            window['LogReturn'].values,
            window['RealizedVol'].values,
            window['VIX'].values,
            window_vol.values,
            window['TNX'].values,
            window['Buffett_Ind'].values
        ], axis=1)

        X_high_dim.append(features.flatten())

        # 2. Latent Z
        x_tensor = torch.tensor(features.reshape(1, 30, 6), dtype=torch.float32).to(device)
        with torch.no_grad():
            y_pred, z = model(x_tensor, base_grid.expand(1, -1, -1))
            y_pred = y_pred.cpu().numpy().flatten()
            z = z.cpu().numpy().flatten()

        Z_latent.append(z[:3]) # Use first 3 dims as in paper visualization

        # 3. Baseline Features
        Baseline_features.append([vix, realized_vol, log_ret])

        # 4. BSM vs DeepONet Error Analysis
        # Ground Truth Surface
        v0 = realized_vol ** 2
        simulator.setup_engine(spot, v0, 2.0, v0, 0.3, -0.7)
        surface_true = simulator.generate(spot)

        # Indices: ATM=10 (Moneyness 1.0), Deep OTM=20 (Moneyness 1.2, longest maturity)
        # Actually surface is 3 maturities x 7 strikes.
        # Strikes: 0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2
        # Maturities: 1M, 3M, 6M
        # Index 10 is 3M ATM (Row 1, Col 3 -> 1*7 + 3 = 10)
        # Index 20 is 6M Deep OTM (Row 2, Col 6 -> 2*7 + 6 = 20)

        idx_atm = 10
        idx_otm = 20

        price_atm_true = surface_true[idx_atm]
        price_otm_true = surface_true[idx_otm]

        # DeepONet Prices
        price_atm_pred = y_pred[idx_atm]
        price_otm_pred = y_pred[idx_otm]

        # BSM Prices (Calibrate to ATM)
        # Find IV that matches ATM price
        # Note: price_atm_true is normalized (Price/Spot). We need dollar price for BSM function.
        price_atm_dollar = price_atm_true * spot

        T_atm = 3/12
        K_atm = spot * 1.0
        iv_atm = bsm_implied_vol(price_atm_dollar, spot, K_atm, T_atm, 0.04)

        if i % 10 == 0:
            print(f"Sample {i}: Spot={spot:.2f}, ATM Price (Norm)={price_atm_true:.4f}, IV={iv_atm:.4f}")

        # Price Deep OTM using this IV (Flat Vol Assumption)
        T_otm = 6/12
        K_otm = spot * 1.2
        price_otm_bsm_dollar = bsm_price(spot, K_otm, T_otm, 0.04, iv_atm)
        price_atm_bsm_dollar = bsm_price(spot, K_atm, T_atm, 0.04, iv_atm)

        # Normalize BSM prices for comparison with DeepONet (which predicts normalized)
        price_otm_bsm = price_otm_bsm_dollar / spot
        price_atm_bsm = price_atm_bsm_dollar / spot

        # Calculate MAPEs
        def get_mape(pred, true):
            return abs(pred - true) / (true + 1e-6)

        deeponet_errors['ATM'].append(get_mape(price_atm_pred, price_atm_true))
        deeponet_errors['Deep_OTM'].append(get_mape(price_otm_pred, price_otm_true))

        bsm_errors['ATM'].append(get_mape(price_atm_bsm, price_atm_true))
        bsm_errors['Deep_OTM'].append(get_mape(price_otm_bsm, price_otm_true))

    X_high_dim = np.array(X_high_dim)
    Z_latent = np.array(Z_latent)
    Baseline_features = np.array(Baseline_features)

    print("\n--- Manifold Quality Metrics (Venna & Kaski) ---")
    n_neighbors = 12

    # Trustworthiness
    t_latent = trustworthiness(X_high_dim, Z_latent, n_neighbors=n_neighbors)
    t_baseline = trustworthiness(X_high_dim, Baseline_features, n_neighbors=n_neighbors)

    # Continuity (Swap X and Emb)
    c_latent = trustworthiness(Z_latent, X_high_dim, n_neighbors=n_neighbors)
    c_baseline = trustworthiness(Baseline_features, X_high_dim, n_neighbors=n_neighbors)

    print(f"Trustworthiness (k={n_neighbors}):")
    print(f"  DeepONet Latent: {t_latent:.4f}")
    print(f"  Baseline (VIX):  {t_baseline:.4f}")

    print(f"Continuity (k={n_neighbors}):")
    print(f"  DeepONet Latent: {c_latent:.4f}")
    print(f"  Baseline (VIX):  {c_baseline:.4f}")

    print("\n--- Pricing Error in the Wings (Deep OTM) ---")
    print(f"DeepONet ATM MAPE: {np.mean(deeponet_errors['ATM'])*100:.2f}%")
    print(f"BSM (Flat) ATM MAPE: {np.mean(bsm_errors['ATM'])*100:.2f}% (Calibrated)")

    print(f"DeepONet Deep OTM MAPE: {np.mean(deeponet_errors['Deep_OTM'])*100:.2f}%")
    print(f"BSM (Flat) Deep OTM MAPE: {np.mean(bsm_errors['Deep_OTM'])*100:.2f}%")

if __name__ == "__main__":
    run_analysis()
