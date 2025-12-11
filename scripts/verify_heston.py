
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Ensure src is in path
sys.path.append(os.getcwd())

from src.models import DeepONet, get_standard_grid
from src.data_loader import MarketData, MacroData
from src.strategies.bsm_dynamic import HestonRegimeStrategy
from src.strategies.benchmark import BenchmarkStrategy

def verify_heston():
    print("Initializing Model...")
    device = torch.device('cpu')
    model = DeepONet().to(device)
    # Load weights if available, else random is fine for integration test
    try:
        model.load_state_dict(torch.load('checkpoints/best_model.pt', map_location=device))
        print("Loaded checkpoint.")
    except:
        print("No checkpoint found, using random weights (OK for integration test).")

    print("Fetching Data (Short Period)...")
    tickers = ['^GSPC']
    start_date = '2023-01-01'
    end_date = '2023-03-31' # 3 months

    market = MarketData(tickers, start_date=start_date, end_date=end_date)
    macro = MacroData(start_date)
    merged_data = market.fetch().join(macro.fetch(), how='inner').dropna()

    print(f"Data Length: {len(merged_data)}")

    # Process
    strategies = [HestonRegimeStrategy(), BenchmarkStrategy(ticker='^GSPC')]

    # Prepare Data
    # For test, we generate random Z or use model encode if implemented?
    # model.encode needs history.
    # Let's simple use "dummy" prices for speed test of calibration?
    # Or actually run the pipeline:

    # Create windows
    seq_len = 30
    features = ['Close', 'High', 'Low', 'RealizedVol', 'VIX', 'TNX']

    # We need a proper dataset loop or just manual batch
    df = merged_data[merged_data['Ticker'] == '^GSPC']

    # Mock loop
    print("Running Calibration Loop on 10 days...")

    # Fake prices surface (normalized)
    # 21 points
    # Price exp(-rT) * BS(S=1, K...)
    # We generate "Market" prices using Heston pricer itself to see if it recovers?
    # Or just run strategy on random model output.

    # Let's run the strategy signal generation on dummy prices to verify calls.
    # Prices: (T, 21)
    T = 10
    dummy_prices = np.random.uniform(0.01, 0.2, (T, 21))

    strat = HestonRegimeStrategy()
    print("Starting generation...")
    signals = strat.generate_signals(np.zeros((T, 16)), prices=dummy_prices)

    print("Signals Generated:")
    print(signals)

    if len(signals) == T:
        print("SUCCESS: Heston Strategy ran and produced signals.")
    else:
        print("FAILURE: Signal length mismatch.")

import torch
if __name__ == "__main__":
    verify_heston()
