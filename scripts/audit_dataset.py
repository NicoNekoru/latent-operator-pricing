import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def audit_dataset():
    path = 'data/processed_dataset.parquet'
    print(f"Loading dataset from {path}...")

    try:
        df = pd.read_parquet(path)
    except FileNotFoundError:
        print("Dataset not found!")
        return

    print(f"Dataset Shape: {df.shape}")

    # 1. Check Target Prices
    print("\n--- Auditing Target Prices ---")
    # Target_Prices is a column of arrays (21 elements each)
    # Stack them to get a big matrix
    prices = np.stack(df['Target_Prices'].values)

    print(f"Total Option Prices: {prices.size}")

    zeros = (prices == 0).sum()
    negatives = (prices < 0).sum()
    nans = np.isnan(prices).sum()

    print(f"Exact Zeros: {zeros} ({zeros/prices.size*100:.4f}%)")
    print(f"Negatives: {negatives}")
    print(f"NaNs: {nans}")

    min_price = prices.min()
    max_price = prices.max()
    mean_price = prices.mean()

    print(f"Min Price: {min_price}")
    print(f"Max Price: {max_price}")
    print(f"Mean Price: {mean_price}")

    # Check distribution of small prices
    small_threshold = 1e-6
    small_prices = (prices < small_threshold) & (prices > 0)
    print(f"Prices < {small_threshold} (but > 0): {small_prices.sum()}")

    # 2. Check Input Features
    print("\n--- Auditing Input Features ---")
    features = np.stack(df['Input_Features'].values) # (N, 180)

    # Reshape to (N, 30, 6) to check specific channels
    # Channels: [LogReturn, RealizedVol, VIX, Volume, TNX, Buffett]
    features_reshaped = features.reshape(-1, 30, 6)

    channel_names = ['LogReturn', 'RealizedVol', 'VIX', 'Volume', 'TNX', 'Buffett']

    for i, name in enumerate(channel_names):
        channel_data = features_reshaped[:, :, i].flatten()
        nans = np.isnan(channel_data).sum()
        infs = np.isinf(channel_data).sum()
        zeros = (channel_data == 0).sum()

        print(f"Channel {name}:")
        print(f"  NaNs: {nans}")
        print(f"  Infs: {infs}")
        print(f"  Zeros: {zeros}")
        print(f"  Min: {channel_data.min():.4f}, Max: {channel_data.max():.4f}, Mean: {channel_data.mean():.4f}")

    # 3. Check for "Flat" Volatility (potential data error)
    # If RealizedVol is 0, Heston v0 will be 0, which might cause issues
    vol_channel = features_reshaped[:, :, 1]
    zero_vol_windows = (vol_channel.sum(axis=1) == 0).sum()
    print(f"\nWindows with all-zero RealizedVol: {zero_vol_windows}")

if __name__ == "__main__":
    audit_dataset()
