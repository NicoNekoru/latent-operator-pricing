import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D
import os

from src.data_loader import MarketData, MacroData

def generate_baseline_clusters():
    print("Fetching Data for Baseline Clustering...")
    tickers = ['^GSPC', '^NDX', '^RUT', '^DJI']
    market = MarketData(tickers=tickers, start_date='2006-01-01').fetch()
    macro = MacroData(start_date='2006-01-01').fetch()
    merged = market.join(macro, how='left').ffill().dropna()

    # Features: VIX, RealizedVol, LogReturn
    # We want to see how well these separate "regimes"

    data = merged[['VIX', 'RealizedVol', 'LogReturn']].values

    # K-Means
    kmeans = KMeans(n_clusters=3, random_state=42)
    labels = kmeans.fit_predict(data)

    # Plot
    import seaborn as sns
    sns.set_theme(style="whitegrid")

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Color map
    # Use matplotlib colormap directly as seaborn doesn't manage 3D scatter colors easily
    cmap = plt.get_cmap('viridis', 3)

    sc = ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=labels, cmap=cmap, alpha=0.5, s=5)

    ax.set_xlabel('VIX')
    ax.set_ylabel('Realized Vol')
    ax.set_zlabel('Log Return')
    ax.set_title('Baseline Clusters (Raw Features)')

    plt.tight_layout()
    os.makedirs('plots', exist_ok=True)
    plt.savefig('plots/baseline_clusters.png', dpi=300)
    print("Saved plots/baseline_clusters.png")

if __name__ == "__main__":
    generate_baseline_clusters()
