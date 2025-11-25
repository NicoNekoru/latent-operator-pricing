# Trading Strategies

This directory contains the implementation of various trading strategies used to benchmark the Neural Operator model.

## 1. Benchmark Strategy (`benchmark.py`)
*   **Rationale:** The standard "Buy & Hold" baseline.
*   **Logic:** Always Long (Signal = 1).
*   **Purpose:** To compare active strategies against the market beta.

## 2. Latent Regime Strategy (`regime.py`)
*   **Rationale:** The latent space norm ($||z||$) is a proxy for "Market Temperature."
*   **Logic:**
    *   Calculate $||z||$.
    *   If $||z|| > 80^{th}$ percentile (High Vol Regime), go to Cash.
    *   Else, go Long.
*   **Purpose:** To test if the latent space successfully encodes risk regimes.

## 3. Latent Momentum Strategy (`momentum.py`)
*   **Rationale:** Volatility tends to cluster. If latent volatility is rising, it will likely continue to rise.
*   **Logic:**
    *   Calculate Moving Average (MA) of $||z||$.
    *   If Current $||z|| >$ MA (Vol is rising), go Defensive (Cash).
    *   Else, go Long.
*   **Purpose:** To exploit the autocorrelation of volatility using the latent representation.

## 4. Latent Mean Reversion (`mean_reversion.py`)
*   **Rationale:** Extreme volatility spikes are often followed by a reversion to the mean (calm).
*   **Logic:**
    *   Calculate Z-Score of $||z||$.
    *   If Z-Score > 2.0 (Extreme Fear), Buy the Dip (Long).
    *   If Z-Score < -2.0 (Extreme Complacency), Cash/Short.
*   **Purpose:** To catch "oversold" conditions using latent anomalies.

## 5. Neural Skew Strategy (`skew.py`)
*   **Rationale:** The Neural Operator predicts the full Option Price Surface. The "Skew" (difference between OTM Put and OTM Call prices) is a measure of the market's fear of a crash.
*   **Logic:**
    *   Calculate Implied Skew: $P_{Put} - P_{Call}$.
    *   If Skew > Threshold (Steep Smile), the market is pricing in a crash $\to$ Go Short/Defensive.
    *   Else, go Long.
*   **Purpose:** To leverage the **physics-consistent** output of the model (the shape of the surface) rather than just the latent vector.
