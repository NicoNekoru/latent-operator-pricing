# Guide to Project Visualizations

This document explains the visualizations from the Neural Operator analysis. These graphs show how the model represents complex financial market dynamics in a low-dimensional "Latent Manifold."

## 1. Latent Space Topology
**File:** `plots/latent_space_3d.png`

### What it is
A 3D scatter plot where each point represents a single trading day's market state compressed into the 3-dimensional latent space ($z \in \mathbb{R}^3$).

### How to Read It
- **Axes:** The three spatial dimensions (Latent 1, 2, 3) are the abstract features learned by the Neural Operator.
- **Color:** Represents **Realized Volatility** (The standard deviation of returns over the past 30 days).
    - **Blue/Cool Colors:** Low Volatility ("Calm" markets).
    - **Red/Hot Colors:** High Volatility ("Crisis" markets).

### Key Insight
You should see distinct clusters or gradients. "Calm" days should group together, and "Crisis" days should form their own region. The fact that these are separated and connected by a gradient implies the model has learned a **topological map of market risk**.

---

## 2. Physics-Consistent Interpolation
**File:** `plots/interpolation_test.png`

### What it is
A comparison of two ways to estimate what happens "between" two market states (e.g., transitioning from a Calm day to a Volatile day).
- **Left Panel (Latent Space):** Interpolating the latent vectors $z$ first, then decoding to prices.
- **Right Panel (Naive Data Space):** A direct average of the option prices of the two days.

### Key Insight
- **Naive Interpolation (Right):** The lines are linear averages. They look "stiff" and may violate no-arbitrage constraints.
- **Latent Interpolation (Left):** The curves deform smoothly and non-linearly. This shows the Neural Operator understands the **non-linear physics** of option pricing (the "Smile").

---

## 3. Market Trajectory
**File:** `plots/trajectory_2023.png`

### What it is
A 3D line plot tracing the path of the market through the latent space over a specific period (2023).

### How to Read It
- **Line:** The continuous path of the market state.
- **Color:** Represents time progression (Viridis colormap).

### Key Insight
The market does not jump randomly; it flows continuously. You might see "orbits" (cycles of volatility) or "phase transitions" (rapid movement from one region to another during a shock).

---

## 4. Latent Physics Landscape
**File:** `plots/latent_landscape.png`

### What it is
A "Heatmap" or "Contour Map" showing how the **Option Price** varies across the latent space. We fix Latent Dim 3 and vary Dims 1 & 2.

### How to Read It
- **X/Y Axes:** Latent Dimensions 1 and 2.
- **Color/Contours:** The price of an **At-The-Money (ATM) Call Option**.
    - **Brighter/Yellow:** Higher Prices (Implies Higher Volatility).
    - **Darker/Purple:** Lower Prices.

### Key Insight
This reveals the **"Geometry of Value."** You can see that price is a smooth function of the latent coordinates.

---

## 5. Latent Velocity Field
**File:** `plots/latent_velocity.png`

### What it is
A "Flow Map" (Quiver Plot) showing the average direction and speed of market changes at different points in the latent space.

### How to Read It
- **Arrows:** Point in the direction the market tends to move from that location.
- **Length:** Indicates the speed of change.

### Key Insight
- **Attractors:** If arrows point inwards towards a center, that region is a stable "attractor" (e.g., the market tends to revert to a calm baseline).
- **Instability:** If arrows point outwards or swirl rapidly, that region represents unstable or transitional dynamics.

---

## 6. Index Comparison (Generalization Test)
**File:** `plots/index_comparison.png`

### What it is
A time-series comparison of the model's performance on two different indices:
1.  **S&P 500 (^GSPC):** The index used for training.
2.  **Nasdaq 100 (^NDX):** A test index to evaluate generalization.

### How to Read It
- **Top Panel (Price):**
    - **Solid Lines:** The Ground Truth (Heston) ATM Call Price.
    - **Dashed Lines:** The Model's Prediction.
- **Bottom Panel (Error):**
    - **Filled Area:** The **Mean Absolute Error (MAE)** between the model and the truth over time.

### Key Insight
- **Training Fit:** The model should track the S&P 500 (Blue) very closely.
- **Generalization:** If the model tracks the Nasdaq 100 (Orange) well, it proves that the learned "Latent Manifold" is **universal** across indices with similar physics.

---

## 7. Strategy Benchmarking
**File:** `plots/strategy_comparison.png`

### What it is
A cumulative return comparison of different trading strategies informed by the model's latent space signals.

### Strategies
1.  **Benchmark:** Buy & Hold S&P 500.
2.  **Latent Regime:** Exit market when Latent Volatility is high.
3.  **Latent Momentum:** Follow the trend of Latent Volatility.
4.  **Latent Mean Reversion:** Bet on reversion during extreme volatility.

### Key Insight
Demonstrates the practical utility of the learned latent representation for risk management and alpha generation.

---

## 8. PCA Projection (2D)
**File:** `plots/pca_projection.png`

### What it is
A projection of the high-dimensional latent space onto its first two Principal Components. Points are colored by Realized Volatility.

### How to Read It
- **Axes:** PC1 and PC2 represent the directions of maximum variance in the latent space.
- **Gradient:** A smooth color gradient from Blue (Low Vol) to Red (High Vol) along PC1 indicates that the primary factor learned by the model is indeed volatility.
- **Spread:** The spread along PC2 indicates secondary features (possibly skew or kurtosis).

---

## 9. t-SNE Manifold Projection
**File:** `plots/tsne_projection.png`

### What it is
A non-linear projection of the latent manifold into 2D using t-Distributed Stochastic Neighbor Embedding.

### How to Read It
- **Clusters:** t-SNE preserves local neighborhoods. Distinct clusters indicate distinct market "regimes" that are topologically separated.
- **Continuity:** If the points form a continuous "snake" or curve, it suggests the market moves smoothly between states rather than jumping between discrete clusters.

---

## 10. Parallel Coordinates Plot
**File:** `plots/parallel_coordinates.png`

### What it is
A visualization of the latent vector values ($Z_1, Z_2, Z_3$) for different volatility regimes. Each line represents a single day's market state.

### How to Read It
- **Disentanglement:** If "High Volatility" lines (Yellow) all have high $Z_1$ and low $Z_2$, while "Low Volatility" lines (Purple) have low $Z_1$, it means $Z_1$ is disentangled and represents volatility.
- **Crossing Lines:** If lines cross chaotically, the dimensions are entangled.

---

## 11. Latent Feature Correlation Heatmap
**File:** `plots/latent_correlation.png`

### What it is
The correlation matrix between the learned Latent Dimensions and the physical "Realized Volatility".

### How to Read It
- **Feature Attribution:** If `Latent_1` has a +0.9 correlation with `Realized_Vol`, that supports interpreting volatility as a primary learned factor.
- **Independence:** Low correlation between latent dimensions (off-diagonal elements close to 0) indicates an efficient, orthogonal encoding.
