# Project Specification: Latent Manifold Learning of Market Regimes via Neural Operators

## 1. Executive Summary
**Research Question:** Can a Neural Operator architecture learn a low-dimensional latent manifold that characterizes distinct financial market regimes (e.g., "Calm," "Crisis," "Recovery") more effectively than traditional parametric calibrations?

**Objective:**
1.  Ingest real historical market data (SPX/QQQ).
2.  Generate a semi-synthetic "Ground Truth" dataset of option prices using a Stochastic Volatility model (Heston) to simulate complex, non-Black-Scholes dynamics.
3.  Train a **Manifold-Preserving Neural Operator** (Autoencoder architecture) to map market history to option price surfaces.
4.  Analyze the learned Latent Space ($Z$) to demonstrate that market regimes cluster topologically and that interpolation along the manifold yields superior pricing stability.

**Extended Vision:**
While the initial Proof of Concept (POC) focuses on a low-dimensional latent space ($d \approx 3-4$), the ultimate goal is to expand the embedding to include standard BSM parameters (volatility, risk-free rate), statistical estimators (Hurst exponent), and technical indicators. This will allow the neural operator to efficiently traverse the full "solution manifold" of option pricing dynamics, effectively learning a universal operator for market regimes.

---

## 2. Technology Stack
*   **Language:** Python 3.10+
*   **Data Source:** `yfinance` (for underlying asset history), `numpy` (for simulation).
*   **Deep Learning Framework:** `PyTorch` (preferred) or `JAX`.
*   **Dimensionality Reduction:** `scikit-learn` (PCA), `umap-learn` (UMAP).
*   **Visualization:** `matplotlib`, `seaborn`, `plotly` (for 3D manifold plotting).

---

## 3. Phase I: Data Acquisition & Simulation
*Since free high-fidelity historical option data is unavailable, we will use **Real Market Dynamics** to drive a **Complex Pricing Model**. This ensures the input noise is realistic (heavy tails, volatility clustering) while providing a mathematical ground truth for the network to learn.*

### Step 3.1: Scrape Underlying History
**Action:** Download daily OHLCV data for liquid indices.
*   **Tickers:** `^GSPC` (S&P 500), `^NDX` (Nasdaq 100).
*   **Range:** Jan 1, 2010 – Present.
*   **Feature Engineering:**
    *   Calculate Log Returns: $r_t = \ln(S_t / S_{t-1})$.
    *   Calculate Realized Volatility: 21-day rolling standard deviation.
    *   **Normalization:** Standardize inputs (Z-score) based on *training set statistics only*.

### Step 3.2: The Heston World Simulation (The Target Generation)
**Rationale:** Standard Black-Scholes is too simple. We use the Heston Model to generate option prices because it admits stochastic volatility, creating a complex surface that the Neural Net must "solve."

**Simulation Logic:**
For every day $t$ in the dataset:
1.  Extract Spot Price $S_t$ and Estimate Current Volatility $v_t$ from real data.
2.  **Assumed Parameters:** Fix Heston parameters (Mean reversion $\kappa$, Vol-of-Vol $\xi$, Correlation $\rho$) or randomize them slightly to ensure robustness.
3.  **Generate Surface:** Calculate option prices for:
    *   **Maturities:** $\tau = [1 \text{ month}, 3 \text{ months}, 6 \text{ months}]$.
    *   **Moneyness:** $K/S = [0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2]$.
    *   *Total Output Dimension:* 3 Maturities $\times$ 7 Strikes = 21 Prices per day.
4.  **Method:** Use QuantLib or a Fourier-Cos method (fast integration) to solve the Heston price. *Do not use Monte Carlo for training data generation if possible, it is too slow and noisy.*

**Deliverable:** A Parquet file `market_options_dataset.parquet` containing:
*   Index: Date
*   Features ($X$): Past 30 days of returns and realized vol.
*   Targets ($Y$): 21 normalized option prices (the "Surface").

---

## 4. Phase II: Model Architecture (The Neural Operator)
We treat this as an **Inverse Problem / Operator Learning** task. The network acts as an Autoencoder.

### Component A: The Encoder (Market State $\to$ Manifold)
*   **Input:** Time series of market state (Shape: `[Batch, 30, 2]` - 30 days, Return & Vol).
*   **Architecture:**
    *   1D Convolutional Layers (to capture temporal dependencies/path roughness).
    *   OR: LSTM/GRU layer.
*   **Bottleneck (The Latent Space $Z$):**
    *   Dense layer mapping to dimension $d=3$ or $d=4$.
    *   *Activation:* Linear (allows the manifold to stretch infinitely) or Tanh.
    *   **Constraint:** Apply a variational penalty (VAE) OR simply an L2 sparsity penalty to force efficient representation.

### Component B: The Decoder (Manifold $\to$ Price Surface)
*   **Input:** Latent vector $z$ (Shape: `[Batch, d]`).
*   **Architecture:** Multi-Layer Perceptron (MLP).
*   **Output:** Option Surface (Shape: `[Batch, 21]`).
*   **Physics-Informed Constraint (Optional but Recommended):**
    *   Add a "Call Spread" penalty in the loss function: $Price(K_1) \ge Price(K_2)$ for $K_1 < K_2$. This forces the network to respect no-arbitrage rules.

---

## 5. Phase III: Training Protocol
*   **Split:**
    *   **Train:** 2010 – 2022.
    *   **Validation:** 2023.
    *   **Test:** 2024 – Present (Out of Time sample).
*   **Loss Function:** Mean Squared Error (MSE) on price + Penalty for arbitrage violations.
*   **Optimizer:** AdamW with Cosine Annealing scheduler.
*   **Early Stopping:** Monitor Validation Loss.

---

## 6. Phase IV: Analysis & Tests for Structure
*This is the core scientific contribution. We prove the Latent Space is meaningful.*

### Test 1: Topological Clustering (Regime Identification)
**Hypothesis:** Distinct market behaviors map to distinct regions in $Z$.
*   **Procedure:**
    1.  Feed all Test Data through the Encoder to get $Z_{test}$.
    2.  Color-code points by "Market State" (e.g., Red = High Volatility/Crash, Blue = Low Volatility).
    3.  **Metric:** Calculate the *Silhouette Score* of these clusters in Latent Space.
    4.  **Visual:** Plot 3D scatter of $Z$. Do the "Crisis" points form a separate island or a long tail?

### Test 2: Manifold Interpolation (The "Physics" Check)
**Hypothesis:** Linear movement in Latent Space represents a realistic evolution of market physics, whereas linear interpolation in Data Space does not.
*   **Procedure:**
    1.  Pick Point A (Calm Day) and Point B (Crisis Day).
    2.  **Path 1 (Latent):** Draw a line between $z_A$ and $z_B$. Decode points along the line into surfaces.
    3.  **Path 2 (Naive):** Average the prices of Day A and Day B directly.
    4.  **Check:** Compute the implied volatility surface for the interpolated points.
    5.  **Pass Criteria:** Path 1 yields smooth, smile-shaped volatility curves. Path 2 yields jagged or arbitrage-violating curves.

### Test 3: Temporal Trajectory Analysis
**Hypothesis:** The market moves continuously along the manifold.
*   **Procedure:** Plot the trajectory of $z_t, z_{t+1}, z_{t+2}...$ for the month of March 2020 (COVID Crash).
*   **Visual:** Does the trajectory look like a random walk (Brownian motion), or does it "jump" to a new attractor basin?
*   **Insight:** If it jumps, the Neural SPDE has identified a "Phase Transition" in the market.

---

## 7. Implementation Roadmap (Checklist)

- [ ] **Week 1: Data Pipeline**
    - [ ] Implement `yfinance` scraper.
    - [ ] Implement Heston pricing engine (QuantLib or Fourier method).
    - [ ] Generate `market_options_dataset.parquet`.

- [ ] **Week 2: Model Training**
    - [ ] Build Encoder (Conv1D) and Decoder (MLP) in PyTorch.
    - [ ] Train on 2010-2022 data.
    - [ ] Achieve MSE < $0.01$ on normalized prices.

- [ ] **Week 3: Latent Exploration**
    - [ ] Extract $Z$ vectors for 2023-2025.
    - [ ] Run PCA/UMAP on $Z$.
    - [ ] Generate 3D Scatter plots color-coded by Volatility.

- [ ] **Week 4: Structural Tests**
# Project Specification: Latent Manifold Learning of Market Regimes via Neural Operators

## 1. Executive Summary
**Research Question:** Can a Neural Operator architecture learn a low-dimensional latent manifold that characterizes distinct financial market regimes (e.g., "Calm," "Crisis," "Recovery") more effectively than traditional parametric calibrations?

**Objective:**
1.  Ingest real historical market data (SPX/QQQ).
2.  Generate a semi-synthetic "Ground Truth" dataset of option prices using a Stochastic Volatility model (Heston) to simulate complex, non-Black-Scholes dynamics.
3.  Train a **Manifold-Preserving Neural Operator** (Autoencoder architecture) to map market history to option price surfaces.
4.  Analyze the learned Latent Space ($Z$) to demonstrate that market regimes cluster topologically and that interpolation along the manifold yields superior pricing stability.

**Extended Vision:**
While the initial Proof of Concept (POC) focuses on a low-dimensional latent space ($d \approx 3-4$), the ultimate goal is to expand the embedding to include standard BSM parameters (volatility, risk-free rate), statistical estimators (Hurst exponent), and technical indicators. This will allow the neural operator to efficiently traverse the full "solution manifold" of option pricing dynamics, effectively learning a universal operator for market regimes.

---

## 2. Technology Stack
*   **Language:** Python 3.10+
*   **Data Source:** `yfinance` (for underlying asset history), `numpy` (for simulation).
*   **Deep Learning Framework:** `PyTorch` (preferred) or `JAX`.
*   **Dimensionality Reduction:** `scikit-learn` (PCA), `umap-learn` (UMAP).
*   **Visualization:** `matplotlib`, `seaborn`, `plotly` (for 3D manifold plotting).

---

## 3. Phase I: Data Acquisition & Simulation
*Since free high-fidelity historical option data is unavailable, we will use **Real Market Dynamics** to drive a **Complex Pricing Model**. This ensures the input noise is realistic (heavy tails, volatility clustering) while providing a mathematical ground truth for the network to learn.*

### Step 3.1: Scrape Underlying History
**Action:** Download daily OHLCV data for liquid indices.
*   **Tickers:** `^GSPC` (S&P 500), `^NDX` (Nasdaq 100).
*   **Range:** Jan 1, 2010 – Present.
*   **Feature Engineering:**
    *   Calculate Log Returns: $r_t = \ln(S_t / S_{t-1})$.
    *   Calculate Realized Volatility: 21-day rolling standard deviation.
    *   **Normalization:** Standardize inputs (Z-score) based on *training set statistics only*.

### Step 3.2: The Heston World Simulation (The Target Generation)
**Rationale:** Standard Black-Scholes is too simple. We use the Heston Model to generate option prices because it admits stochastic volatility, creating a complex surface that the Neural Net must "solve."

**Simulation Logic:**
For every day $t$ in the dataset:
1.  Extract Spot Price $S_t$ and Estimate Current Volatility $v_t$ from real data.
2.  **Assumed Parameters:** Fix Heston parameters (Mean reversion $\kappa$, Vol-of-Vol $\xi$, Correlation $\rho$) or randomize them slightly to ensure robustness.
3.  **Generate Surface:** Calculate option prices for:
    *   **Maturities:** $\tau = [1 \text{ month}, 3 \text{ months}, 6 \text{ months}]$.
    *   **Moneyness:** $K/S = [0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2]$.
    *   *Total Output Dimension:* 3 Maturities $\times$ 7 Strikes = 21 Prices per day.
4.  **Method:** Use QuantLib or a Fourier-Cos method (fast integration) to solve the Heston price. *Do not use Monte Carlo for training data generation if possible, it is too slow and noisy.*

**Deliverable:** A Parquet file `market_options_dataset.parquet` containing:
*   Index: Date
*   Features ($X$): Past 30 days of returns and realized vol.
*   Targets ($Y$): 21 normalized option prices (the "Surface").

---

## 4. Phase II: Model Architecture (The Neural Operator)
We treat this as an **Inverse Problem / Operator Learning** task. The network acts as an Autoencoder.

### Component A: The Encoder (Market State $\to$ Manifold)
*   **Input:** Time series of market state (Shape: `[Batch, 30, 2]` - 30 days, Return & Vol).
*   **Architecture:**
    *   1D Convolutional Layers (to capture temporal dependencies/path roughness).
    *   OR: LSTM/GRU layer.
*   **Bottleneck (The Latent Space $Z$):**
    *   Dense layer mapping to dimension $d=3$ or $d=4$.
    *   *Activation:* Linear (allows the manifold to stretch infinitely) or Tanh.
    *   **Constraint:** Apply a variational penalty (VAE) OR simply an L2 sparsity penalty to force efficient representation.

### Component B: The Decoder (Manifold $\to$ Price Surface)
*   **Input:** Latent vector $z$ (Shape: `[Batch, d]`).
*   **Architecture:** Multi-Layer Perceptron (MLP).
*   **Output:** Option Surface (Shape: `[Batch, 21]`).
*   **Physics-Informed Constraint (Optional but Recommended):**
    *   Add a "Call Spread" penalty in the loss function: $Price(K_1) \ge Price(K_2)$ for $K_1 < K_2$. This forces the network to respect no-arbitrage rules.

---

## 5. Phase III: Training Protocol
*   **Split:**
    *   **Train:** 2010 – 2022.
    *   **Validation:** 2023.
    *   **Test:** 2024 – Present (Out of Time sample).
*   **Loss Function:** Mean Squared Error (MSE) on price + Penalty for arbitrage violations.
*   **Optimizer:** AdamW with Cosine Annealing scheduler.
*   **Early Stopping:** Monitor Validation Loss.

---

## 6. Phase IV: Analysis & Tests for Structure
*This is the core scientific contribution. We prove the Latent Space is meaningful.*

### Test 1: Topological Clustering (Regime Identification)
**Hypothesis:** Distinct market behaviors map to distinct regions in $Z$.
*   **Procedure:**
    1.  Feed all Test Data through the Encoder to get $Z_{test}$.
    2.  Color-code points by "Market State" (e.g., Red = High Volatility/Crash, Blue = Low Volatility).
    3.  **Metric:** Calculate the *Silhouette Score* of these clusters in Latent Space.
    4.  **Visual:** Plot 3D scatter of $Z$. Do the "Crisis" points form a separate island or a long tail?

### Test 2: Manifold Interpolation (The "Physics" Check)
**Hypothesis:** Linear movement in Latent Space represents a realistic evolution of market physics, whereas linear interpolation in Data Space does not.
*   **Procedure:**
    1.  Pick Point A (Calm Day) and Point B (Crisis Day).
    2.  **Path 1 (Latent):** Draw a line between $z_A$ and $z_B$. Decode points along the line into surfaces.
    3.  **Path 2 (Naive):** Average the prices of Day A and Day B directly.
    4.  **Check:** Compute the implied volatility surface for the interpolated points.
    5.  **Pass Criteria:** Path 1 yields smooth, smile-shaped volatility curves. Path 2 yields jagged or arbitrage-violating curves.

### Test 3: Temporal Trajectory Analysis
**Hypothesis:** The market moves continuously along the manifold.
*   **Procedure:** Plot the trajectory of $z_t, z_{t+1}, z_{t+2}...$ for the month of March 2020 (COVID Crash).
*   **Visual:** Does the trajectory look like a random walk (Brownian motion), or does it "jump" to a new attractor basin?
*   **Insight:** If it jumps, the Neural SPDE has identified a "Phase Transition" in the market.

---

## 7. Implementation Roadmap (Checklist)

- [ ] **Week 1: Data Pipeline**
    - [ ] Implement `yfinance` scraper.
    - [ ] Implement Heston pricing engine (QuantLib or Fourier method).
    - [ ] Generate `market_options_dataset.parquet`.

- [ ] **Week 2: Model Training**
    - [ ] Build Encoder (Conv1D) and Decoder (MLP) in PyTorch.
    - [ ] Train on 2010-2022 data.
    - [ ] Achieve MSE < $0.01$ on normalized prices.

- [ ] **Week 3: Latent Exploration**
    - [ ] Extract $Z$ vectors for 2023-2025.
    - [ ] Run PCA/UMAP on $Z$.
    - [ ] Generate 3D Scatter plots color-coded by Volatility.

- [ ] **Week 4: Structural Tests**
    - [ ] Run Interpolation Test (Latent vs Naive).
    - [ ] Run Arbitrage Checks on decoded surfaces.
    - [ ] Write Analysis Report.

---

## 8. Project Structure
```text
project_root/
|
+-- data/
|   +-- processed_dataset.parquet # Generated Option Data
|
+-- models/
|   +-- neural_operator.pth       # Trained Model
|
+-- plots/                        # Generated Visualizations (PNG)
|
+-- scripts/                      # Executable Scripts
|   +-- train.py                  # Train the model
|   +-- evaluate.py               # Evaluate on Test Set
|   +-- analysis.py               # Generate Visualizations
|   +-- backtest.py               # Run Strategy Benchmarks
|
+-- src/                          # Reusable Modules
|   +-- strategies/               # Trading Strategy Implementations
|   +-- data_loader.py            # Data Scraper & Heston Simulator
|   +-- dataset.py                # PyTorch Dataset Class
|   +-- models.py                 # Neural Operator Architecture
|   +-- utils.py                  # Metric Calculations
|
+-- setup.py                      # Package Installation
+-- requirements.txt              # Dependencies
```

## 9. Usage

### Installation
```bash
pip install -r requirements.txt
pip install -e .
```

### Training
```bash
python scripts/train.py
```

### Evaluation
```bash
python scripts/evaluate.py
```

### Analysis & Visualization
Generates latent space plots in `plots/`:
```bash
python scripts/analysis.py
```

### Strategy Benchmarking
Runs backtests and generates equity curves:
```bash
python scripts/backtest.py
```