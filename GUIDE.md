# Project Guide: Neural SPDE Option Pricing

This guide explains how to set up, run, and extend the Neural SPDE Option Pricing project.

## 1. Installation

Ensure you have Python 3.8+ installed.

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/NicoNekoru/bsm-neural-spde.git
    cd bsm-neural-spde
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *Note: This project uses `torch`, `pandas`, `numpy`, `yfinance`, `QuantLib`, and `pandas_datareader`.*

## 2. Data Generation

The project uses a two-step data process:
1.  **Fetch Data:** Pulls market data (Price, Vol) and macro data (VIX, Rates, GDP) from `yfinance` and FRED.
2.  **Simulate Surfaces:** Uses the Heston model to generate "ground truth" option price surfaces for training.

**To generate the dataset:**
```bash
python src/data_loader.py
```
*   **Output:** `data/processed_dataset.parquet`
*   **Time:** This may take 5-10 minutes as it processes ~20 years of data for 4 indices.

## 3. Training the Model

To train the Neural Operator on the generated dataset:

```bash
python scripts/train.py
```
*   **Configuration:** You can modify `epochs`, `batch_size`, and `lr` in `scripts/train.py`.
*   **Output:** `models/neural_operator.pth` (Saved model weights).
*   **Input Dimension:** The model now uses **6 input features**:
    1.  Log Returns
    2.  Realized Volatility
    3.  VIX (Market Fear)
    4.  Volume (Liquidity)
    5.  10-Year Treasury Yield (Risk-Free Rate)
    6.  Buffett Indicator (Market Cap / GDP)

## 4. Backtesting & Analysis

To evaluate the model and trading strategies:

```bash
python scripts/backtest.py
```
*   **What it does:**
    *   Loads the trained model.
    *   Fetches fresh market/macro data.
    *   Runs trading strategies (Neural Skew, Mean Reversion, Momentum, Regime) on historical data.
    *   Generates performance plots for:
        *   **Train Set:** 2006-2023
        *   **Crisis Period:** 2006-2011 (Zoom in on 2008)
        *   **Test Set:** 2023-Present
*   **Output:** Plots are saved in `plots/`.

## 5. Project Structure

*   `src/`: Core source code.
    *   `data_loader.py`: Modular data fetching and processing.
    *   `dataset.py`: PyTorch Dataset definition.
    *   `models.py`: Neural Operator architecture (Encoder-Decoder).
    *   `strategies/`: Trading strategy implementations.
*   `scripts/`: Executable scripts for training and backtesting.
*   `data/`: Stores processed parquet files.
*   `models/`: Stores trained model weights.
*   `plots/`: Stores generated visualizations.

## 6. Extending the Project

*   **New Strategies:** Add a new class in `src/strategies/` inheriting from `BaseStrategy`.
*   **New Features:** Update `MacroData` in `src/data_loader.py` to fetch new indicators, and update `NeuralOperator` input dimension in `src/models.py`.
