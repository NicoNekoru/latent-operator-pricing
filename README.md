# Latent Manifold Learning of Market Regimes via Neural Operators

This repository contains the code for the experiments documented in [`writeup/final.pdf`](https://github.com/NicoNekoru/latent-operator-pricing/blob/master/writeup/final.pdf). The project studies whether a neural operator's latent manifold can learn option price surfaces that generalize across market regimes better than standard parametric or purely data-driven baselines.

The main model is `SpectralDeepONet` in `src/models.py`. It maps a 30-day market state window to a 21-point option surface over moneyness and maturity. Training uses a short synthetic Rough Heston pre-training phase, then fine-tunes on empirical option surfaces with a differentiable BSM pricing layer. The analysis scripts inspect reconstruction error, latent geometry, manifold interpolation, inference speed, and simple trading/backtest signals.

## Repository Layout

```text
src/                  model, dataset, pricing utilities, and strategies
analysis/scripts/     training, diagnostics, visualizations, backtests
analysis/notebooks/   exploratory notebooks
data/                 local parquet data used by the experiments
writeup/              paper source
```

## Reproduce

Create an environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

The training and analysis scripts expect:

```text
data/processed_dataset.parquet
```

That file is present in this workspace. If rebuilding from raw option CSVs, place the raw files under `data/raw/` and run:

```bash
cd data
python compile_data.py
cd ..
```

Train the model:

```bash
python analysis/scripts/train.py
```

This writes model weights and logs under `training/` and plots under `analysis/plots/`.

Run the main diagnostics:

```bash
python analysis/scripts/analyze_latent.py
python analysis/scripts/inspect_preds.py
python analysis/scripts/investigate_surface_signal.py
python analysis/scripts/manifold_walk.py
python analysis/scripts/benchmark_inference.py
python analysis/scripts/backtest.py
```

Useful outputs include latent-space plots, prediction-error diagnostics, surface interpolation figures, inference benchmarks, and strategy comparison plots. The paper figures referenced from the writeup are stored in `writeup/figures/`.

## Notes

- Default training uses CPU or CUDA automatically, depending on what PyTorch detects.
- The train/validation split is implemented in `src/dataset.py`: training is 2010-2021, validation is 2022, and post-2022 data is treated as test data.
- Generated outputs are not required to import the package, but the analysis scripts that load `training/models/deeponet.pth` require a prior training run.
