import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import least_squares
import QuantLib as ql
from tqdm import tqdm
import os

from src.models import DeepONet
from src.dataset import OptionDataset
from src.data_loader import HestonGenerator

def calibrate_heston(target_prices, spot, r, maturities, moneyness):
    """
    Calibrates Heston parameters (v0, kappa, theta, sigma, rho) to target prices.
    """
    generator = HestonGenerator(risk_free_rate=r)

    def objective(params):
        v0, kappa, theta, sigma, rho = params
        # Constraints
        if v0 < 0 or kappa < 0 or theta < 0 or sigma < 0 or abs(rho) > 1:
            return np.ones_like(target_prices) * 1e6

        generator.setup_engine(spot, v0, kappa, theta, sigma, rho)
        try:
            model_prices = generator.generate(spot)
            # Filter to match target length if needed, but here we assume fixed grid
            # generator.generate returns fixed grid defined in class
            # We assume target_prices matches that grid
            return model_prices - target_prices
        except:
            return np.ones_like(target_prices) * 1e6

    # Initial guess
    x0 = [0.04, 2.0, 0.04, 0.3, -0.5]
    bounds = ([0, 0, 0, 0, -1], [1.0, 10.0, 1.0, 2.0, 1])

    res = least_squares(objective, x0, bounds=bounds, method='trf', max_nfev=50)
    return res.x

def stability_analysis():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load Data
    dataset_path = 'data/processed_dataset.parquet'
    dataset = OptionDataset(dataset_path, mode='test')

    # Select subset
    n_samples = 20 # Keep small for calibration speed
    indices = np.random.choice(len(dataset), n_samples, replace=False)

    # Load Model
    model = DeepONet(latent_dim=16).to(device)
    try:
        model.load_state_dict(torch.load('models/deeponet.pth', map_location=device))
    except FileNotFoundError:
        print("Model not found.")
        return
    model.eval()

    # Perturbation Params
    n_perturbations = 20
    noise_level = 0.05 # 5% noise

    deeponet_variances = []
    calibration_variances = []

    print(f"Running Stability Analysis on {n_samples} samples with {n_perturbations} perturbations...")

    for idx in tqdm(indices):
        x, y = dataset[idx] # x: (30, 6), y: (21,)

        # --- DeepONet Stability ---
        # Perturb Input History
        latents = []
        for _ in range(n_perturbations):
            noise = torch.randn_like(x) * noise_level * x.std(dim=0) # Relative to feature scale
            x_perturbed = x + noise

            with torch.no_grad():
                # Get latent z
                # x needs batch dim
                x_batch = x_perturbed.unsqueeze(0).to(device)

                # We need a dummy grid to call forward, or just use branch_cnn/mlp manually
                # But forward is safer.
                # Create a minimal grid (1 point) just to satisfy forward
                dummy_grid = torch.zeros((1, 1, 2), device=device)
                _, z = model(x_batch, dummy_grid)

                latents.append(z.cpu().numpy().flatten())

        latents = np.array(latents)
        # Calculate variance of latent vector (mean variance across dimensions)
        # Normalize by mean norm of z to make it relative?
        # Or just raw variance? Let's use Coefficient of Variation proxy: std / mean_norm
        z_mean_norm = np.linalg.norm(latents.mean(axis=0))
        z_std_norm = np.mean(np.std(latents, axis=0))
        deeponet_variances.append(z_std_norm / (z_mean_norm + 1e-6))

        # --- Calibration Stability ---
        # Perturb Target Prices
        # We need spot price for calibration.
        # In dataset, we don't store spot explicitly in y, but we normalized by spot.
        # Let's assume Spot=100 for calibration since prices are normalized.
        spot = 100.0
        y_np = y.numpy()

        params_list = []

        # Base calibration (to get "true" params for this sample)
        # Skip base, just perturb

        for _ in range(n_perturbations):
            noise = np.random.normal(0, noise_level * np.mean(y_np), size=y_np.shape)
            y_perturbed = y_np + noise
            y_perturbed = np.maximum(y_perturbed, 0.0) # Ensure positive

            # Calibrate
            # Note: HestonGenerator hardcodes maturities/moneyness.
            # We must match them.
            # dataset.py doesn't export them, but they are standard:
            # Mats: 1, 3, 6 months. Moneyness: 0.8-1.2
            mats = [1/12, 3/12, 6/12]
            mon = [0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2]

            params = calibrate_heston(y_perturbed, spot, 0.03, mats, mon)
            params_list.append(params)

        params_list = np.array(params_list)
        # Calculate variance of parameters
        # Normalize by mean param value
        p_mean = np.abs(params_list.mean(axis=0))
        p_std = params_list.std(axis=0)
        # Average CV across parameters
        calibration_variances.append(np.mean(p_std / (p_mean + 1e-6)))

    # Plotting
    data = pd.DataFrame({
        'DeepONet (Latent)': deeponet_variances,
        'Calibration (Params)': calibration_variances
    })

    plt.figure(figsize=(8, 6))
    sns.boxplot(data=data)
    plt.title('Inverse Stability: Latent vs. Calibration Variance (CV)')
    plt.ylabel('Coefficient of Variation (Noise Sensitivity)')
    plt.yscale('log') # Log scale because calibration might be very unstable

    output_plot = 'writeup/figures/stability_boxplot.png'
    os.makedirs(os.path.dirname(output_plot), exist_ok=True)
    plt.savefig(output_plot)
    print(f"Saved stability plot to {output_plot}")

    print(f"DeepONet Mean CV: {np.mean(deeponet_variances):.4f}")
    print(f"Calibration Mean CV: {np.mean(calibration_variances):.4f}")

if __name__ == "__main__":
    stability_analysis()
