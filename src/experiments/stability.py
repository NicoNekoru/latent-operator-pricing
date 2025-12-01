import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.models import ManifoldAutoencoder

def heston_price(S, K, T, v, theta, kappa, xi, rho):
    """
    Dummy Heston pricer for demonstration.
    In a real scenario, this would use QuantLib or a Fourier pricer.
    Here we use a simple Black-Scholes approximation with vol = sqrt(v)
    just to simulate the "Forward Operator" F(z).
    """
    # This is a placeholder. For the stability experiment, we just need
    # a function F(z) -> Prices.
    # We will assume the "Neural Decoder" IS the forward operator for this test,
    # or we can use a simplified proxy.

    # To be rigorous, we should use the SAME forward operator used to generate training data.
    # But since we don't have QuantLib bindings easily here, we will use the
    # Neural Decoder as the "Ground Truth Physics" for this specific stability test.
    # i.e. We test: Is E(D(z) + noise) stable? vs. Is LeastSquares(D(z) + noise) stable?
    pass

def main():
    print("Running Calibration Stability Experiment...")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ManifoldAutoencoder(input_dim=6, latent_dim=3).to(device)

    # Load weights
    model_path = 'models/neural_operator.pth'
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        print("Model not found. Using random weights.")

    model.eval()

    # 1. Generate a "Ground Truth" Latent State
    z_true = torch.tensor([[0.5, -0.2, 0.1]]).to(device) # Arbitrary state

    # 2. Generate "True" Prices using Decoder (acting as Forward Operator)
    with torch.no_grad():
        prices_true = model.decoder(z_true)

    # 3. Add Noise and Measure Stability
    noise_levels = np.linspace(0, 0.1, 20)
    neural_errors = []
    ls_errors = []

    print(f"Testing stability for noise levels: {noise_levels}")

    for sigma in noise_levels:
        # Add noise to prices
        noise = torch.randn_like(prices_true) * sigma
        prices_noisy = prices_true + noise

        # Method A: Neural Inverse (Encoder)
        # We need to map Prices -> Latent.
        # But the Encoder takes (30, 6) history, not Prices.
        # This reveals a structural asymmetry: The Encoder is History -> Z, Decoder is Z -> Prices.
        # We don't have a direct Price -> Z map in this architecture!

        # CRITICAL REALIZATION: The "Inverse Problem" described in the paper is History -> Z.
        # But "Calibration" is usually Price -> Z.
        # The paper claims we solve the inverse problem of mapping *market states* to regimes.
        # To test stability, we should perturb the *Input History* (x) and see how Z changes.

        # Let's perturb the INPUTS (History) instead of Prices.

        # Generate dummy history
        x_true = torch.randn(1, 180).to(device) # Flattened (1, 30*6)

        # Get Z_true
        with torch.no_grad():
            z_true_enc = model.encoder(x_true)

        # Perturb Input
        noise_x = torch.randn_like(x_true) * sigma
        x_noisy = x_true + noise_x

        # Get Z_noisy
        with torch.no_grad():
            z_noisy = model.encoder(x_noisy)

        # Measure deviation in Z
        dist = torch.norm(z_noisy - z_true_enc).item()
        neural_errors.append(dist)

        # Method B: Standard "Calibration" (Least Squares on Decoder)
        # We try to find z such that Decoder(z) approx Decoder(z_true_enc) + noise_in_price_space?
        # No, we want to compare Input Stability.
        # Standard calibration doesn't map History -> Params. It maps Prices -> Params.

        # Alternative comparison:
        # Compare Neural Encoder stability vs. a "Linear Regression" baseline?
        # Or simply show that the Neural map is Lipschitz continuous?

        # Let's stick to the "Lipschitz" test.
        # We plot Output Error (in Z) vs Input Noise (in X).
        # A stable map should be linear (Lipschitz). An unstable map would explode.

    plt.figure(figsize=(8, 5))
    plt.plot(noise_levels, neural_errors, marker='o', label='Manifold Autoencoder')
    plt.xlabel("Input Noise $\sigma$")
    plt.ylabel("Latent State Deviation $||z - z_{true}||$")
    plt.title("Stability of the Inverse Map")
    plt.legend()
    plt.grid(True)
    plt.savefig('plots/stability_test.png')
    print("Saved plot to plots/stability_test.png")

if __name__ == "__main__":
    main()
