import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.models import ManifoldAutoencoder

def fractional_brownian_motion(n_steps, T, H):
    """
    Generate fBm using the Davies-Harte method.
    """
    # Simple Cholesky implementation for small n_steps (since we only need 30 days)
    # For larger simulations, use fbm library or Davies-Harte
    t = np.linspace(0, T, n_steps)
    cov = np.zeros((n_steps, n_steps))
    for i in range(n_steps):
        for j in range(n_steps):
            cov[i, j] = 0.5 * (abs(t[i]**(2*H) + t[j]**(2*H) - abs(t[i]-t[j])**(2*H)))

    L = np.linalg.cholesky(cov + 1e-6 * np.eye(n_steps))
    Z = np.random.normal(0, 1, n_steps)
    fBm = L @ Z
    return t, fBm

def simulate_rough_heston(n_steps=30, dt=1/252, H=0.1):
    r"""
    Simulate Rough Heston paths.
    v_t = v_0 + \frac{1}{\Gamma(H+0.5)} \int_0^t (t-s)^{H-0.5} \lambda (\theta - v_s) ds + \xi \int_0^t (t-s)^{H-0.5} dW_s

    Approximation: Use fBm for the noise term.
    """
    # Parameters
    v0 = 0.04
    theta = 0.04
    kappa = 2.0
    xi = 0.3

    # Generate fBm for volatility
    _, W_H = fractional_brownian_motion(n_steps, n_steps*dt, H)

    # Euler-Maruyama for Rough Vol (Simplified)
    # Note: True Rough Heston requires fractional calculus.
    # Here we approximate by driving the vol process with fBm increments.
    v = np.zeros(n_steps)
    v[0] = v0

    for t in range(1, n_steps):
        # Volatility cannot be negative
        drift = kappa * (theta - v[t-1]) * dt
        # fBm increment
        diffusion = xi * np.sqrt(v[t-1]) * (W_H[t] - W_H[t-1])
        v[t] = abs(v[t-1] + drift + diffusion)

    return v

def main():
    print("Generating Rough Heston Data (H=0.1)...")

    # 1. Generate Synthetic Data
    # We need inputs of shape (Batch, 30, 6)
    # Features: [LogRet, RealizedVol, VIX, LogVol, TNX, Buffett]
    # We will simulate 'RealizedVol' using Rough Heston and keep others constant/noise

    n_samples = 100
    seq_len = 30
    input_dim = 6

    inputs = torch.zeros(n_samples, seq_len, input_dim)

    for i in range(n_samples):
        vol_path = simulate_rough_heston(n_steps=seq_len, H=0.1)
        # Feature 1: Realized Vol (The rough path)
        inputs[i, :, 1] = torch.tensor(vol_path).float()
        # Feature 2: VIX (Proxy as scaled vol)
        inputs[i, :, 2] = torch.tensor(vol_path * 100 + 2.0).float() # Simple proxy
        # Other features: Gaussian noise for now
        inputs[i, :, 0] = torch.randn(seq_len) * 0.01 # Returns
        inputs[i, :, 3] = torch.randn(seq_len) + 10 # LogVolume
        inputs[i, :, 4] = 4.0 # TNX
        inputs[i, :, 5] = 1.5 # Buffett

    # 2. Load Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ManifoldAutoencoder(input_dim=6, latent_dim=3).to(device)

    # Load weights (assuming standard path)
    model_path = 'models/neural_operator.pth'
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded model from {model_path}")
    else:
        print(f"Model not found at {model_path}. Using random weights for demo.")

    model.eval()

    # 3. Evaluate
    with torch.no_grad():
        inputs = inputs.to(device)
        prices, z = model(inputs)

    # 4. Analyze Latent Space
    # Check if Rough Heston maps to a specific region?
    z_np = z.cpu().numpy()

    print(f"Latent Mean: {z_np.mean(axis=0)}")
    print(f"Latent Std: {z_np.std(axis=0)}")

    # Save plot
    plt.figure(figsize=(10, 6))
    plt.scatter(z_np[:, 0], z_np[:, 1], c='red', label='Rough Heston (H=0.1)')
    plt.title("Latent Space Projection of Rough Volatility")
    plt.xlabel("Z1")
    plt.ylabel("Z2")
    plt.legend()
    plt.grid(True)
    os.makedirs('plots', exist_ok=True)
    plt.savefig('plots/rough_heston_projection.png')
    print("Saved plot to plots/rough_heston_projection.png")

if __name__ == "__main__":
    main()
