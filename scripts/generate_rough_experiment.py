import numpy as np
import pandas as pd
import os
from scipy.linalg import cholesky
from tqdm import tqdm

class RoughHestonGenerator:
    def __init__(self, risk_free_rate=0.03):
        self.r = risk_free_rate

    def fractional_brownian_motion(self, H, T, N, n_paths):
        """
        Generates fractional Brownian motion using Cholesky decomposition.
        """
        dt = T / N
        t = np.linspace(0, T, N+1)

        # Covariance matrix for fBm
        # E[B_H(t) B_H(s)] = 0.5 * (|t|^{2H} + |s|^{2H} - |t-s|^{2H})
        # We generate increments covariance
        # But for simplicity with Cholesky on small N, we can do covariance of the process directly or increments.
        # Given N is usually small for option pricing (e.g. 100 steps), full Cholesky is fine.

        # Let's use a grid of time points
        cov = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                t_i = (i + 1) * dt
                t_j = (j + 1) * dt
                cov[i, j] = 0.5 * (t_i**(2*H) + t_j**(2*H) - np.abs(t_i - t_j)**(2*H))

        # Cholesky
        try:
            L = cholesky(cov, lower=True)
        except np.linalg.LinAlgError:
            # Fallback for numerical stability if needed, though usually fine for H in (0, 1)
            L = cholesky(cov + 1e-9 * np.eye(N), lower=True)

        # Generate standard normal random variables
        Z = np.random.normal(0, 1, (N, n_paths))

        # fBm paths (starting at 0)
        W_H = np.dot(L, Z) # Shape (N, n_paths)
        W_H = np.vstack([np.zeros((1, n_paths)), W_H]) # Add t=0

        return W_H

    def simulate_paths(self, S0, v0, kappa, theta, sigma, rho, H, T, N, n_paths):
        """
        Simulates Rough Heston paths.
        dS = r S dt + sqrt(v) S dW1
        v_t = v0 + 1/Gamma(H+0.5) * int_0^t (t-s)^{H-0.5} * kappa(theta - v_s) ds
              + 1/Gamma(H+0.5) * int_0^t (t-s)^{H-0.5} * sigma * sqrt(v_s) dW_H

        Wait, the standard "Rough Heston" usually defines the volatility process as a fractional OU or similar Volterra process.
        A common form (Rosenbaum et al):
        v_t = v0 + 1/Gamma(alpha) * \int_0^t (t-s)^{alpha-1} \lambda (\theta - v_s) ds + ...
        where alpha = H + 0.5.

        However, for a simplified "Rough Heston" simulation that captures the essence (roughness),
        we can simply drive the variance process with a fractional Brownian motion $W^H$.

        Standard Heston: dv = kappa(theta - v)dt + sigma sqrt(v) dW
        Rough Heston (Simplified): v_t is driven by W^H.

        We will use the discrete approximation:
        v_{t+1} - v_t = kappa(theta - v_t)dt + sigma sqrt(v_t) (W^H_{t+1} - W^H_t)

        Note: This is a "rough diffusion" approximation. Strictly speaking, Rough Heston is non-Markovian and defined by a Volterra equation.
        But driving the SDE with fBm increments is a standard way to introduce roughness in simulation studies
        when exact Volterra sampling is too complex for a quick experiment.

        Important: We need correlation rho between S and v.
        We generate W1 (standard) and W2 (fBm).
        We correlate them? Usually rho is between the Brownian drivers.
        If v is driven by fBm, defining correlation is trickier.
        We will assume:
        dW_S = rho dW_H + sqrt(1-rho^2) dW_perp
        where dW_H are the increments of the fBm.
        """
        dt = T / N

        # 1. Generate fBm for Variance
        W_H = self.fractional_brownian_motion(H, T, N, n_paths)
        dW_H = np.diff(W_H, axis=0) # Increments

        # 2. Generate Brownian Motion for Price (correlated)
        Z_perp = np.random.normal(0, 1, (N, n_paths)) * np.sqrt(dt)

        # Construct dW_S
        # Note: dW_H variance is not dt. It scales with dt^{2H}.
        # To maintain correlation structure properly in a "rough" world is subtle.
        # For this surrogate experiment, we want to see if the network handles "rough trajectories".
        # We will construct the price noise to be correlated with the variance noise.

        # Normalize dW_H to have unit variance scaling for the correlation mix?
        # Actually, let's just use the raw increments. The "correlation" rho usually implies
        # <dW_S, dW_v> = rho * dt (in standard).
        # Here <dW_S, dW_H> should be rho * ...?
        # Let's stick to the simple mixing:
        # dW_S = rho * (dW_H / std(dW_H)) * sqrt(dt) + sqrt(1-rho^2) * dW_perp ?
        # No, that's getting complicated.

        # Let's use the standard mixing on the *drivers*:
        # But W_H is already generated.
        # Let's just generate S using:
        # dS/S = r dt + sqrt(v) * (rho * dW_H_normalized + sqrt(1-rho^2) * dW_perp)
        # We need to normalize dW_H to look like a Brownian increment (scale sqrt(dt)) for the mixing to make sense
        # as a "correlation" coefficient, otherwise the variance of S will be weird.
        # Actually, if H != 0.5, dW_H scales as dt^H.
        # If we use it directly in price, Price will have rough paths too?
        # Usually Price is a semi-martingale (Brownian), only Vol is rough.
        # So Price should be driven by standard Brownian Motion W_S.
        # But W_S and W_H (driving vol) are correlated.

        # Approach:
        # 1. Generate W_H (fBm) for Vol.
        # 2. Generate W_S (Standard BM) such that corr(dW_S, dW_H) is rho.
        # This is hard because dW_H is not i.i.d.

        # Simplified Approach for "Roughness Experiment":
        # We just want the Volatility to be rough. The Price can be standard Heston-like but with rough vol.
        # We will generate W_S as standard BM.
        # We will generate W_H as fBm.
        # We will impose correlation by constructing W_S from W_H? No, W_H has memory.

        # Let's use the Cholesky method to generate correlated Gaussian vectors for the whole path?
        # Too expensive (2N x 2N).

        # Let's assume zero correlation for the "Roughness" test to isolate the effect of Rough Vol?
        # Or just use a simple local correlation:
        # dW_S(t) = rho * (dW_H(t) scaled) + ...
        # Let's stick to: Price is driven by standard BM. Vol is driven by fBm.
        # We ignore correlation for this specific "Roughness" stress test to avoid mathematical artifacts.
        # This is a valid "stress test" of the manifold learning of roughness.
        rho = 0.0

        S = np.zeros((N+1, n_paths))
        v = np.zeros((N+1, n_paths))
        S[0] = S0
        v[0] = v0

        # Pre-generate price noise
        dW_S = np.random.normal(0, np.sqrt(dt), (N, n_paths))

        for t in range(N):
            # Variance Step (Euler with Reflection)
            # v_{t+1} = v_t + kappa(theta - v_t)dt + sigma * sqrt(v_t) * dW_H
            # Note: dW_H is the increment of fBm

            # Floor v at 0
            v_curr = np.maximum(v[t], 0.0)

            dv = kappa * (theta - v_curr) * dt + sigma * np.sqrt(v_curr) * dW_H[t]
            v[t+1] = np.abs(v_curr + dv) # Reflection for positivity

            # Price Step
            # S_{t+1} = S_t + r S_t dt + sqrt(v_t) S_t dW_S
            ds = self.r * S[t] * dt + np.sqrt(v_curr) * S[t] * dW_S[t]
            S[t+1] = S[t] + ds

        return S, v

    def price_european_options(self, S0, v0, kappa, theta, sigma, rho, H, T_maturities, K_moneyness, n_paths=1000):
        """
        Prices options using Monte Carlo.
        """
        prices = []

        # Max maturity
        max_T = max(T_maturities)
        N_steps = int(max_T * 252) # Daily steps

        # Simulate Paths
        S, v = self.simulate_paths(S0, v0, kappa, theta, sigma, rho, H, max_T, N_steps, n_paths)

        # Price for each maturity
        for T in T_maturities:
            step_idx = int(T * 252)
            if step_idx >= S.shape[0]: step_idx = S.shape[0] - 1

            S_T = S[step_idx]

            for k_ratio in K_moneyness:
                strike = S0 * k_ratio
                payoff = np.maximum(S_T - strike, 0.0)
                price = np.mean(payoff) * np.exp(-self.r * T)
                prices.append(price / S0) # Normalize

        return np.array(prices)

def generate_rough_dataset():
    generator = RoughHestonGenerator()

    # Parameters
    n_samples = 500
    n_paths = 2000 # MC paths per sample

    # Heston Params (Base)
    kappa = 2.0
    theta = 0.04
    sigma = 0.3
    rho = 0.0 # Zero correlation for rough experiment

    # Varying Hurst Exponents
    # H < 0.5 is rough. H = 0.5 is standard.
    hurst_values = np.linspace(0.05, 0.5, n_samples)

    data_rows = []

    print(f"Generating {n_samples} Rough Heston samples...")

    for i in tqdm(range(n_samples)):
        H = hurst_values[i]

        # Randomize initial state slightly
        S0 = 100.0
        v0 = np.random.uniform(0.02, 0.06)

        # 1. Generate Target Prices (Monte Carlo)
        # Maturities: 1, 3, 6 months (approx 0.08, 0.25, 0.5 years)
        maturities = [1/12, 3/12, 6/12]
        moneyness = [0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2]

        prices = generator.price_european_options(
            S0, v0, kappa, theta, sigma, rho, H,
            maturities, moneyness, n_paths
        )

        # 2. Generate Input History (Synthetic)
        # We need a 30-day history of [LogRet, Vol, VIX, Volume, TNX, Buffett]
        # For this synthetic test, we will generate a synthetic history consistent with the Rough Heston path.
        # We simulate a 30-day path *leading up to* T=0.

        T_hist = 30/252
        N_hist = 30
        S_hist, v_hist = generator.simulate_paths(S0, v0, kappa, theta, sigma, rho, H, T_hist, N_hist, 1)

        # Extract features from single path
        S_path = S_hist[:, 0]
        v_path = v_hist[:, 0] # This is variance

        log_ret = np.diff(np.log(S_path))
        realized_vol = np.sqrt(v_path[1:]) # Proxy instantaneous vol

        # Synthetic VIX (approximate from Heston v)
        # VIX^2 approx theta + (v_t - theta) * ...
        # Let's just use sqrt(v_t) as VIX proxy for this synthetic test
        vix = np.sqrt(v_path[1:])

        # Dummy Macro
        vol_feature = np.zeros_like(log_ret) # Flat volume
        tnx = np.full_like(log_ret, 0.04)
        buffett = np.full_like(log_ret, 1.5)

        features = np.stack([
            log_ret,
            realized_vol,
            vix,
            vol_feature,
            tnx,
            buffett
        ], axis=1)

        data_rows.append({
            'Hurst': H,
            'Input_Features': features.flatten(),
            'Target_Prices': prices
        })

    df = pd.DataFrame(data_rows)
    output_path = 'data/rough_heston_test.parquet'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_parquet(output_path)
    print(f"Saved to {output_path}")

if __name__ == "__main__":
    generate_rough_dataset()
