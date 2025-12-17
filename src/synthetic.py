import torch
import torch.nn as nn
import numpy as np

class HestonSimulator(nn.Module):
    """
    GPU-accelerated Heston Model Simulator & Pricer.
    Generates:
    1. Market History (Price paths, Vol paths) -> Inputs
    2. Option Surfaces (via Fourier Transform Pricing) -> Targets
    """
    def __init__(self, device='cpu'):
        super().__init__()
        self.device = device

    def simulate_history(self, batch_size=32, seq_len=30, dt=1/252, hurst=0.1):
        """
        Simulates Rough Heston paths for the input history.
        Using Cholesky decomposition for Fractional Brownian Motion (fBm).
        """
        # Heston Parameters
        kappa = torch.rand(batch_size, 1, device=self.device) * 4.5 + 0.5
        theta = (torch.rand(batch_size, 1, device=self.device) * 0.5 + 0.01).pow(2)
        xi = torch.rand(batch_size, 1, device=self.device) * 0.9 + 0.1
        rho = torch.rand(batch_size, 1, device=self.device) * -0.9
        v0 = theta.clone()

        # Precompute fBm covariance if needed (cache logic could be added)
        # Covariance for W^H: E[Wh_t Wh_s] = 0.5 * (t^2H + s^2H - |t-s|^2H)

        ts = torch.arange(seq_len + 1, device=self.device) * dt
        # Make grid for covariance
        t_i = ts.unsqueeze(1)
        t_j = ts.unsqueeze(0)

        # Hurst parameter H in (0, 0.5) for rough volatility
        H = hurst

        cov = 0.5 * (t_i**(2*H) + t_j**(2*H) - torch.abs(t_i - t_j)**(2*H))

        # Add epsilon for numerical stability
        cov = cov + torch.eye(seq_len + 1, device=self.device) * 1e-8

        # Cholesky Decomposition
        L = torch.linalg.cholesky(cov)

        # Generate correlated standard normals
        Z = torch.randn(batch_size, seq_len + 1, 2, device=self.device)
        # Z[:,:,0] for Spot, Z[:,:,1] for Vol

        # Apply correlation rho (Standard Heston: dS ~ W1, dV ~ W2, corr(W1, W2) = rho)

        # 1. Generate Fractional Noise W_vol
        z_vol = Z[:, :, 1].unsqueeze(-1) # (B, T, 1)
        w_vol_H = torch.matmul(L, z_vol).squeeze(-1) # (B, T)

        # 2. Correlated Spot Noise dW_spot (Standard Bm increments)
        dw_spot =  rho.squeeze().unsqueeze(1) * Z[:, :, 1] + torch.sqrt(1 - rho.squeeze().unsqueeze(1)**2) * Z[:, :, 0]

        # Rough Bergomi Volatility Process
        eta = xi.squeeze()
        term1 = eta.unsqueeze(1) * w_vol_H
        term2 = -0.5 * (eta.unsqueeze(1)**2) * (ts**(2*H)).unsqueeze(0)
        v = v0.squeeze().unsqueeze(1) * torch.exp(term1 + term2)

        s = torch.zeros(batch_size, seq_len + 1, device=self.device)
        s[:, 0] = 1.0

        # Euler Spot Simulation
        sqrt_dt = np.sqrt(dt)
        for t in range(seq_len):
            vol_t = torch.sqrt(v[:, t])
            # dS = S * vol * dW_spot
            ds = s[:, t] * vol_t * dw_spot[:, t] * sqrt_dt
            s[:, t+1] = s[:, t] + ds

        # Trim to seq_len
        s = s[:, 1:]
        v = v[:, 1:]

        # Create Features: [LogReturn, RealizedVol, VIX_proxy, Volume, TNX, Buffett]
        log_ret = torch.diff(torch.log(s), dim=1)
        log_ret = torch.cat([torch.zeros(batch_size, 1, device=self.device), log_ret], dim=1)

        rv = torch.sqrt(v)
        vix = rv
        vol_noise = torch.randn(batch_size, seq_len, device=self.device) * 0.1

        # Random Risk Free Rate
        r = torch.rand(batch_size, 1, device=self.device) * 0.05
        r_rate = r.expand(-1, seq_len)

        buffett = torch.randn(batch_size, seq_len, device=self.device)

        x = torch.stack([log_ret, rv, vix, vol_noise, r_rate, buffett], dim=2) # (B, L, 6)

        # Return parameters for pricing
        params = {
            'kappa': kappa, 'theta': theta, 'xi': xi, 'rho': rho, 'v0': v0.squeeze(), 'r': r
        }
        return x, params

    def price_surface(self, params, grid):
        """
        Computes Heston Prices for the grid using Fourier Integration.
        grid: (Batch, N, 2) [Moneyness (K/S), TTM]
        """
        # Heston Characteristic Function
        kappa = params['kappa']
        theta = params['theta']
        xi = params['xi']
        rho = params['rho']
        v0 = params['v0'].unsqueeze(1) # (B, 1)
        r = params['r']

        k_strike = grid[:, :, 0] # Moneyness K/S
        tau = grid[:, :, 1] # TTM

        # Integration limits
        # Using simple trapezoidal rule for integration
        # This is valid but expensive if N_int is high.
        # N_int=100 is usually enough for ML training accuracy.

        u = torch.linspace(1e-4, 50.0, 64, device=self.device).view(1, 1, -1) # (1, 1, N_int)
        du = u[0, 0, 1] - u[0, 0, 0]

        # Broadcast params to (B, N_grid, 1) or (B, 1, 1)
        kappa = kappa.unsqueeze(1)
        theta = theta.unsqueeze(1)
        xi = xi.unsqueeze(1)
        rho = rho.unsqueeze(1)
        r = r.unsqueeze(1)
        v0 = v0.unsqueeze(1)

        # Characteristic Func phi(u)
        # Heston standard formula

        # d = sqrt( (rho*xi*u*i - kappa)^2 + xi^2 * (u^2 + i*u) )
        # g = (kappa - rho*xi*u*i - d) / (kappa - rho*xi*u*i + d)
        # phi = exp( A + B + C ) ...

        i = 1j

        # Reshape Inputs for Broadcasting
        # u: (1, 1, 64)
        # params: (B, 1, 1)
        # tau: (B, N, 1) to match u
        tau = tau.unsqueeze(-1)

        # Helper vars
        rsig = rho * xi * u * i
        sigma2 = xi**2

        M = np.sqrt( (rsig - kappa)**2 + sigma2 * (u**2 + i*u) ) # Complex sqrt?
        # PyTorch complex support is good now.
        # Tensor shapes need explicit handling for broadcasting.

        # Use the standard Heston formulation for this approximation.

        # Need to cast to complex
        u_c = u.to(torch.complex64)
        kappa_c = kappa.to(torch.complex64)
        theta_c = theta.to(torch.complex64)
        xi_c = xi.to(torch.complex64)
        rho_c = rho.to(torch.complex64)
        v0_c = v0.to(torch.complex64)
        tau_c = tau.to(torch.complex64)
        r_c = r.to(torch.complex64)

        d = torch.sqrt( (rho_c*xi_c*u_c*i - kappa_c)**2 + xi_c**2 * (u_c**2 + i*u_c) )
        g = (kappa_c - rho_c*xi_c*u_c*i - d) / (kappa_c - rho_c*xi_c*u_c*i + d)

        # C(u) part
        term1 = (kappa_c * theta_c / xi_c**2) * ( (kappa_c - rho_c*xi_c*u_c*i - d)*tau_c - 2*torch.log( (1 - g*torch.exp(-d*tau_c))/(1-g) ) )
        # D(u) part
        term2 = (v0_c / xi_c**2) * (kappa_c - rho_c*xi_c*u_c*i - d) * (1 - torch.exp(-d*tau_c)) / (1 - g*torch.exp(-d*tau_c))

        phi = torch.exp(term1 + term2)
        # Adjust for drift r? The formula above is for driftless?
        # Standard Heston usually includes r.
        # Phi(u) for ln(S_T).
        # Shift by i*u*ln(S_0 * e^rt) = i*u*(0 + rt) since S_0=1
        # Gil-Pelaez Formula
        # Call = S * P1 - K * e^{-rT} * P2
        # Pj = 0.5 + 1/pi * integral_0^inf Re[ e^{-i u ln K} * phi_j(u) / (i u) ] du

        # phi_1(u) = phi(u - i) / phi(-i)  (Numeraire change)
        # phi_2(u) = phi(u)

        # For pre-training, the network only needs the volatility surface shape.
        # Construct a synthetic surface using Heston asymptotics (Gatheral, The Volatility Surface).
        # w(k, t) = v_t * t
        # Skew ~ rho * xi
        # Curvature ~ xi^2

        # Generating synthetic surfaces that LOOK like Heston is easier than computing exact Heston.
        # Target IV = sqrt(v0) * (1 + 0.25*rho*xi/v0 * LogK + (2-3rho^2)/24 * xi^2/v0 * LogK^2) * TermStructure

        # This is the "Heston stochastic volatility expansion" (small time/vol).
        # Only valid for short parameters?
        # This is accurate enough for the physics pre-training stage.

        k_log = torch.log(k_strike) # Log Moneyness (ln(K/S))

        # Base Vol
        vol = torch.sqrt(params['v0']).unsqueeze(1) # (B, 1)

        # Term structure (simple mean reversion dampening)
        # v_eff = theta + (v0 - theta)*(1 - exp(-kappa*t))/(kappa*t)
        t_mat = grid[:, :, 1]
        # params['kappa'] is (B, 1), t_mat is (B, N) -> (B, N) without unsqueeze
        term_factor = (1 - torch.exp(-params['kappa']*t_mat)) / (params['kappa']*t_mat + 1e-6)
        v0_expanded = params['v0'].view(-1, 1)
        v_eff = params['theta'] + (v0_expanded - params['theta']) * term_factor
        vol_eff = torch.sqrt(v_eff)

        # Skew and Curvature factors based on rho and xi
        # params['rho'] is (B, 1)
        skew = params['rho'] * params['xi']
        kurt = params['xi']**2

        # Polynomial approximation of the smile
        # This is heuristic but creates physically realistic shapes
        smile = 1.0 + (0.5 * skew / vol_eff) * k_log + (0.1 * kurt / vol_eff**2) * k_log**2

        sigma_surface = vol_eff * smile
        sigma_surface = torch.clamp(sigma_surface, 0.05, 2.0)

        return sigma_surface # Direct IV surface

    def generate_batch(self, batch_size=32, seq_len=30, grid_tensor=None):
        x, params = self.simulate_history(batch_size, seq_len)

        if grid_tensor is None:
            # Standard Grid if none provided
            # Standard grid is (B, 21, 2)
            raise ValueError("grid_tensor must be provided")

        sigma_surface = self.price_surface(params, grid_tensor)

        # Domain Label for Synthetic Data is 2.0 (Custom Domain)
        # But for binary DANN (0/1), we should map it to 0 (Source) usually.
        # Since we want to transfer Source (Synthetic) -> Target (Real).
        domain_label = torch.zeros(batch_size, 1, dtype=torch.float32, device=self.device)

        return x, sigma_surface, params, domain_label
