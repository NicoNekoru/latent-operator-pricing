import torch
import numpy as np
from scipy.stats import norm

def calculate_metrics(y_pred, y_true):
    # MAPE: Mean Absolute Percentage Error
    # Mask out very small values to avoid division by zero
    mask = y_true > 1e-4
    if mask.sum() == 0:
        return 0.0, 0.0

    diff = torch.abs(y_pred[mask] - y_true[mask])
    mape = torch.mean(diff / y_true[mask]) * 100.0

    # Dollar Error (assuming Index ~ 4000)
    dollar_err = torch.mean(diff) * 4000.0

    return mape.item(), dollar_err.item()

def black_scholes_price(S, K, T, r, sigma, option_type='call'):
    # Vectorized BSM pricing
    T = np.maximum(T, 1e-9)
    sigma = np.maximum(sigma, 1e-9)

    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == 'call': price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else: price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

    return price

def calculate_implied_volatility(price, S, K, T, r, option_type='call'):
    """
    Calculate Implied Volatility using Newton-Raphson method.
    """
    MAX_ITER = 100
    PRECISION = 1e-5

    sigma = 0.5 # Initial guess
    for i in range(MAX_ITER):
        P = black_scholes_price(S, K, T, r, sigma, option_type)
        diff = price - P

        if abs(diff) < PRECISION:
            return sigma

        # Vega
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        vega = S * norm.pdf(d1) * np.sqrt(T)

        if vega < 1e-8:
            break

        sigma = sigma + diff / vega

    return sigma
