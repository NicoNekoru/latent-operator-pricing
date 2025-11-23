import yfinance as yf
import pandas as pd
import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data_loader import HestonSimulator
import matplotlib.pyplot as plt

def validate_prices():
    print("Fetching SPY data...")
    spy = yf.Ticker("SPY")

    # Get current spot price
    history = spy.history(period="1mo")
    current_spot = history['Close'].iloc[-1]
    print(f"Current SPY Spot: {current_spot:.2f}")

    # Calculate realized volatility
    log_returns = np.log(history['Close'] / history['Close'].shift(1))
    realized_vol = log_returns.std() * np.sqrt(252)
    print(f"Realized Vol (1mo): {realized_vol:.4f}")

    # Get option chain for ~1 month out
    expirations = spy.options
    # Find expiration closest to 30 days
    target_days = 30
    best_date = expirations[0]
    min_diff = 999

    from datetime import datetime
    today = datetime.now()

    for date_str in expirations:
        exp_date = datetime.strptime(date_str, "%Y-%m-%d")
        days_diff = abs((exp_date - today).days - target_days)
        if days_diff < min_diff:
            min_diff = days_diff
            best_date = date_str

    print(f"Using expiration: {best_date}")

    opt_chain = spy.option_chain(best_date)
    calls = opt_chain.calls

    # Filter strikes around spot (Moneyness 0.8 to 1.2)
    calls = calls[(calls['strike'] >= current_spot * 0.8) & (calls['strike'] <= current_spot * 1.2)]

    # Prepare Heston Simulator
    # Heuristic parameters
    v0 = realized_vol ** 2
    kappa = 2.0
    theta = v0
    sigma = 0.3
    rho = -0.7

    sim = HestonSimulator()

    # Calculate Heston prices for these strikes
    heston_prices = []
    market_prices = []
    strikes = []

    print("Calculating Heston prices...")
    for idx, row in calls.iterrows():
        strike = row['strike']
        market_price = (row['bid'] + row['ask']) / 2

        # Heston price
        # We need to use the generate_surface logic but for specific strikes
        # Re-using sim.generate_surface is hard because it uses fixed grid.
        # Let's access the internal engine directly or modify the class.
        # For quick validation, I will instantiate the engine here directly using the same logic.

        import QuantLib as ql
        ql_date = ql.Date(today.day, today.month, today.year)
        ql.Settings.instance().evaluationDate = ql_date

        flat_ts = ql.YieldTermStructureHandle(ql.FlatForward(ql_date, 0.04, ql.Actual365Fixed())) # Assume 4% risk free
        dividend_ts = ql.YieldTermStructureHandle(ql.FlatForward(ql_date, 0.0, ql.Actual365Fixed()))
        spot_handle = ql.QuoteHandle(ql.SimpleQuote(current_spot))

        heston_process = ql.HestonProcess(flat_ts, dividend_ts, spot_handle, v0, kappa, theta, sigma, rho)
        heston_model = ql.HestonModel(heston_process)
        engine = ql.AnalyticHestonEngine(heston_model)

        exp_date_ql = ql.Date(int(best_date.split('-')[2]), int(best_date.split('-')[1]), int(best_date.split('-')[0]))

        payoff = ql.PlainVanillaPayoff(ql.Option.Call, strike)
        exercise = ql.EuropeanExercise(exp_date_ql)
        option = ql.VanillaOption(payoff, exercise)
        option.setPricingEngine(engine)

        try:
            h_price = option.NPV()
        except:
            h_price = 0

        heston_prices.append(h_price)
        market_prices.append(market_price)
        strikes.append(strike)

    # Create DataFrame
    df = pd.DataFrame({
        'Strike': strikes,
        'Market_Price': market_prices,
        'Heston_Price': heston_prices
    })

    df['Diff'] = df['Heston_Price'] - df['Market_Price']
    df['Moneyness'] = df['Strike'] / current_spot

    print("\nValidation Results (Sample):")
    print(df.iloc[::5]) # Print every 5th row

    # Check correlation
    corr = df['Market_Price'].corr(df['Heston_Price'])
    print(f"\nCorrelation between Market and Heston: {corr:.4f}")

    if corr > 0.95:
        print("SUCCESS: Heston model is highly correlated with market prices.")
    else:
        print("WARNING: Heston model deviation is high.")

if __name__ == "__main__":
    validate_prices()
