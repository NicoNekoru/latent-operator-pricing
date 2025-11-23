import yfinance as yf
import pandas as pd
import numpy as np
import QuantLib as ql
from datetime import date, timedelta
import os

class MarketScraper:
    def __init__(self, tickers=['^GSPC', '^NDX'], start_date='2010-01-01', end_date=None):
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.data = None

    def download_data(self):
        cache_path = 'data/raw_market_data.parquet'
        if os.path.exists(cache_path):
            print(f"Loading cached data from {cache_path}...")
            self.data = pd.read_parquet(cache_path)
            return self.data

        print(f"Downloading data for {self.tickers}...")
        data = yf.download(self.tickers, start=self.start_date, end=self.end_date, group_by='ticker')

        # Save to cache
        os.makedirs('data', exist_ok=True)
        # yfinance returns MultiIndex columns, which parquet handles, but let's be safe
        data.to_parquet(cache_path)

        self.data = data
        return data

    def process_data(self):
        if self.data is None:
            self.download_data()

        processed_dfs = []
        for ticker in self.tickers:
            df = self.data[ticker].copy()
            # Calculate Log Returns
            df['LogReturn'] = np.log(df['Close'] / df['Close'].shift(1))
            # Calculate Realized Volatility (21-day rolling std dev of log returns)
            df['RealizedVol'] = df['LogReturn'].rolling(window=21).std() * np.sqrt(252)

            df = df.dropna()
            df['Ticker'] = ticker
            processed_dfs.append(df)

        combined_df = pd.concat(processed_dfs)
        return combined_df

class HestonSimulator:
    def __init__(self, risk_free_rate=0.03):
        self.risk_free_rate = risk_free_rate
        self.day_count = ql.Actual365Fixed()
        self.calendar = ql.UnitedStates(ql.UnitedStates.NYSE)

    def generate_surface(self, spot_price, v0, kappa=2.0, theta=0.04, sigma=0.3, rho=-0.7):
        """
        Generates option prices for a grid of maturities and moneyness using the Heston model.
        """
        today = date.today()
        ql_date = ql.Date(today.day, today.month, today.year)
        ql.Settings.instance().evaluationDate = ql_date

        # Heston Process
        # v0: initial variance
        # kappa: mean reversion speed
        # theta: long-term variance
        # sigma: volatility of volatility
        # rho: correlation between spot and vol

        # Ensure parameters are within valid ranges
        v0 = max(v0, 1e-4)
        theta = max(theta, 1e-4)

        flat_ts = ql.YieldTermStructureHandle(ql.FlatForward(ql_date, self.risk_free_rate, self.day_count))
        dividend_ts = ql.YieldTermStructureHandle(ql.FlatForward(ql_date, 0.0, self.day_count))
        spot_handle = ql.QuoteHandle(ql.SimpleQuote(spot_price))

        heston_process = ql.HestonProcess(flat_ts, dividend_ts, spot_handle, v0, kappa, theta, sigma, rho)
        heston_model = ql.HestonModel(heston_process)
        engine = ql.AnalyticHestonEngine(heston_model)

        maturities_months = [1, 3, 6]
        moneyness_levels = [0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2]

        surface_data = []

        for m in maturities_months:
            maturity_date = ql_date + ql.Period(m, ql.Months)
            # Ensure maturity is a business day
            maturity_date = self.calendar.adjust(maturity_date)

            time_to_maturity = self.day_count.yearFraction(ql_date, maturity_date)

            for k_ratio in moneyness_levels:
                strike_price = spot_price * k_ratio

                # Option type: Call
                payoff = ql.PlainVanillaPayoff(ql.Option.Call, strike_price)
                exercise = ql.EuropeanExercise(maturity_date)
                option = ql.VanillaOption(payoff, exercise)

                option.setPricingEngine(engine)

                try:
                    price = option.NPV()
                    # Normalize price by spot? Or keep raw?
                    # Keeping raw for now, but usually normalized price (Price/Spot) is better for ML
                    normalized_price = price / spot_price
                except Exception as e:
                    print(f"Error pricing option: {e}")
                    normalized_price = 0.0

                surface_data.append({
                    'Maturity': time_to_maturity,
                    'Moneyness': k_ratio,
                    'Price': normalized_price,
                    'Strike': strike_price
                })

        return surface_data

def generate_dataset(output_path='data/processed_dataset.parquet'):
    print("Initializing Scraper...")
    scraper = MarketScraper()
    market_data = scraper.process_data()

    print("Initializing Heston Simulator...")
    simulator = HestonSimulator()

    # Filter for just one ticker for simplicity initially, or handle both
    # Let's stick to ^GSPC (S&P 500) for the main dataset
    spx_data = market_data[market_data['Ticker'] == '^GSPC'].sort_index()

    dataset_rows = []

    print(f"Generating surfaces for {len(spx_data)} days...")

    # Parameters for Heston (can be randomized slightly per day to add robustness)
    # For now, we fix them or make them dependent on realized vol

    for date_idx, row in spx_data.iterrows():
        spot = row['Close']
        realized_vol = row['RealizedVol']

        # Heuristic: Map realized vol to Heston parameters
        # v0 (initial variance) ~ realized_vol^2
        v0 = realized_vol ** 2

        # Stable parameters to ensure learnability
        # We want the mapping X -> Y to be deterministic or close to it.
        kappa = 2.0
        theta = v0 # Long term vol tracks current vol regime
        sigma = 0.3
        rho = -0.7

        surface = simulator.generate_surface(spot, v0, kappa, theta, sigma, rho)

        # Flatten surface into a feature vector
        # We need a consistent ordering: M1_K1, M1_K2, ... M3_K7
        flat_prices = []
        for point in surface:
            flat_prices.append(point['Price'])

        # Input Features: Past 30 days of returns and vol
        # We need to look back 30 days.
        # This requires us to have access to the window.
        # Efficient way: Pre-compute rolling windows or just grab them here (slower)

        # Check if we have enough history
        loc_idx = spx_data.index.get_loc(date_idx)
        if loc_idx < 30:
            continue

        past_30_days = spx_data.iloc[loc_idx-30:loc_idx]

        # Feature vector: Flattened 30x2 array (LogReturn, RealizedVol)
        # Or keep as array. Parquet supports arrays.

        input_returns = past_30_days['LogReturn'].values
        input_vols = past_30_days['RealizedVol'].values

        row_dict = {
            'Date': date_idx,
            'Spot': spot,
            'RealizedVol': realized_vol,
            'Input_Returns': input_returns,
            'Input_Vols': input_vols,
            'Target_Prices': np.array(flat_prices),
            'Heston_Params': {'kappa': kappa, 'theta': theta, 'sigma': sigma, 'rho': rho, 'v0': v0}
        }
        dataset_rows.append(row_dict)

        if len(dataset_rows) % 100 == 0:
            print(f"Processed {len(dataset_rows)} days...")

    print("Saving dataset...")
    df_final = pd.DataFrame(dataset_rows)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    df_final.to_parquet(output_path)
    print(f"Dataset saved to {output_path}")

if __name__ == "__main__":
    generate_dataset()
