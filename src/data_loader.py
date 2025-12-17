import yfinance as yf
import pandas as pd
import numpy as np
import pandas_datareader.data as web
import os
from scipy.interpolate import LinearNDInterpolator
from src.utils import calculate_implied_volatility

# Define the standard grid as expected by the model
# Maturities: 1, 3, 6 months -> Approx 30, 91, 182 days
# Moneyness: 0.8 to 1.2
GRID_MATURITIES_YEARS = np.array([1/12, 3/12, 6/12]) # [0.0833, 0.25, 0.5]
GRID_MONEYNESS = np.array([0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2])

class MarketDataLoader:
    """Fetches historical market and macro data for input features."""
    def __init__(self, start_date='2010-01-01', end_date='2023-12-31', ticker='SPY'):
        self.start_date = start_date
        self.end_date = end_date
        self.ticker = ticker

    def fetch(self):
        print(f"Fetching Market Data for {self.ticker}...")
        # 1. Market Data (Price, Volume)
        df = yf.download(self.ticker, start=self.start_date, end=self.end_date, auto_adjust=False)
        print("DEBUG: Market Data Columns:", df.columns)
        if isinstance(df.columns, pd.MultiIndex):
            try:
                df = df.xs(self.ticker, axis=1, level=0)
            except KeyError:
                 # Check if ticker is in level 1
                 if self.ticker in df.columns.levels[1]:
                     df = df.xs(self.ticker, axis=1, level=1)
                 else:
                     print("DEBUG: MultiIndex found but Ticker not in level 0 or 1. Flattening.")
                     # yfinance can return a MultiIndex with a redundant populated level
                     # or columns in (PriceType, Ticker) format.
                     pass


        # Calc Returns & Vol
        df['LogReturn'] = np.log(df['Close'] / df['Close'].shift(1))
        df['RealizedVol'] = df['LogReturn'].rolling(window=21).std() * np.sqrt(252)

        # 2. Macro Data (VIX, TNX)
        print("Fetching Macro Data...")
        macro_tickers = ['^VIX', '^TNX', '^W5000']
        macro_df = yf.download(macro_tickers, start=self.start_date, end=self.end_date, group_by='ticker')

        # Extract Close prices safely
        clean_macro = pd.DataFrame(index=macro_df.index)
        for t in macro_tickers:
            try:
                if isinstance(macro_df.columns, pd.MultiIndex):
                    clean_macro[t] = macro_df[t]['Close']
                else:
                    # If flat, might fail if multiple tickers. Assuming fetch works.
                    continue
            except KeyError:
                pass

        # Rename
        clean_macro = clean_macro.rename(columns={'^VIX': 'VIX', '^TNX': 'TNX', '^W5000': 'W5000'})

        # Merge
        full_df = df.join(clean_macro, how='left')

        # 3. GDP (Low frequency, forward fill)
        try:
            print("Fetching GDP...")
            gdp = web.DataReader('GDP', 'fred', self.start_date)
            gdp_daily = gdp.resample('D').ffill()
            full_df = full_df.join(gdp_daily, how='left')
            full_df['GDP'] = full_df['GDP'].ffill()
        except:
            print("GDP fetch failed. Using constant fallback.")
            full_df['GDP'] = 20000.0

        # 4. Feature Engineering
        # Buffett Indicator
        full_df['Buffett_Ind'] = full_df['W5000'] / (full_df['GDP'] + 1e-9)

        # Normalize Features
        full_df['VIX'] = full_df['VIX'] / 100.0
        full_df['TNX'] = full_df['TNX'] / 1000.0  # 40 -> 0.04

        # Log Volume
        full_df['Vol_Feature'] = np.log(full_df['Volume'] + 1) / 20.0

        return full_df.dropna()

class EmpiricalOptionProcessor:
    """Processes the raw empirical parquet file into training samples."""
    def __init__(self, parquet_path):
        self.parquet_path = parquet_path

    def load_and_process(self, market_df):
        print(f"Loading empirical options from {self.parquet_path}...")
        try:
            raw_ops = pd.read_parquet(self.parquet_path)
        except Exception as e:
            print(f"Error loading parquet: {e}")
            raise

        # Ensure Dates
        raw_ops['QUOTE_DATE'] = pd.to_datetime(raw_ops['QUOTE_DATE'])
        market_dates = set(market_df.index)

        dataset_rows = []

        # Group by Date
        print("Processing daily surfaces...")
        grouped = raw_ops.groupby('QUOTE_DATE')

        for current_date, group in grouped:
            if current_date not in market_dates:
                continue

            # 1. Get History Window (30 days)
            # Find integer location in market_df
            try:
                idx = market_df.index.get_loc(current_date)
            except KeyError:
                continue

            if idx < 30: continue

            window = market_df.iloc[idx-30:idx]
            if len(window) != 30: continue

            # Construct Input Features [LogReturn, Vol, VIX, Volume, TNX, Buffett]
            features = np.stack([
                window['LogReturn'].values,
                window['RealizedVol'].values,
                window['VIX'].values,
                window['Vol_Feature'].values,
                window['TNX'].values,
                window['Buffett_Ind'].values
            ], axis=1).flatten() # (180,)

            # 2. Construct Target Surface
            # Need to interpolate available options onto standard grid
            # Available points:
            # X: (Moneyness, TTM)
            # Y: Normalized Price (Price / Spot)

            # Calculate metrics
            spot = group['UNDERLYING_LAST'].iloc[0] # Assume constant for the day eod

            # We use C_LAST. Filter for valid data.
            # TTM in Years
            valid_ops = group[(group['DTE'] > 0) & (group['C_LAST'] > 0)].copy()
            if valid_ops.empty: continue

            valid_ops['TTM'] = valid_ops['DTE'] / 365.0
            valid_ops['Moneyness'] = valid_ops['STRIKE'] / spot
            valid_ops['NormPrice'] = valid_ops['C_LAST'] / spot

            # Prepare interpolation
            points = valid_ops[['Moneyness', 'TTM']].values
            values = valid_ops['NormPrice'].values

            # Target grid points at specific (k, m) pairs.
            # Output order follows models.get_standard_grid:
            # inner loop moneyness, outer loop maturity.

            target_grid = []
            for m in GRID_MATURITIES_YEARS:
                for k in GRID_MONEYNESS:
                    target_grid.append([k, m])
            target_grid = np.array(target_grid) # (21, 2)

            # Interpolate
            # Use LinearNDInterpolator
            interp = LinearNDInterpolator(points, values, fill_value=np.nan)
            interpolated_prices = interp(target_grid)

            # Check for NaNs (extrapolation)
            # If too many NaNs, skip date or fill?
            # Fill deep OTM NaNs with 0; reject missing ITM/ATM values.
            # Strategy: If NaN and Moneyness > 1.1, assume 0.
            # Else, skip if NaN.

            # Retrieve TNX (Risk-Free Rate) for this date
            # TNX is in window['TNX'], last value.
            # window['TNX'] was normalized / 1000 in fetch() ? No, check fetch().
            # fetch(): full_df['TNX'] = full_df['TNX'] / 1000.0 (line 84)
            # So it is 0.04 for 4%. Correct.
            r_rate = window['TNX'].iloc[-1]

            final_prices = []
            final_ivs = []
            valid_surface = True

            for i, (k, m) in enumerate(target_grid):
                val = interpolated_prices[i]
                if np.isnan(val):
                    # Heuristic: deep OTM calls are 0
                    if k > 1.1:
                        val = 0.0
                    else:
                        valid_surface = False
                        break

                final_prices.append(val)

                # Calculate IV
                # val is Price/Spot.
                # BSM Inputs: S=1, K=k, T=m, r=r_rate, Price=val
                iv = calculate_implied_volatility(
                    price=val, S=1.0, K=k, T=m, r=r_rate, option_type='call'
                )

                # Sanity check IV
                if np.isnan(iv) or iv < 0.01: iv = 0.01
                if iv > 3.0: iv = 3.0

                final_ivs.append(iv)

            if not valid_surface:
                continue

            # Data Cleaning: Filter out samples with extreme IVs
            # These are usually deep OTM artifacts that confuse the model
            iv_array = np.array(final_ivs)
            if np.any(iv_array > 1.5) or np.any(iv_array < 0.01):
                continue

            dataset_rows.append({
                'Date': current_date,
                'Ticker': 'SPY',
                'Input_Features': features,
                'Target_Prices': np.array(final_prices), # (21,)
                'Target_IVs': iv_array                   # (21,)
            })

        return pd.DataFrame(dataset_rows)

class DatasetBuilder:
    def __init__(self, output_path='data/processed_dataset.parquet', raw_path='data/data/empirical/processed/data.parquet'):
        self.output_path = output_path
        self.raw_path = raw_path

    def build(self):
        # 1. Fetch Features
        loader = MarketDataLoader()
        market_df = loader.fetch()

        # 2. Process Targets
        processor = EmpiricalOptionProcessor(self.raw_path)
        final_df = processor.load_and_process(market_df)

        if final_df.empty:
            raise ValueError("No valid samples generated! Check data alignment/interpolation.")

        # 3. Save
        print(f"Generated {len(final_df)} samples.")
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        final_df.to_parquet(self.output_path)
        print(f"Saved to {self.output_path}")

if __name__ == "__main__":
    DatasetBuilder().build()
