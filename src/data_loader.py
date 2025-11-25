import yfinance as yf
import pandas as pd
import numpy as np
import QuantLib as ql
from datetime import date, timedelta
import os
import pandas_datareader.data as web

class MarketData:
    """Fetches and processes standard asset price data."""
    def __init__(self, tickers=['^GSPC', '^NDX', '^RUT', '^DJI'], start_date='2006-01-01', end_date=None):
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date

    def fetch(self):
        print(f"Fetching Market Data for {self.tickers}...")
        data = yf.download(self.tickers, start=self.start_date, end=self.end_date, group_by='ticker')

        processed_dfs = []
        is_multi = isinstance(data.columns, pd.MultiIndex)

        for ticker in self.tickers:
            try:
                if is_multi:
                    df = data[ticker].copy()
                elif len(self.tickers) == 1:
                    df = data.copy()
                else:
                    continue

                if df.empty: continue

                # Calculate Returns & Vol
                df['LogReturn'] = np.log(df['Close'] / df['Close'].shift(1))
                df['RealizedVol'] = df['LogReturn'].rolling(window=21).std() * np.sqrt(252)

                df = df.dropna()
                df['Ticker'] = ticker
                processed_dfs.append(df)
            except KeyError:
                print(f"Warning: Could not process {ticker}")
                continue

        if not processed_dfs:
            raise ValueError("No market data processed!")

        return pd.concat(processed_dfs)

class MacroData:
    """Fetches macro indicators and calculates Buffett Indicator."""
    def __init__(self, start_date='2006-01-01'):
        self.start_date = start_date

    def fetch(self):
        print("Fetching Macro Data (VIX, TNX, Wilshire 5000)...")

        # 1. Fetch Daily Macro from yfinance
        # ^VIX: Volatility Index
        # ^TNX: 10-Year Treasury Yield (x10)
        # ^W5000: Wilshire 5000 (Total Market Cap Proxy)
        macro_tickers = ['^VIX', '^TNX', '^W5000']
        yf_data = yf.download(macro_tickers, start=self.start_date, group_by='ticker')

        # Extract Close prices
        macro_df = pd.DataFrame(index=yf_data.index)

        # Handle MultiIndex or Single Index
        if isinstance(yf_data.columns, pd.MultiIndex):
            if '^VIX' in yf_data.columns.levels[0]:
                macro_df['VIX'] = yf_data['^VIX']['Close']
            if '^TNX' in yf_data.columns.levels[0]:
                macro_df['TNX'] = yf_data['^TNX']['Close']
            if '^W5000' in yf_data.columns.levels[0]:
                macro_df['W5000'] = yf_data['^W5000']['Close']
        else:
            # Fallback if structure is weird
            pass

        # 2. Fetch Quarterly GDP from FRED
        print("Fetching GDP from FRED...")
        try:
            gdp = web.DataReader('GDP', 'fred', self.start_date)
            # Resample GDP to daily (forward fill) to align with market data
            gdp_daily = gdp.resample('D').ffill()
            macro_df = macro_df.join(gdp_daily, how='left')
            macro_df['GDP'] = macro_df['GDP'].ffill() # Forward fill for days after last quarter
        except Exception as e:
            print(f"Error fetching GDP: {e}. Using constant growth approximation.")
            # Fallback: Assume 20T growing at 2%
            macro_df['GDP'] = 20000.0 # Placeholder

        # 3. Calculate Features
        # Buffett Indicator: Market Cap / GDP
        # Note: Wilshire 5000 is an index, not raw cap in dollars, but it's a valid proxy for the ratio's trend.
        macro_df['Buffett_Ind'] = macro_df['W5000'] / (macro_df['GDP'] + 1e-9)

        # Normalize/Scale features roughly to model range (0-1 or similar)
        # VIX is 0-100, divide by 100
        macro_df['VIX'] = macro_df['VIX'] / 100.0
        # TNX is yield * 10 (e.g. 40 = 4.0%), divide by 1000 to get 0.04
        macro_df['TNX'] = macro_df['TNX'] / 1000.0

        # Fill missing
        macro_df = macro_df.ffill().bfill()

        return macro_df[['VIX', 'TNX', 'Buffett_Ind']]

class HestonGenerator:
    """Simulates option surfaces using QuantLib."""
    def __init__(self, risk_free_rate=0.03):
        self.risk_free_rate = risk_free_rate
        self.day_count = ql.Actual365Fixed()
        self.calendar = ql.UnitedStates(ql.UnitedStates.NYSE)
        self.engine = None

    def setup_engine(self, spot, v0, kappa, theta, sigma, rho):
        today = date.today()
        ql_date = ql.Date(today.day, today.month, today.year)
        ql.Settings.instance().evaluationDate = ql_date

        flat_ts = ql.YieldTermStructureHandle(ql.FlatForward(ql_date, self.risk_free_rate, self.day_count))
        dividend_ts = ql.YieldTermStructureHandle(ql.FlatForward(ql_date, 0.0, self.day_count))
        spot_handle = ql.QuoteHandle(ql.SimpleQuote(spot))

        process = ql.HestonProcess(flat_ts, dividend_ts, spot_handle, v0, kappa, theta, sigma, rho)
        model = ql.HestonModel(process)
        self.engine = ql.AnalyticHestonEngine(model)
        self.ql_date = ql_date

    def generate(self, spot):
        maturities = [1, 3, 6] # Months
        moneyness = [0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2]
        prices = []

        for m in maturities:
            maturity_date = self.calendar.adjust(self.ql_date + ql.Period(m, ql.Months))
            for k_ratio in moneyness:
                strike = spot * k_ratio
                payoff = ql.PlainVanillaPayoff(ql.Option.Call, strike)
                exercise = ql.EuropeanExercise(maturity_date)
                option = ql.VanillaOption(payoff, exercise)
                option.setPricingEngine(self.engine)
                try:
                    price = option.NPV()
                    prices.append(price / spot) # Normalize
                except:
                    prices.append(0.0)
        return np.array(prices)

class DatasetBuilder:
    """Orchestrates the creation of the enriched dataset."""
    def __init__(self, output_path='data/processed_dataset.parquet'):
        self.output_path = output_path

    def build(self):
        # 1. Fetch Data
        market = MarketData().fetch()
        macro = MacroData().fetch()

        # 2. Merge
        # Join Macro data to Market data on Date index
        merged = market.join(macro, how='left').ffill().dropna()

        # 3. Generate Surfaces
        heston = HestonGenerator()
        dataset_rows = []

        print("Generating Heston Surfaces and building dataset...")
        for ticker in merged['Ticker'].unique():
            df = merged[merged['Ticker'] == ticker].sort_index()

            for i in range(30, len(df)):
                row = df.iloc[i]

                # Heston Params (Heuristic)
                v0 = row['RealizedVol'] ** 2
                kappa, theta, sigma, rho = 2.0, v0, 0.3, -0.7

                heston.setup_engine(row['Close'], v0, kappa, theta, sigma, rho)
                prices = heston.generate(row['Close'])

                # Input Window (30 days)
                window = df.iloc[i-30:i]

                # Features: [LogReturn, Vol, VIX, Volume, TNX, Buffett]
                # Note: Volume needs normalization. Let's use Log Volume relative to mean?
                # For simplicity in this POC, we'll use raw Log Volume.
                # Actually, let's use Volume Change or just normalized Volume.
                # Let's use Log(Volume) / 20 (rough scale).

                vol_feature = np.log(window['Volume'] + 1) / 20.0

                # Stack Features
                # Shape: (30, 6)
                features = np.stack([
                    window['LogReturn'].values,
                    window['RealizedVol'].values,
                    window['VIX'].values,
                    vol_feature.values,
                    window['TNX'].values,
                    window['Buffett_Ind'].values
                ], axis=1)

                dataset_rows.append({
                    'Date': df.index[i],
                    'Ticker': ticker,
                    'Input_Features': features.flatten(), # Flatten to 1D (180,) for Parquet
                    'Target_Prices': prices
                })

            print(f"Processed {ticker}...")

        # 4. Save
        final_df = pd.DataFrame(dataset_rows)
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        final_df.to_parquet(self.output_path)
        print(f"Saved enriched dataset to {self.output_path}")

if __name__ == "__main__":
    DatasetBuilder().build()
