import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

class OptionDataset(Dataset):
    def __init__(self, parquet_path, mode='train'):
        self.df = pd.read_parquet(parquet_path)

        self.df['Date'] = pd.to_datetime(self.df['Date'])

        # Split based on User Requirement
        # Train Era: 2010-2022
        # Train: 2010-01-01 to 2021-12-31
        # Val:   2022-01-01 to 2022-12-31

        train_start = pd.Timestamp('2010-01-01')
        val_start = pd.Timestamp('2022-01-01')
        end_date = pd.Timestamp('2022-12-31')

        if mode == 'train':
            self.df = self.df[(self.df['Date'] >= train_start) & (self.df['Date'] < val_start)]
        elif mode == 'val':
            self.df = self.df[(self.df['Date'] >= val_start) & (self.df['Date'] <= end_date)]
        elif mode == 'test':
             # Test sets are handled manually in backtest.py, but for completeness:
             # This would be the "Recent" test set
             self.df = self.df[self.df['Date'] >= pd.Timestamp('2023-01-01')]
        elif mode == 'crisis':
             # Crisis Test Set: 2006-01-01 to 2009-12-31
             start = pd.Timestamp('2006-01-01')
             end = pd.Timestamp('2009-12-31')
             self.df = self.df[(self.df['Date'] >= start) & (self.df['Date'] <= end)]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Input: Flattened (180,) -> Reshape to (30, 6)
        x_flat = np.array(row['Input_Features'], dtype=np.float32)
        x = x_flat.reshape(30, 6)

        # Target: (21,)
        y_price = np.array(row['Target_Prices'], dtype=np.float32)
        y_iv = np.array(row['Target_IVs'], dtype=np.float32)

        # Domain Label for Adversarial Training
        # Domain 0: Pre-COVID/Inflation (<= 2019)
        # Domain 1: Post-COVID/Inflation (> 2019) -> Target Regime
        date = pd.Timestamp(row['Date'])
        domain_label = 1.0 if date.year > 2019 else 0.0

        return torch.tensor(x), torch.tensor(y_price), torch.tensor(y_iv), torch.tensor(domain_label, dtype=torch.float32)
