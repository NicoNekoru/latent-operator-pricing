import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

class OptionDataset(Dataset):
    def __init__(self, parquet_path, mode='train'):
        self.df = pd.read_parquet(parquet_path)

        self.df['Date'] = pd.to_datetime(self.df['Date'])

        # Split based on date
        # Available range: 2023-03 to 2025-11
        # Train: <= 2024-06
        # Val: > 2024-06

        cutoff_date = pd.Timestamp('2024-06-01')

        if mode == 'train':
            self.df = self.df[self.df['Date'] <= cutoff_date]
        elif mode == 'val':
            self.df = self.df[self.df['Date'] > cutoff_date]
        elif mode == 'test':
             # For now, use Val set as Test or define a later cutoff
             self.df = self.df[self.df['Date'] > cutoff_date]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Input: Flattened (180,) -> Reshape to (30, 6)
        x_flat = np.array(row['Input_Features'], dtype=np.float32)
        x = x_flat.reshape(30, 6)

        # Target: (21,)
        y = np.array(row['Target_Prices'], dtype=np.float32)

        return torch.tensor(x), torch.tensor(y)
