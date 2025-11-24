import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

class OptionDataset(Dataset):
    def __init__(self, parquet_path, mode='train'):
        self.df = pd.read_parquet(parquet_path)

        self.df['Date'] = pd.to_datetime(self.df['Date'])

        # Split based on date
        # Train: <= 2022
        # Val: >= 2023 (Includes 2024-2025)

        if mode == 'train':
            self.df = self.df[self.df['Date'].dt.year <= 2022]
        elif mode == 'val':
            self.df = self.df[self.df['Date'].dt.year >= 2023]
        elif mode == 'test':
             self.df = self.df[self.df['Date'].dt.year >= 2024]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Inputs: (30, 2)
        returns = np.array(row['Input_Returns'], dtype=np.float32)
        vols = np.array(row['Input_Vols'], dtype=np.float32)

        x = np.stack([returns, vols], axis=1) # Shape (30, 2)

        # Targets: (21,)
        y = np.array(row['Target_Prices'], dtype=np.float32)

        return torch.tensor(x), torch.tensor(y)
