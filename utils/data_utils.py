# -*- coding: utf-8 -*-
"""
Data loading and preprocessing utilities.
Features: Per-Window Scaling + Lazy Loading (CPU tensors)
"""

import torch
import torch.utils.data
import pandas as pd
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MinMaxNorm01:
    """Scale data to range [0, 1] (Global Scaler - Helper)"""
    def __init__(self):
        pass
    def fit(self, x):
        self.min = x.min(axis=0)
        self.max = x.max(axis=0)
    def transform(self, x):
        x = 1.0 * (x - self.min) / (self.max - self.min + 1e-8)
        return x
    def fit_transform(self, x):
        self.fit(x)
        return self.transform(x)
    def inverse_transform(self, x):
        x_out = x * (self.max[0] - self.min[0] + 1e-8) + self.min[0]
        return x_out

def data_loader(X, Y, MM, batch_size, shuffle=True, drop_last=True):
    """
    Create PyTorch DataLoader.
    Inputs are kept on CPU to save GPU memory.
    """
    data = torch.utils.data.TensorDataset(X, Y, MM)
    dataloader = torch.utils.data.DataLoader(
        data, 
        batch_size=batch_size,
        shuffle=shuffle, 
        drop_last=drop_last
    )
    return dataloader

def load_raw_data(csv_file, target_col_name='close'):
    """Loads raw CSV and ensures target is at index 0."""
    df = pd.read_csv(csv_file, index_col=0, parse_dates=True)
    df.index.name = "Date"
    
    if target_col_name not in df.columns:
        raise KeyError(f"Fatal Error: Target column '{target_col_name}' not found.")
        
    try:
        price_index = df.columns.get_loc(target_col_name)
    except KeyError:
        raise KeyError(f"Target column '{target_col_name}' not found in DataFrame.")
        
    print(f"Target column '{target_col_name}' found at index: {price_index}")

    if price_index != 0:
        print(f"Moving target column '{target_col_name}' to index 0.")
        cols = [target_col_name] + [col for col in df.columns if col != target_col_name]
        df = df[cols]
        price_index = 0

    df.dropna(inplace=True)
    num_features = len(df.columns)
    print(f"Data loaded with {num_features} features.")
    
    return df, price_index

def create_per_window_sequences(raw_data_df, window, predict, price_index=0):
    """
    Creates sequences with PER-WINDOW scaling.
    Returns: XX, YY, MM (Local Min/Max values) on CPU.
    """
    raw_data = raw_data_df.to_numpy(dtype=np.float32)
    X_seq, Y_seq, MM_seq = [], [], []
    len_data = len(raw_data)
    
    print("Generating sequences with per-window scaling...")
    
    for i in range(len_data - window - predict + 1):
        # 1. Get Raw Window and Target
        x_raw = raw_data[i : i + window] 
        y_raw = raw_data[i + window : i + window + predict, price_index]
        
        # 2. Calculate Local Min/Max for this specific window
        local_min = np.min(x_raw, axis=0)
        local_max = np.max(x_raw, axis=0)
        
        # 3. Scale the Input Window (X)
        denom = local_max - local_min + 1e-8
        x_scaled = (x_raw - local_min) / denom
        
        # 4. Scale the Target (Y) using the TARGET'S min/max
        target_min = local_min[price_index]
        target_max = local_max[price_index]
        target_denom = target_max - target_min + 1e-8
        
        y_scaled = (y_raw - target_min) / target_denom
        
        X_seq.append(x_scaled)
        Y_seq.append(y_scaled)
        MM_seq.append([target_min, target_max])

    # --- MEMORY FIX: Keep tensors on CPU ---
    XX = torch.from_numpy(np.array(X_seq)).float()
    YY = torch.from_numpy(np.array(Y_seq)).float()
    MM = torch.from_numpy(np.array(MM_seq)).float()
    
    if YY.dim() == 2:
        YY = YY.unsqueeze(-1)

    return XX, YY, MM
