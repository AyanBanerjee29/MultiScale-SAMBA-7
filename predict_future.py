# -*- coding: utf-8 -*-
"""
PRODUCTION FORECAST SCRIPT
--------------------------
1. Loads 'production_samba_model.pth'.
2. Loads the latest data.
3. Predicts the next 7 days relative to today.
"""

import torch
import pandas as pd
import numpy as np
import json
import os
from datetime import timedelta

# Import from your existing files
from paper_config import get_paper_config
from models import SAMBA
from utils.data_utils import load_raw_data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def predict_future():
    print("🔮 FreqSAMBA Production Forecast")
    print("==============================")

    # 1. Paths
    output_dir = "final_model_outputs"
    dataset_file = "Dataset/NIFTY50_features_wide.csv"
    best_params_file = os.path.join(output_dir, "best_params.json")
    production_model_file = os.path.join(output_dir, "production_samba_model.pth")

    # 2. Validate Files
    if not os.path.exists(production_model_file):
        print("❌ Error: Production model not found.")
        print("   Run 'python train_production.py' first.")
        return

    # 3. Load Data
    print("Loading latest market data...")
    df_full, price_index = load_raw_data(dataset_file, target_col_name='close')
    
    model_args, config = get_paper_config()
    lag = config.lag
    
    # Get the specific Last Window (Lookback)
    if len(df_full) < lag:
        print("Error: Not enough data.")
        return

    last_window_df = df_full.tail(lag)
    last_date = last_window_df.index[-1]
    print(f"📅 Input Data Ends On: {last_date.date()}")

    # 4. Per-Window Scaling (Exact logic from training)
    raw_data = last_window_df.to_numpy(dtype=np.float32)
    local_min = np.min(raw_data, axis=0)
    local_max = np.max(raw_data, axis=0)
    
    # Scale and Clamp
    x_scaled = (raw_data - local_min) / (local_max - local_min + 1e-8)
    x_scaled = np.clip(x_scaled, -3.0, 3.0) # Safety clamp
    x_scaled = np.nan_to_num(x_scaled, nan=0.0) # Safety sanitizer
    
    input_tensor = torch.from_numpy(x_scaled).float().unsqueeze(0).to(device)

    # 5. Load Production Model
    print("Loading Production Model...")
    with open(best_params_file, 'r') as f:
        best_params = json.load(f)
        
    model_args.vocab_size = len(df_full.columns)
    if not hasattr(model_args, 'cutoff_ratio'): model_args.cutoff_ratio = 0.5
    
    model = SAMBA(
        model_args, 
        best_params['hid'], 
        lag, 
        config.horizon, 
        best_params['embed_dim'], 
        best_params["cheb_k"]
    ).to(device)
    
    model.load_state_dict(torch.load(production_model_file, map_location=device))
    model.eval()

    # 6. Predict
    print("Calculating Forecast...")
    with torch.no_grad():
        prediction_scaled = model(input_tensor)

    # 7. Inverse Transform (to Real Price)
    target_min = local_min[price_index]
    target_max = local_max[price_index]
    
    prediction_real = prediction_scaled.cpu().numpy().flatten()
    prediction_real = prediction_real * (target_max - target_min) + target_min

    # 8. Output
    print("\n📊 PREDICTION FOR NEXT 7 DAYS:")
    print("-" * 30)
    
    future_dates = pd.bdate_range(start=last_date + timedelta(days=1), periods=7)
    
    for date, price in zip(future_dates, prediction_real):
        print(f"{date.date()} : {price:.2f}")
    print("-" * 30)

if __name__ == "__main__":
    predict_future()
