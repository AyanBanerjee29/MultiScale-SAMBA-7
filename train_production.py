# -*- coding: utf-8 -*-
"""
PRODUCTION TRAINING SCRIPT
--------------------------
1. Loads the best hyperparameters found by main.py (Research Mode).
2. Loads 100% of the data from the CSV.
3. Trains a fresh model on the FULL dataset.
4. Saves the model as 'production_samba_model.pth'.
"""

import os
import torch
import torch.nn as nn
import json
import numpy as np

# Import from your existing files (No changes needed there)
from paper_config import get_paper_config
from models import SAMBA
from utils import init_seed
from utils.data_utils import load_raw_data, create_per_window_sequences, data_loader
from trainer import Trainer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_production_model():
    print("🚀 PRODUCTION MODE: Training on Full Dataset")
    print(f"Using device: {device}")
    
    # 1. Setup Paths
    output_dir = "final_model_outputs"
    os.makedirs(output_dir, exist_ok=True)
    
    best_params_file = os.path.join(output_dir, "best_params.json")
    production_model_file = os.path.join(output_dir, "production_samba_model.pth")
    dataset_file = "Dataset/NIFTY50_features_wide.csv"

    # 2. Check prerequisites
    if not os.path.exists(best_params_file):
        print("❌ Error: 'best_params.json' not found.")
        print("   Please run 'python main.py --mode train' first to find the best hyperparameters.")
        return

    if not os.path.exists(dataset_file):
        print(f"❌ Dataset {dataset_file} not found!")
        return

    # 3. Load Configuration & Seed
    model_args, config = get_paper_config()
    init_seed(config.seed)
    args = config.to_dict()

    # 4. Load Best Hyperparameters
    with open(best_params_file, 'r') as f:
        best_params = json.load(f)
    print(f"✅ Loaded Best Params: {best_params}")

    # 5. Load FULL Data
    print("Loading full dataset...")
    df_full, price_index = load_raw_data(dataset_file, target_col_name='close')
    
    # Update vocab size based on actual data
    num_features = len(df_full.columns)
    model_args.vocab_size = num_features
    if not hasattr(model_args, 'cutoff_ratio'): model_args.cutoff_ratio = 0.5

    # 6. Create Sequences (No Train/Test Split -> Use ALL data)
    window = config.lag
    predict = config.horizon
    XX_all, YY_all, MM_all = create_per_window_sequences(df_full, window, predict, price_index)
    
    print(f"Total Samples Available: {len(XX_all)}")

    # 7. Create Data Loaders
    # Even in production, we need a tiny "validation" set just so the Trainer 
    # knows when to stop (Early Stopping). We use the last 5% for this.
    split_idx = int(len(XX_all) * 0.95)
    
    X_train = XX_all[:split_idx]
    y_train = YY_all[:split_idx]
    mm_train = MM_all[:split_idx]
    
    X_val = XX_all[split_idx:]
    y_val = YY_all[split_idx:]
    mm_val = MM_all[split_idx:]
    
    train_loader = data_loader(X_train, y_train, mm_train, batch_size=64, shuffle=True)
    val_loader = data_loader(X_val, y_val, mm_val, batch_size=64, shuffle=False)
    
    print(f"Training Set: {len(X_train)} samples")
    print(f"Internal Val Set (for Early Stopping): {len(X_val)} samples")

    # 8. Initialize Model (Fresh Weights)
    print("Initializing fresh Multi-Scale SAMBA model...")
    model = SAMBA(
        model_args, 
        best_params['hid'], 
        window, 
        predict, 
        best_params['embed_dim'], 
        best_params["cheb_k"]
    ).to(device)

    # Reset weights to ensure clean training
    for p in model.parameters():
        if p.dim() > 1: nn.init.xavier_uniform_(p)
        else: nn.init.uniform_(p)

    # 9. Train
    loss_fn = torch.nn.MSELoss().to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=best_params['lr_init'])
    
    trainer = Trainer(model, loss_fn, optimizer, train_loader, val_loader, args)
    
    print("Starting Production Training...")
    best_model_state, _ = trainer.train()

    # 10. Save Production Model
    torch.save(best_model_state, production_model_file)
    print("-" * 50)
    print(f"✅ SUCCESS! Production model saved to: {production_model_file}")
    print("You can now run 'python predict_production.py' to forecast the future.")
    print("-" * 50)

if __name__ == "__main__":
    train_production_model()
