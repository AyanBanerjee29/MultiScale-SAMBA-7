# -*- coding: utf-8 -*-
"""
Main script for FreqSAMBA (FFT Multi-Scale + Per-Window Scaling + Lazy Loading)
"""
import argparse
import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import json
from sklearn.model_selection import TimeSeriesSplit, ParameterGrid

# --- Import project modules ---
from paper_config import get_paper_config, get_dataset_info
from models import SAMBA
from utils import (
    init_seed, pearson_correlation, rank_information_coefficient, All_Metrics, get_logger
)
from utils.data_utils import load_raw_data, create_per_window_sequences, data_loader
from trainer import Trainer

# --- GPU/CPU Configuration ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# -----------------------------

def evaluate_loader_with_inverse(model, loader):
    """
    Custom evaluation loop that handles Per-Window Inverse Transformation.
    Returns Real (De-scaled) Predictions and Real Targets.
    Handles Lazy Loading (CPU -> GPU).
    """
    model.eval()
    y_pred_real_list = []
    y_true_real_list = []
    
    # Determine model device
    device = next(model.parameters()).device
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            # MEMORY FIX: Move batch components to GPU here
            data = batch[0].to(device)  # Input
            target = batch[1].to(device) # Target (Scaled)
            mm = batch[2].to(device)    # Min/Max params

            output = model(data) # Output is Scaled (0-1)
            
            # --- INVERSE TRANSFORM LOGIC ---
            # Real = Scaled * (Max - Min) + Min
            
            batch_min = mm[:, 0].view(-1, 1, 1) # [batch, 1, 1]
            batch_max = mm[:, 1].view(-1, 1, 1) # [batch, 1, 1]
            
            pred_real = output * (batch_max - batch_min) + batch_min
            target_real = target * (batch_max - batch_min) + batch_min
            
            # Move results back to CPU to save GPU memory
            y_pred_real_list.append(pred_real.cpu())
            y_true_real_list.append(target_real.cpu())
            
    y_p = torch.cat(y_pred_real_list, dim=0)
    y_t = torch.cat(y_true_real_list, dim=0)
    
    return y_p, y_t

def main(cli_args):
    """Main training and testing function"""
    
    # --- 1. Initial Setup ---
    model_args, config = get_paper_config()
    init_seed(config.seed)
    
    output_dir = "final_model_outputs"
    os.makedirs(output_dir, exist_ok=True)
    best_params_file = os.path.join(output_dir, "best_params.json")
    final_model_file = os.path.join(output_dir, "final_samba_model.pth")
    
    print("🚀 FreqSAMBA: Multi-Scale Stock Prediction")
    print(f"Using device: {device}")
    
    # --- 2. STAGE 0: Load Data ---
    print("STAGE 0: Loading and Windowing Data...")
    dataset_file = "Dataset/NIFTY50_features_wide.csv"
    if not os.path.exists(dataset_file):
        print(f"❌ Dataset {dataset_file} not found! Run create_feature_dataset.py first.")
        return

    df_full, price_index = load_raw_data(dataset_file, target_col_name='close')
    num_features = len(df_full.columns)
    model_args.vocab_size = num_features
    config.num_nodes = num_features
    
    # --- Create Sequences (Stored on CPU) ---
    window = config.lag
    predict = config.horizon
    XX_all, YY_all, MM_all = create_per_window_sequences(df_full, window, predict, price_index)
    
    # --- Split 80% (Dev) / 20% (Final Test) ---
    total_samples = len(XX_all)
    test_split_ratio = 0.20
    test_size = int(total_samples * test_split_ratio)
    dev_size = total_samples - test_size
    
    XX_dev, YY_dev, MM_dev = XX_all[:dev_size], YY_all[:dev_size], MM_all[:dev_size]
    XX_test, YY_test, MM_test = XX_all[dev_size:], YY_all[dev_size:], MM_all[dev_size:]
    
    print(f"Total sequences: {total_samples}")
    print(f"Development Set: {len(XX_dev)}")
    print(f"Final Test Set: {len(XX_test)}")

    args = config.to_dict()

    # --- 5. STAGE 1: Hyperparameter Tuning ---
    if cli_args.mode == 'train':
        print("\n===== STAGE 1: HYPERPARAMETER TUNING (on 80% Dev Set) =====")

        # Optimized grid for speed
        param_grid = {
            'lr_init': [0.001],       # Stick to 0.001 for now
            'hid': [32],              # Keep models small/fast
            'embed_dim': [10],
            'cheb_k': [2, 3]          # Test graph connectivity
        }
        
        param_list = list(ParameterGrid(param_grid))
        results = []
        
        for i, params in enumerate(param_list):
            print(f"\n--- Tuning Run {i+1}/{len(param_list)}: {params} ---")
            fold_scores = []
            tscv = TimeSeriesSplit(n_splits=3) # Reduced folds to 3
            
            for fold, (train_index, val_index) in enumerate(tscv.split(XX_dev)):
                # Standard Fold Split (Views on CPU)
                X_train_fold, y_train_fold, mm_train_fold = XX_dev[train_index], YY_dev[train_index], MM_dev[train_index]
                X_val_fold, y_val_fold, mm_val_fold = XX_dev[val_index], YY_dev[val_index], MM_dev[val_index]

                # Inner Split for Early Stopping
                val_ratio_inner = 0.15
                inner_split = int(len(X_train_fold) * (1 - val_ratio_inner))
                
                if inner_split < 10: continue 

                X_train_in, y_train_in, mm_train_in = X_train_fold[:inner_split], y_train_fold[:inner_split], mm_train_fold[:inner_split]
                X_val_in, y_val_in, mm_val_in = X_train_fold[inner_split:], y_train_fold[inner_split:], mm_train_fold[inner_split:]

                # DataLoaders
                train_loader = data_loader(X_train_in, y_train_in, mm_train_in, 64, shuffle=True)
                val_loader_in = data_loader(X_val_in, y_val_in, mm_val_in, 64, shuffle=False)
                fold_val_loader = data_loader(X_val_fold, y_val_fold, mm_val_fold, 64, shuffle=False)

                model = SAMBA(
                    model_args, params['hid'], window, predict,
                    params['embed_dim'], params["cheb_k"]
                ).to(device)

                for p in model.parameters():
                    if p.dim() > 1: nn.init.xavier_uniform_(p)
                    else: nn.init.uniform_(p)

                loss_fn = torch.nn.MSELoss().to(device)
                optimizer = torch.optim.Adam(params=model.parameters(), lr=params['lr_init'])
                
                trainer = Trainer(model, loss_fn, optimizer, train_loader, val_loader_in, args)
                best_model_state, _ = trainer.train()
                
                # Evaluate on Fold Val Set
                model.load_state_dict(best_model_state)
                y_p_real, y_t_real = evaluate_loader_with_inverse(model, fold_val_loader)
                
                mae, _, _ = All_Metrics(y_p_real, y_t_real, None, None)
                fold_scores.append(mae.item())

                # --- MEMORY CLEANUP ---
                del model, trainer, optimizer, loss_fn
                torch.cuda.empty_cache()
                # ----------------------

            if len(fold_scores) > 0:
                avg_score = np.mean(fold_scores)
                print(f"--- Avg. Real MAE for {params}: {avg_score:.4f} ---")
                results.append({'params': params, 'score': avg_score})

        if not results:
             print("No valid results.")
             return

        best_result = min(results, key=lambda x: x['score'])
        best_params = best_result['params']
        print("\n===== STAGE 1 Complete =====")
        print(f"Best MAE: {best_result['score']:.4f}")
        print(f"Best Params: {best_params}")
        
        with open(best_params_file, 'w') as f:
            json.dump(best_params, f)

        # --- 6. STAGE 2: Train Final Model ---
        print("\n===== STAGE 2: Training Final Model =====")
        
        final_split = int(len(XX_dev) * 0.85)
        X_train_final = XX_dev[:final_split]
        y_train_final = YY_dev[:final_split]
        mm_train_final = MM_dev[:final_split]
        
        X_val_final = XX_dev[final_split:]
        y_val_final = YY_dev[final_split:]
        mm_val_final = MM_dev[final_split:]
        
        train_loader_final = data_loader(X_train_final, y_train_final, mm_train_final, 64, shuffle=True)
        val_loader_final = data_loader(X_val_final, y_val_final, mm_val_final, 64, shuffle=False)
        
        final_model = SAMBA(
            model_args, best_params['hid'], window, predict,
            best_params['embed_dim'], best_params["cheb_k"]
        ).to(device)
        
        for p in final_model.parameters():
            if p.dim() > 1: nn.init.xavier_uniform_(p)
            else: nn.init.uniform_(p)

        loss_fn = torch.nn.MSELoss().to(device)
        optimizer = torch.optim.Adam(params=final_model.parameters(), lr=best_params['lr_init'])

        final_trainer = Trainer(final_model, loss_fn, optimizer, train_loader_final, val_loader_final, args)
        best_model_state, _ = final_trainer.train()

        torch.save(best_model_state, final_model_file)
        print(f"Final model saved to {final_model_file}")

    # --- 7. STAGE 3: Final Evaluation ---
    print(f"\n===== STAGE 3: Final Evaluation (on 20% Held-Out Test Set) =====")
    
    if not os.path.exists(final_model_file) or not os.path.exists(best_params_file):
        print("Error: Model files not found. Run train mode first.")
        return
        
    with open(best_params_file, 'r') as f:
        best_params = json.load(f)
    
    final_model = SAMBA(
        model_args, best_params['hid'], window, predict,
        best_params['embed_dim'], best_params["cheb_k"]
    ).to(device)
    
    final_model.load_state_dict(torch.load(final_model_file, map_location=device))
    
    test_loader_final = data_loader(XX_test, YY_test, MM_test, 64, shuffle=False)
    
    y_p_real, y_t_real = evaluate_loader_with_inverse(final_model, test_loader_final)
    
    mae, rmse, _ = All_Metrics(y_p_real, y_t_real, None, None)
    IC = pearson_correlation(y_t_real, y_p_real)
    RIC = rank_information_coefficient(y_t_real.squeeze(), y_p_real.squeeze())

    print("\n===== FINAL MODEL PERFORMANCE (REAL PRICES) =====")
    print(f"MAE:  {mae.item():.4f}")
    print(f"RMSE: {rmse.item():.4f}")
    print(f"IC:   {IC.item():.4f}")
    print(f"RIC:  {RIC.item() if torch.is_tensor(RIC) else RIC:.4f}")

    # --- Plotting Logic ---
    print("\nPlotting final results...")
    y_t_plot = y_t_real.squeeze().numpy()
    y_p_plot = y_p_real.squeeze().numpy()
    if y_t_plot.ndim == 1: y_t_plot = y_t_plot.reshape(-1, 1)
    if y_p_plot.ndim == 1: y_p_plot = y_p_plot.reshape(-1, 1)

    for i in range(config.horizon):
        plt.figure(figsize=(12, 6))
        plt.plot(y_t_plot[:, i], label=f"Actual Price (Day {i+1})", linewidth=2)
        plt.plot(y_p_plot[:, i], label=f"Predicted Price (Day {i+1})", linewidth=2, linestyle="--")
        plt.xlabel("Time (Final Test Samples)")
        plt.ylabel("Stock Price")
        plt.title(f"FreqSAMBA Prediction (Day {i+1})")
        plt.legend()
        plt.grid(True)
        plot_path = os.path.join(output_dir, f"final_test_plot_day_{i+1}.png")
        plt.savefig(plot_path)
        print(f"Saved plot: {plot_path}")
        plt.close()

    print("\nScript finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    cli_args = parser.parse_args()
    main(cli_args)
