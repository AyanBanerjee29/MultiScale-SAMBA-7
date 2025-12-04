# -*- coding: utf-8 -*-
"""
Configuration for FreqSAMBA
"""

from config import ModelArgs, TrainingConfig

def get_paper_config():
    """Get configuration matching the FreqSAMBA architecture"""
    
    # Model configuration
    model_args = ModelArgs(
        d_model=32,
        n_layer=3,
        vocab_size=82,  # Will be updated in main.py based on actual data
        seq_in=60,
        seq_out=7,
        d_state=128,
        expand=2,
        dt_rank='auto',
        d_conv=3,
        pad_vocab_size_multiple=8,
        conv_bias=True,
        bias=False,
        cutoff_ratio=0.5 # NEW: 50% frequencies are considered "Trend"
    )
    
    # Training configuration
    training_config = TrainingConfig(
        dataset='STOCK_DATA',
        lag=60,
        horizon=7,
        num_nodes=82,
        val_ratio=0.15,
        test_ratio=0.15,
        input_dim=1,
        output_dim=1,
        embed_dim=10,
        rnn_units=128,
        num_layers=3,
        cheb_k=3,
        d_in=32,
        hid=32,
        batch_size=32,
        epochs=100,             # Reduced from 400 for speed
        lr_init=0.001,
        lr_decay=True,
        lr_decay_rate=0.5,
        lr_decay_step=[40, 70, 100],
        early_stop=True,
        early_stop_patience=20, # Reduced from 200 for speed
        grad_norm=False,
        max_grad_norm=5,
        loss_func='mae',
        mae_thresh=None,
        mape_thresh=0,
        device='cuda:0',
        seed=1,
        debug=True,
        log_step=20,
        log_dir='./'
    )
    
    return model_args, training_config

def get_dataset_info():
    """Metadata for logging"""
    return {
        'paper_title': 'FreqSAMBA: Multi-Scale Stock Prediction',
        'conference': 'Custom Implementation',
        'authors': ['User'],
        'total_features': 'Dynamic'
    }
