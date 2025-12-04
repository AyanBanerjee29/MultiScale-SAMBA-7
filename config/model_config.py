# -*- coding: utf-8 -*-
"""
Configuration classes for SAMBA model
"""
import math
from dataclasses import dataclass, field
from typing import List, Optional, Union

@dataclass
class ModelArgs:
    d_model: int
    n_layer: int
    vocab_size: int
    seq_in: int = 60
    seq_out: int = 7
    d_state: int = 16
    expand: int = 2
    dt_rank: Union[int, str] = 'auto'
    d_conv: int = 4
    pad_vocab_size_multiple: int = 8
    conv_bias: bool = True
    bias: bool = False
    
    # NEW: Added for FreqSAMBA (FFT Split)
    cutoff_ratio: float = 0.5

    def __post_init__(self):
        self.d_inner = int(self.expand * self.d_model)
        
        if self.dt_rank == 'auto':
            self.dt_rank = math.ceil(self.d_model / 16)
        
        if self.vocab_size % self.pad_vocab_size_multiple != 0:
            self.vocab_size += (self.pad_vocab_size_multiple - self.vocab_size % self.pad_vocab_size_multiple)

@dataclass
class TrainingConfig:
    dataset: str = 'STOCK_DATA'
    lag: int = 60
    horizon: int = 7
    num_nodes: int = 82
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    input_dim: int = 1
    output_dim: int = 1
    embed_dim: int = 10
    rnn_units: int = 128
    num_layers: int = 3
    cheb_k: int = 3
    d_in: int = 32
    hid: int = 32
    batch_size: int = 32
    epochs: int = 100
    lr_init: float = 0.001
    lr_decay: bool = True
    lr_decay_rate: float = 0.5
    lr_decay_step: List[int] = field(default_factory=lambda: [40, 70, 100])
    early_stop: bool = True
    early_stop_patience: int = 20
    grad_norm: bool = False
    max_grad_norm: int = 5
    loss_func: str = 'mae'
    mae_thresh: Optional[float] = None
    mape_thresh: float = 0
    device: str = 'cuda:0'
    seed: int = 1
    debug: bool = True
    log_step: int = 20
    log_dir: str = './'

    def to_dict(self):
        return {k: v for k, v in self.__dict__.items()}

    def get(self, key, default=None):
        return getattr(self, key, default)
