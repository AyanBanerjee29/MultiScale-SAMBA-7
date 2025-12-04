# -*- coding: utf-8 -*-
"""
Utility package initialization
"""

try:
    from .metrics import All_Metrics, pearson_correlation, rank_information_coefficient
except ImportError:
    print("Warning: metrics.py not found in utils")
    All_Metrics = None
    pearson_correlation = None
    rank_information_coefficient = None

try:
    from .model_utils import init_seed
except ImportError:
    print("Warning: model_utils.py not found in utils")
    init_seed = None

try:
    from .logger import get_logger
except ImportError:
    print("Warning: logger.py not found in utils")
    get_logger = None

from .data_utils import MinMaxNorm01, data_loader, load_raw_data, create_per_window_sequences

__all__ = [
    'All_Metrics',
    'pearson_correlation',
    'rank_information_coefficient',
    'init_seed',
    'get_logger',
    'MinMaxNorm01',
    'data_loader',
    'load_raw_data',
    'create_per_window_sequences'
]
