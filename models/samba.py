# -*- coding: utf-8 -*-
"""
FreqSAMBA: Multi-Scale SAMBA with FFT-based Decomposition
Corrected Fusion Layer Dimensions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .mamba import Mamba

class SeriesDecomp(nn.Module):
    """
    Series decomposition block using FFT (Fast Fourier Transform).
    Splits the time series into:
    1. Trend (Low Frequency Band)
    2. Residual (High Frequency Band)
    """
    def __init__(self, cutoff_ratio=0.5):
        super().__init__()
        self.cutoff_ratio = cutoff_ratio

    def forward(self, x):
        """
        x: [Batch, Seq_Len, Channels]
        """
        # 1. Perform Real FFT
        x_fft = torch.fft.rfft(x, dim=1)
        
        # 2. Determine the "Cutoff" index
        freq_len = x_fft.shape[1]
        cutoff = int(freq_len * self.cutoff_ratio)
        if cutoff == 0: cutoff = 1

        # --- Branch A: TREND (Low Freq Band) ---
        trend_fft = x_fft.clone()
        trend_fft[:, cutoff:, :] = 0 
        x_trend = torch.fft.irfft(trend_fft, n=x.shape[1], dim=1)
        
        # --- Branch B: RESIDUAL (High Freq Band) ---
        x_resid = x - x_trend
        
        return x_trend, x_resid

class SAMBA(nn.Module):
    """
    FreqSAMBA Architecture
    """
    
    def __init__(self, model_args, hidden, inp, out, embed, cheb_k):
        super().__init__()
        self.args = model_args
        
        # --- DECOMPOSITION MODULE ---
        ratio = getattr(model_args, 'cutoff_ratio', 0.5)
        self.decomp = SeriesDecomp(cutoff_ratio=ratio)
        
        # --- BRANCHES ---
        self.mam_trend = Mamba(model_args, hidden)
        self.mam_resid = Mamba(model_args, hidden)
        
        # --- FUSION LAYER (FIXED) ---
        # The Mamba backbone outputs logits of size 'vocab_size' (34).
        # Two branches = 34 + 34 = 68.
        # The GNN expects nodes = 'vocab_size' (34).
        # Therefore: Linear(vocab_size * 2 -> vocab_size)
        self.fusion = nn.Linear(model_args.vocab_size * 2, model_args.vocab_size)

        # --- GRAPH COMPONENT (GNN) ---
        self.cheb_k = cheb_k
        self.gamma = nn.Parameter(torch.tensor(1.))
        self.adj = nn.Parameter(torch.randn(model_args.vocab_size, embed), requires_grad=True)
        self.embed_w = nn.Parameter(torch.randn(embed, embed), requires_grad=True)
        self.weights_pool = nn.Parameter(torch.FloatTensor(embed, cheb_k, inp, out))
        self.bias_pool = nn.Parameter(torch.FloatTensor(embed, out))
        
        self.proj = nn.Linear(model_args.vocab_size, 1)
    
    def gaussian_kernel_graph(self, E_A, x, gamma=1.0):
        x_mean = torch.mean(x, dim=0)
        N = E_A.size(0)
        E_A_expanded = E_A.unsqueeze(0).expand(N, N, -1)
        E_A_T_expanded = E_A.unsqueeze(1).expand(N, N, -1)
        distance_matrix = torch.sum((E_A_expanded - E_A_T_expanded)**2, dim=2)
        A = torch.exp(-gamma * distance_matrix)
        dr = nn.Dropout(0.35)
        A = F.softmax(A, dim=1)
        return dr(A)
    
    def forward(self, input_ids):
        """
        Args:
            input_ids: [Batch, Seq_Len, Num_Nodes]
        """
        # 1. FFT Decomposition
        x_trend, x_resid = self.decomp(input_ids)
        
        # 2. Process Branches
        # Outputs are [Batch, Seq, Vocab_Size] (e.g., 34)
        out_trend = self.mam_trend(x_trend)  
        out_resid = self.mam_resid(x_resid)  
        
        # 3. Fusion
        # Concatenate: [Batch, Seq, 68]
        x_combined = torch.cat([out_trend, out_resid], dim=-1)
        # Project: [Batch, Seq, 34]
        xx = self.fusion(x_combined)
        
        # 4. Graph Neural Network
        ADJ = self.gaussian_kernel_graph(self.adj, xx, gamma=self.gamma)
        device = input_ids.device
        I = torch.eye(input_ids.size(2), device=device)
        
        support_set = [I, ADJ]
        for k in range(2, self.cheb_k):
            support_set.append(torch.matmul(2 * ADJ, support_set[-1]) - support_set[-2])
        
        supports = torch.stack(support_set, dim=0)
        
        # xx permute: [Batch, 34, Seq]
        # supports: [K, 34, 34]
        weights = torch.einsum('nd,dkio->nkio', self.adj, self.weights_pool)
        bias = torch.matmul(self.adj, self.bias_pool)
        
        x_g = torch.einsum("knm,bmc->bknc", supports, xx.permute(0, 2, 1))
        x_g = x_g.permute(0, 2, 1, 3)
        out = torch.einsum('bnki,nkio->bno', x_g, weights) + bias
        
        return self.proj(out.permute(0, 2, 1))
