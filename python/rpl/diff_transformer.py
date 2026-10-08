#!/usr/bin/env python3
"""
Differential Transformer (Diff Transformer) Implementation in RPL
===================================================================
Reference Paper:
  "Differential Transformer"
  Tianzhu Ye, Li Dong, Yuqing Xia, Yutao Sun, Yi Zhu, Gao Huang, Furu Wei
  Microsoft Research & Tsinghua University
  arXiv:2410.05258 (ICLR 2025 Oral Presentation)

Key Mechanism:
  DiffAttn(X) = (softmax(Q1 @ K1.T / sqrt(d)) - lambda * softmax(Q2 @ K2.T / sqrt(d))) @ V

  By subtracting two separate softmax attention maps, Differential Attention
  cancels out common-mode attention noise, effectively focusing attention on
  critical tokens while eliminating hallucination and irrelevant context.
"""

import math
import numpy as np
from .core import Tensor
from . import nn

class MultiHeadDifferentialAttention(nn.Module):
    """
    Multi-Head Differential Attention Layer.
    Splits Q and K into two sub-spaces (Q1, Q2) and (K1, K2),
    computes two attention maps, and subtracts them with learnable/scheduled lambda.
    """
    def __init__(self, d_model, n_heads=4, layer_idx=1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.layer_idx = layer_idx

        # Q and K projected to 2 * d_model (for Q1, Q2 and K1, K2)
        self.q_proj = nn.Linear(d_model, d_model * 2)
        self.k_proj = nn.Linear(d_model, d_model * 2)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        # Lambda initialization following Eq. (lambda_init) in Ye et al. (2024):
        # lambda_init(l) = 0.8 - 0.6 * exp(-0.3 * (l - 1))
        self.init_lambda = 0.8 - 0.6 * math.exp(-0.3 * (layer_idx - 1))
        self.lambda_param = self.init_lambda

    def forward(self, x, mask=None):
        """
        Args:
            x: Tensor of shape (seq_len, d_model)
            mask: Optional causal or padding mask
        Returns:
            Tensor of shape (seq_len, d_model)
        """
        seq_len, d_model = x.shape

        # Project Q, K, V
        q_full = self.q_proj(x)  # (seq_len, 2 * d_model)
        k_full = self.k_proj(x)  # (seq_len, 2 * d_model)
        v = self.v_proj(x)       # (seq_len, d_model)

        # Split Q -> Q1, Q2 and K -> K1, K2
        # In RPL, we can slice numpy-backed tensors or reshape
        q_np = q_full.data
        k_np = k_full.data

        q1 = Tensor(q_np[:, :d_model])
        q2 = Tensor(q_np[:, d_model:])
        k1 = Tensor(k_np[:, :d_model])
        k2 = Tensor(k_np[:, d_model:])

        # Compute Attention Map 1: softmax(Q1 @ K1^T / sqrt(d))
        scores1 = (q1 @ k1.transpose(1, 0)) * self.scale
        if mask is not None:
            scores1 = scores1 + mask
        scores1.softmax_()

        # Compute Attention Map 2: softmax(Q2 @ K2^T / sqrt(d))
        scores2 = (q2 @ k2.transpose(1, 0)) * self.scale
        if mask is not None:
            scores2 = scores2 + mask
        scores2.softmax_()

        # Differential Noise Cancellation: A_diff = A1 - lambda * A2
        lambda_scaled_scores2 = scores2 * self.lambda_param
        diff_attn = scores1 - lambda_scaled_scores2

        # Multiply by V
        attn_out = diff_attn @ v
        out = self.out_proj(attn_out)

        # Store attention maps for visualization / telemetry
        self.last_attn_map1 = scores1
        self.last_attn_map2 = scores2
        self.last_diff_map = diff_attn

        return out

class DifferentialTransformerBlock(nn.Module):
    """
    Differential Transformer Block:
      x = x + DiffAttn(LayerNorm(x))
      x = x + SwiGLU_FFN(LayerNorm(x))
    """
    def __init__(self, d_model, intermediate_dim=None, n_heads=4, layer_idx=1):
        super().__init__()
        if intermediate_dim is None:
            intermediate_dim = d_model * 4

        self.ln1 = nn.LayerNorm(d_model)
        self.diff_attn = MultiHeadDifferentialAttention(d_model, n_heads, layer_idx=layer_idx)

        self.ln2 = nn.LayerNorm(d_model)
        # SwiGLU FFN
        self.ffn_gate = nn.Linear(d_model, intermediate_dim)
        self.ffn_up   = nn.Linear(d_model, intermediate_dim)
        self.ffn_down = nn.Linear(intermediate_dim, d_model)

    def forward(self, x, mask=None):
        # Differential attention branch
        norm_x1 = self.ln1(x)
        attn_out = self.diff_attn(norm_x1, mask=mask)
        x = x + attn_out

        # SwiGLU FFN branch
        norm_x2 = self.ln2(x)
        gate = self.ffn_gate(norm_x2).silu()
        up   = self.ffn_up(norm_x2)
        ffn_out = self.ffn_down(gate * up)
        x = x + ffn_out

        return x

class DifferentialTransformerLM(nn.Module):
    """
    Full Differential Transformer Language Model.
    """
    def __init__(self, vocab_size, d_model=128, n_layers=4, n_heads=4, intermediate_dim=256):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers

        self.embed = nn.Embedding(vocab_size, d_model)
        self.blocks = [
            DifferentialTransformerBlock(d_model, intermediate_dim, n_heads, layer_idx=i+1)
            for i in range(n_layers)
        ]
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids, mask=None):
        """
        Forward pass.
        Args:
            input_ids: 1D Tensor of token ids (seq_len,)
        Returns:
            logits: Tensor of shape (seq_len, vocab_size)
        """
        x = self.embed(input_ids)
        for block in self.blocks:
            x = block(x, mask=mask)
        x = self.ln_f(x)
        logits = self.head(x)
        return logits
