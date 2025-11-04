"""Attention mechanism layers for DHC-SSM Architecture."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import math


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention mechanism.
    
    Efficient implementation with O(n²) complexity for attention,
    but can be used selectively in the architecture.
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
    ):
        """
        Initialize multi-head attention.
        
        Args:
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            dropout: Dropout probability
            bias: Whether to use bias in projections
        """
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Projections
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        query: torch.Tensor,
        key: Optional[torch.Tensor] = None,
        value: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            query: Query tensor (batch, seq_len, embed_dim)
            key: Key tensor (batch, seq_len, embed_dim) or None for self-attention
            value: Value tensor (batch, seq_len, embed_dim) or None for self-attention
            mask: Attention mask (batch, seq_len, seq_len) or None
            
        Returns:
            Output tensor (batch, seq_len, embed_dim)
        """
        if key is None:
            key = query
        if value is None:
            value = query
        
        batch_size, seq_len, _ = query.shape
        
        # Project and reshape
        q = self.q_proj(query).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        output = torch.matmul(attn_weights, v)
        
        # Reshape and project
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        output = self.out_proj(output)
        
        return output


class SelfAttention(nn.Module):
    """
    Self-attention layer (simplified version).
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        dropout: float = 0.0,
    ):
        """
        Initialize self-attention.
        
        Args:
            dim: Feature dimension
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()
        self.attention = MultiHeadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
        )
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor (batch, seq_len, dim)
            mask: Optional attention mask
            
        Returns:
            Output tensor (batch, seq_len, dim)
        """
        return self.attention(x, x, x, mask)


class EfficientAttention(nn.Module):
    """
    Efficient attention with linear complexity approximation.
    
    Uses kernel-based attention for O(n) complexity.
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        dropout: float = 0.0,
    ):
        """
        Initialize efficient attention.
        
        Args:
            dim: Feature dimension
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()
        assert dim % num_heads == 0
        
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with linear complexity.
        
        Args:
            x: Input tensor (batch, seq_len, dim)
            
        Returns:
            Output tensor (batch, seq_len, dim)
        """
        batch_size, seq_len, _ = x.shape
        
        # Project
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # Reshape for multi-head
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Apply kernel function (ReLU for linear attention)
        q = F.relu(q)
        k = F.relu(k)
        
        # Linear attention: O(n) complexity
        # Compute k^T v first (head_dim x head_dim)
        kv = torch.matmul(k.transpose(-2, -1), v)  # (batch, heads, head_dim, head_dim)
        
        # Then compute q (k^T v)
        output = torch.matmul(q, kv)  # (batch, heads, seq_len, head_dim)
        
        # Normalize
        k_sum = k.sum(dim=-2, keepdim=True)  # (batch, heads, 1, head_dim)
        normalizer = torch.matmul(q, k_sum.transpose(-2, -1))  # (batch, heads, seq_len, 1)
        output = output / (normalizer + 1e-6)
        
        # Reshape and project
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.dim)
        output = self.out_proj(output)
        output = self.dropout(output)
        
        return output
