"""Normalization layers for DHC-SSM Architecture."""

import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    """
    Layer normalization with optional bias.
    """
    
    def __init__(self, dim: int, eps: float = 1e-5, bias: bool = True):
        """
        Initialize layer normalization.
        
        Args:
            dim: Feature dimension
            eps: Small constant for numerical stability
            bias: Whether to use bias parameter
        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim)) if bias else None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor
            
        Returns:
            Normalized tensor
        """
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        x_norm = x_norm * self.weight
        if self.bias is not None:
            x_norm = x_norm + self.bias
        return x_norm


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.
    
    More efficient than LayerNorm, used in modern architectures.
    """
    
    def __init__(self, dim: int, eps: float = 1e-6):
        """
        Initialize RMS normalization.
        
        Args:
            dim: Feature dimension
            eps: Small constant for numerical stability
        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor
            
        Returns:
            Normalized tensor
        """
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        x_norm = x / rms
        return x_norm * self.weight


class GroupNorm(nn.Module):
    """
    Group normalization for convolutional layers.
    """
    
    def __init__(self, num_channels: int, num_groups: int = 32, eps: float = 1e-5):
        """
        Initialize group normalization.
        
        Args:
            num_channels: Number of channels
            num_groups: Number of groups
            eps: Small constant for numerical stability
        """
        super().__init__()
        assert num_channels % num_groups == 0, "num_channels must be divisible by num_groups"
        self.num_groups = num_groups
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor (batch, channels, height, width)
            
        Returns:
            Normalized tensor
        """
        batch, channels, height, width = x.shape
        
        # Reshape for group-wise normalization
        x = x.view(batch, self.num_groups, -1)
        
        # Normalize
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        x = (x - mean) / torch.sqrt(var + self.eps)
        
        # Reshape back
        x = x.view(batch, channels, height, width)
        
        # Apply learned parameters
        x = x * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)
        
        return x
