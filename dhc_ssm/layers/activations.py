"""Activation functions for DHC-SSM Architecture."""

import torch
import torch.nn as nn
from typing import Union, Callable


def get_activation(name: str) -> nn.Module:
    """
    Get activation function by name.
    
    Args:
        name: Activation function name
        
    Returns:
        Activation module
        
    Raises:
        ValueError: If activation name is unknown
    """
    activations = {
        "relu": nn.ReLU(),
        "gelu": nn.GELU(),
        "silu": nn.SiLU(),
        "swish": nn.SiLU(),  # Swish is same as SiLU
        "tanh": nn.Tanh(),
        "sigmoid": nn.Sigmoid(),
        "leaky_relu": nn.LeakyReLU(0.01),
        "elu": nn.ELU(),
        "selu": nn.SELU(),
        "mish": Mish(),
        "identity": nn.Identity(),
        "none": nn.Identity(),
    }
    
    name = name.lower()
    if name not in activations:
        raise ValueError(
            f"Unknown activation: {name}. "
            f"Available: {list(activations.keys())}"
        )
    
    return activations[name]


class Mish(nn.Module):
    """
    Mish activation function.
    
    Mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))
    """
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor
            
        Returns:
            Activated tensor
        """
        return x * torch.tanh(nn.functional.softplus(x))


class GLU(nn.Module):
    """
    Gated Linear Unit.
    
    Splits input in half and applies gating mechanism.
    """
    
    def __init__(self, dim: int = -1):
        """
        Initialize GLU.
        
        Args:
            dim: Dimension to split along
        """
        super().__init__()
        self.dim = dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor
            
        Returns:
            Gated output
        """
        a, b = x.chunk(2, dim=self.dim)
        return a * torch.sigmoid(b)


class Swish(nn.Module):
    """
    Swish activation function (same as SiLU).
    
    Swish(x) = x * sigmoid(x)
    """
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor
            
        Returns:
            Activated tensor
        """
        return x * torch.sigmoid(x)
