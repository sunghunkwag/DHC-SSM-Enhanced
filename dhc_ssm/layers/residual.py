"""Residual connection layers for DHC-SSM Architecture."""

import torch
import torch.nn as nn
from typing import Optional, Callable


class ResidualBlock(nn.Module):
    """
    Residual block with skip connection.
    
    Implements: output = activation(layer(x) + x)
    """
    
    def __init__(
        self,
        layer: nn.Module,
        dropout: float = 0.0,
        activation: Optional[Callable] = None,
        use_layer_norm: bool = True,
    ):
        """
        Initialize residual block.
        
        Args:
            layer: Main transformation layer
            dropout: Dropout probability
            activation: Activation function (None for no activation)
            use_layer_norm: Whether to use layer normalization
        """
        super().__init__()
        self.layer = layer
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.activation = activation if activation is not None else nn.Identity()
        self.norm = nn.LayerNorm(layer.out_features) if use_layer_norm else nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with residual connection.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor with residual connection
        """
        residual = x
        x = self.layer(x)
        x = self.dropout(x)
        x = x + residual
        x = self.norm(x)
        x = self.activation(x)
        return x


class PreNormResidual(nn.Module):
    """
    Pre-normalization residual block.
    
    Implements: output = x + layer(norm(x))
    """
    
    def __init__(
        self,
        dim: int,
        layer: nn.Module,
        dropout: float = 0.0,
    ):
        """
        Initialize pre-norm residual block.
        
        Args:
            dim: Feature dimension
            layer: Main transformation layer
            dropout: Dropout probability
        """
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layer = layer
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with pre-normalization.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor
        """
        return x + self.dropout(self.layer(self.norm(x)))


class ConvResidualBlock(nn.Module):
    """
    Convolutional residual block for spatial processing.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        use_batch_norm: bool = True,
        activation: str = "relu",
    ):
        """
        Initialize convolutional residual block.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            kernel_size: Convolution kernel size
            stride: Convolution stride
            padding: Convolution padding
            use_batch_norm: Whether to use batch normalization
            activation: Activation function name
        """
        super().__init__()
        
        # Main path
        self.conv1 = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=not use_batch_norm
        )
        self.bn1 = nn.BatchNorm2d(out_channels) if use_batch_norm else nn.Identity()
        
        self.conv2 = nn.Conv2d(
            out_channels, out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            bias=not use_batch_norm
        )
        self.bn2 = nn.BatchNorm2d(out_channels) if use_batch_norm else nn.Identity()
        
        # Activation
        if activation == "relu":
            self.activation = nn.ReLU(inplace=True)
        elif activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "silu":
            self.activation = nn.SiLU()
        else:
            self.activation = nn.Identity()
        
        # Skip connection
        if in_channels != out_channels or stride != 1:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels) if use_batch_norm else nn.Identity()
            )
        else:
            self.skip = nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor (batch, channels, height, width)
            
        Returns:
            Output tensor
        """
        residual = self.skip(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out = out + residual
        out = self.activation(out)
        
        return out
