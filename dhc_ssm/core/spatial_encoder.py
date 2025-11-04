"""
Spatial Encoder for DHC-SSM Architecture.

Layer 1: Enhanced CNN with O(n) complexity for spatial feature extraction.
"""

import torch
import torch.nn as nn
from typing import List, Optional
import logging

from dhc_ssm.layers.residual import ConvResidualBlock
from dhc_ssm.layers.attention import EfficientAttention
from dhc_ssm.utils.validation import validate_tensor, validate_shape

logger = logging.getLogger(__name__)


class SpatialEncoder(nn.Module):
    """
    Spatial feature encoder using enhanced CNN architecture.
    
    Features:
    - Residual connections for better gradient flow
    - Efficient attention mechanisms (optional)
    - Adaptive pooling for flexible input sizes
    - O(n) complexity maintained
    """
    
    def __init__(
        self,
        input_channels: int = 3,
        channels: List[int] = None,
        output_dim: int = 128,
        kernel_sizes: List[int] = None,
        strides: List[int] = None,
        use_residual: bool = True,
        use_attention: bool = False,
        dropout: float = 0.1,
        activation: str = "relu",
    ):
        """
        Initialize spatial encoder.
        
        Args:
            input_channels: Number of input channels
            channels: List of channel dimensions for each layer
            output_dim: Output feature dimension
            kernel_sizes: Kernel sizes for each conv layer
            strides: Strides for each conv layer
            use_residual: Whether to use residual connections
            use_attention: Whether to use attention mechanisms
            dropout: Dropout probability
            activation: Activation function name
        """
        super().__init__()
        
        if channels is None:
            channels = [64, 128, 256]
        if kernel_sizes is None:
            kernel_sizes = [3] * len(channels)
        if strides is None:
            strides = [2] * len(channels)
        
        assert len(channels) == len(kernel_sizes) == len(strides), \
            "channels, kernel_sizes, and strides must have same length"
        
        self.input_channels = input_channels
        self.output_dim = output_dim
        self.use_attention = use_attention
        
        # Build convolutional layers
        layers = []
        in_ch = input_channels
        
        for i, (out_ch, kernel, stride) in enumerate(zip(channels, kernel_sizes, strides)):
            padding = kernel // 2
            
            if use_residual and in_ch == out_ch and stride == 1:
                # Use residual block when dimensions match
                layers.append(
                    ConvResidualBlock(
                        in_channels=in_ch,
                        out_channels=out_ch,
                        kernel_size=kernel,
                        stride=stride,
                        padding=padding,
                        activation=activation,
                    )
                )
            else:
                # Regular conv block
                layers.extend([
                    nn.Conv2d(in_ch, out_ch, kernel, stride, padding, bias=False),
                    nn.BatchNorm2d(out_ch),
                    self._get_activation(activation),
                ])
            
            # Add dropout
            if dropout > 0:
                layers.append(nn.Dropout2d(dropout))
            
            in_ch = out_ch
        
        self.conv_layers = nn.Sequential(*layers)
        
        # Adaptive pooling to fixed size
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        
        # Calculate flattened dimension
        self.flatten_dim = channels[-1] * 4 * 4
        
        # Projection to output dimension
        self.projection = nn.Sequential(
            nn.Linear(self.flatten_dim, output_dim * 2),
            self._get_activation(activation),
            nn.Dropout(dropout),
            nn.Linear(output_dim * 2, output_dim),
        )
        
        # Optional attention
        if use_attention:
            self.attention = EfficientAttention(output_dim, num_heads=4, dropout=dropout)
        else:
            self.attention = None
        
        logger.info(
            f"SpatialEncoder initialized: {input_channels} -> {channels} -> {output_dim}"
        )
    
    def _get_activation(self, name: str) -> nn.Module:
        """Get activation function."""
        activations = {
            "relu": nn.ReLU(inplace=True),
            "gelu": nn.GELU(),
            "silu": nn.SiLU(),
        }
        return activations.get(name, nn.ReLU(inplace=True))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor (batch, channels, height, width)
            
        Returns:
            Encoded features (batch, output_dim)
        """
        # Validate input
        if x.dim() != 4:
            raise ValueError(f"Expected 4D input (batch, channels, H, W), got {x.dim()}D")
        
        batch_size = x.shape[0]
        
        # Convolutional encoding
        features = self.conv_layers(x)  # (batch, channels[-1], H', W')
        
        # Adaptive pooling
        features = self.adaptive_pool(features)  # (batch, channels[-1], 4, 4)
        
        # Flatten
        features = features.view(batch_size, -1)  # (batch, flatten_dim)
        
        # Project to output dimension
        features = self.projection(features)  # (batch, output_dim)
        
        # Optional attention (treating as sequence of length 1)
        if self.attention is not None:
            features = features.unsqueeze(1)  # (batch, 1, output_dim)
            features = self.attention(features)  # (batch, 1, output_dim)
            features = features.squeeze(1)  # (batch, output_dim)
        
        return features
    
    def get_output_dim(self) -> int:
        """Get output feature dimension."""
        return self.output_dim
    
    def get_complexity(self) -> str:
        """Get computational complexity."""
        return "O(n) where n = height * width"
