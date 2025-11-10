"""
DHC-SSM Model v3.1

Simplified architecture with proven components for stable training.
Uses real CIFAR-10 data for validation.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class SpatialEncoder(nn.Module):
    """CNN-based spatial feature extraction with adaptive pooling for small inputs."""
    
    def __init__(self, input_channels=3, hidden_dim=64):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # First conv layer (always applied)
        self.conv1 = nn.Conv2d(input_channels, hidden_dim, 3, padding=1)
        self.relu1 = nn.ReLU()
        
        # Second conv layer (always applied)
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim * 2, 3, padding=1)
        self.relu2 = nn.ReLU()
        
        # Third conv layer (always applied)
        self.conv3 = nn.Conv2d(hidden_dim * 2, hidden_dim * 4, 3, padding=1)
        self.relu3 = nn.ReLU()
        
        # Adaptive pooling (works for any input size)
        self.adaptive_pool = nn.AdaptiveAvgPool2d(1)
    
    def forward(self, x):
        # Get input dimensions
        _, _, h, w = x.shape
        
        # First conv block
        x = self.conv1(x)
        x = self.relu1(x)
        # Only pool if dimensions are large enough (>= 4)
        if h >= 4 and w >= 4:
            x = nn.functional.max_pool2d(x, 2)
            h, w = h // 2, w // 2
        
        # Second conv block
        x = self.conv2(x)
        x = self.relu2(x)
        # Only pool if dimensions are large enough (>= 4)
        if h >= 4 and w >= 4:
            x = nn.functional.max_pool2d(x, 2)
            h, w = h // 2, w // 2
        
        # Third conv block
        x = self.conv3(x)
        x = self.relu3(x)
        
        # Always use adaptive pooling to get fixed output size
        x = self.adaptive_pool(x)
        
        return x.squeeze(-1).squeeze(-1)


class TemporalSSM(nn.Module):
    """Simplified State Space Model for temporal processing."""
    
    def __init__(self, hidden_dim=256, state_dim=64):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.state_dim = state_dim
        
        self.A = nn.Parameter(torch.randn(state_dim, state_dim) * 0.01)
        self.B = nn.Parameter(torch.randn(state_dim, hidden_dim) * 0.01)
        self.C = nn.Parameter(torch.randn(hidden_dim, state_dim) * 0.01)
        self.D = nn.Parameter(torch.zeros(hidden_dim, hidden_dim))
    
    def forward(self, x):
        batch_size = x.size(0)
        state = torch.zeros(batch_size, self.state_dim, device=x.device)
        
        state = torch.tanh(state @ self.A.t() + x @ self.B.t())
        output = state @ self.C.t() + x @ self.D.t()
        
        return output


class DHCSSMModel(nn.Module):
    """
    DHC-SSM v3.1: Simplified three-stage architecture.
    
    Architecture:
    1. Spatial Encoder (CNN) - O(1) per position
    2. Temporal SSM - O(n) complexity
    3. Classification Head - O(1)
    
    Total complexity: O(n)
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Use config attributes or set defaults
        hidden_dim = getattr(config, 'hidden_dim', 64)
        state_dim = getattr(config, 'ssm_state_dim', 64)
        input_channels = getattr(config, 'input_channels', 3)
        output_dim = getattr(config, 'output_dim', 10)
        
        self.spatial_encoder = SpatialEncoder(
            input_channels=input_channels,
            hidden_dim=hidden_dim
        )
        
        self.temporal_ssm = TemporalSSM(
            hidden_dim=hidden_dim * 4,
            state_dim=state_dim
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim * 2, output_dim)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Proper weight initialization for stable training."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x, return_features=False):
        spatial_features = self.spatial_encoder(x)
        temporal_features = self.temporal_ssm(spatial_features)
        logits = self.classifier(temporal_features)
        
        if return_features:
            features = {
                'spatial': spatial_features,
                'temporal': temporal_features,
                'logits': logits
            }
            return logits, features
        
        return logits
    
    def compute_loss(self, logits, targets):
        """Standard cross-entropy loss."""
        return nn.functional.cross_entropy(logits, targets)
    
    def train_step(self, batch, optimizer):
        """Single training step."""
        x, targets = batch
        
        optimizer.zero_grad()
        logits = self(x)
        loss = self.compute_loss(logits, targets)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        optimizer.step()
        
        preds = logits.argmax(dim=1)
        accuracy = (preds == targets).float().mean()
        
        return {
            'loss': loss.item(),
            'accuracy': accuracy.item()
        }
    
    def evaluate_step(self, batch):
        """Single evaluation step."""
        x, targets = batch
        
        with torch.no_grad():
            logits = self(x)
            loss = self.compute_loss(logits, targets)
            preds = logits.argmax(dim=1)
            accuracy = (preds == targets).float().mean()
        
        return {
            'loss': loss.item(),
            'accuracy': accuracy.item()
        }
    
    @property
    def num_parameters(self):
        """Get total number of parameters."""
        return sum(p.numel() for p in self.parameters())
    
    @property
    def device(self):
        """Get model device."""
        return next(self.parameters()).device
    
    def get_diagnostics(self):
        """Get model diagnostics."""
        return {
            'architecture': 'DHC-SSM v3.1',
            'complexity': 'O(n)',
            'num_parameters': self.num_parameters,
            'device': str(self.device),
            'layers': {
                'spatial_encoder': {
                    'type': 'CNN',
                    'output_dim': self.config.hidden_dim * 4 if hasattr(self.config, 'hidden_dim') else 256,
                    'complexity': 'O(n)'
                },
                'temporal_ssm': {
                    'type': 'State Space Model',
                    'output_dim': self.config.hidden_dim * 4 if hasattr(self.config, 'hidden_dim') else 256,
                    'complexity': 'O(n)'
                },
                'classifier': {
                    'type': 'MLP',
                    'output_dim': self.config.output_dim if hasattr(self.config, 'output_dim') else 10,
                    'complexity': 'O(1)'
                }
            }
        }


class DHCSSMConfig:
    """Configuration for DHC-SSM v3.1"""
    def __init__(
        self,
        input_channels: int = 3,
        hidden_dim: int = 64,
        state_dim: int = 64,
        output_dim: int = 10
    ):
        self.input_channels = input_channels
        self.hidden_dim = hidden_dim
        self.state_dim = state_dim
        self.output_dim = output_dim
