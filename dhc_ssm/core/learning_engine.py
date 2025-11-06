"""
Learning Engine for DHC-SSM v3.1

Provides deterministic optimization with gradient clipping.
"""

import torch
from torch.optim import AdamW
from typing import Iterable, Optional


class DeterministicOptimizer:
    """
    Deterministic optimizer wrapper for stable training.
    
    Wraps PyTorch optimizers with additional features:
    - Gradient clipping
    - Deterministic updates
    - Loss tracking
    """
    
    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        gradient_clip: float = 1.0,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
    ):
        """
        Initialize deterministic optimizer.
        
        Args:
            params: Model parameters to optimize
            lr: Learning rate
            weight_decay: Weight decay coefficient
            gradient_clip: Maximum gradient norm
            betas: AdamW beta parameters
            eps: AdamW epsilon
        """
        self.gradient_clip = gradient_clip
        self.optimizer = AdamW(
            params,
            lr=lr,
            weight_decay=weight_decay,
            betas=betas,
            eps=eps,
        )
    
    def zero_grad(self):
        """Zero out gradients."""
        self.optimizer.zero_grad()
    
    def step(self):
        """Perform optimization step with gradient clipping."""
        if self.gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                self.get_parameters(),
                self.gradient_clip
            )
        self.optimizer.step()
    
    def get_parameters(self):
        """Get all parameters from optimizer."""
        params = []
        for param_group in self.optimizer.param_groups:
            params.extend(param_group['params'])
        return params
    
    def state_dict(self):
        """Get optimizer state dict."""
        return self.optimizer.state_dict()
    
    def load_state_dict(self, state_dict):
        """Load optimizer state dict."""
        self.optimizer.load_state_dict(state_dict)
    
    @property
    def param_groups(self):
        """Access optimizer param groups."""
        return self.optimizer.param_groups
