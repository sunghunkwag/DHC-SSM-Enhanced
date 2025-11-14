"""
Simplified RL adapters for DHC-SSM v3.1.

This version uses a simpler architecture that bypasses the spatial encoder
for 1D observation spaces, making it more suitable for MuJoCo tasks.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple

from ..core.model import DHCSSMConfig


class SimpleRLPolicy(nn.Module):
    """
    Simplified policy network that uses SSM-inspired architecture
    without forcing 1D observations through spatial convolutions.
    """
    
    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        state_dim: int = 64
    ):
        super().__init__()
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.state_dim = state_dim
        
        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(observation_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh()
        )
        
        # State space layer (simplified SSM)
        self.state_proj = nn.Linear(hidden_dim, state_dim)
        self.state_update = nn.Linear(state_dim, state_dim)
        
        # Output layers
        self.output_proj = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()  # Bound actions to [-1, 1]
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights for stable RL training."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Smaller initialization for stability
                nn.init.orthogonal_(m.weight, gain=0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            obs: (batch_size, observation_dim)
        
        Returns:
            actions: (batch_size, action_dim)
        """
        # Project input
        h = self.input_proj(obs)
        
        # State space transformation
        state = self.state_proj(h)
        state = self.state_update(state)
        
        # Output projection
        actions = self.output_proj(state)
        
        return actions


class SimpleRLValue(nn.Module):
    """
    Simplified value network.
    """
    
    def __init__(
        self,
        observation_dim: int,
        hidden_dim: int = 128,
        state_dim: int = 64
    ):
        super().__init__()
        self.observation_dim = observation_dim
        self.hidden_dim = hidden_dim
        self.state_dim = state_dim
        
        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(observation_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh()
        )
        
        # State space layer
        self.state_proj = nn.Linear(hidden_dim, state_dim)
        self.state_update = nn.Linear(state_dim, state_dim)
        
        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights for stable RL training."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            obs: (batch_size, observation_dim)
        
        Returns:
            value: (batch_size, 1)
        """
        # Project input
        h = self.input_proj(obs)
        
        # State space transformation
        state = self.state_proj(h)
        state = self.state_update(state)
        
        # Value output
        value = self.value_head(state)
        
        return value


class SimpleRLActorCritic(nn.Module):
    """
    Actor-Critic with shared backbone.
    """
    
    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        state_dim: int = 64
    ):
        super().__init__()
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        
        # Shared backbone
        self.input_proj = nn.Sequential(
            nn.Linear(observation_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh()
        )
        
        self.state_proj = nn.Linear(hidden_dim, state_dim)
        self.state_update = nn.Linear(state_dim, state_dim)
        
        # Actor head
        self.actor_head = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()
        )
        
        # Critic head
        self.critic_head = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights for stable RL training."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(
        self,
        obs: torch.Tensor,
        return_value: bool = True
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.
        
        Args:
            obs: (batch_size, observation_dim)
            return_value: Whether to return value
        
        Returns:
            (actions, values)
        """
        # Shared backbone
        h = self.input_proj(obs)
        state = self.state_proj(h)
        state = self.state_update(state)
        
        # Actor
        actions = self.actor_head(state)
        
        # Critic
        if return_value:
            values = self.critic_head(state)
        else:
            values = None
        
        return actions, values
    
    def get_action(self, obs: torch.Tensor) -> torch.Tensor:
        """Get action only."""
        actions, _ = self.forward(obs, return_value=False)
        return actions
    
    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        """Get value only."""
        _, values = self.forward(obs, return_value=True)
        return values
