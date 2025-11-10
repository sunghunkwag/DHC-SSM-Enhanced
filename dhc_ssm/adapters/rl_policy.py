"""
Reinforcement Learning adapters for DHC-SSM.

Provides policy and value function adapters for RL control tasks.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple
from collections import deque

from ..core.model import DHCSSMModel, DHCSSMConfig


class RLPolicyAdapter(nn.Module):
    """
    Improved adapter to use DHC-SSM as a policy network for RL tasks.
    
    Features:
    - Proper feature extraction for 1D observations
    - Temporal context buffering
    - Adaptive architecture based on observation dimensions
    """
    
    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        config: Optional[DHCSSMConfig] = None,
        sequence_length: int = 1,
        use_temporal_context: bool = False
    ):
        super().__init__()
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.sequence_length = sequence_length
        self.use_temporal_context = use_temporal_context
        
        # Create config if not provided
        if config is None:
            config = DHCSSMConfig(
                input_channels=1,
                hidden_dim=64,
                state_dim=64,
                output_dim=action_dim
            )
        else:
            config.input_channels = 1
            config.output_dim = action_dim
        
        # Determine optimal reshaping strategy
        self.obs_reshape_dim = self._compute_reshape_dim(observation_dim)
        self.pad_size = self.obs_reshape_dim ** 2 - observation_dim
        
        # Feature extraction network
        self.feature_extractor = nn.Sequential(
            nn.Linear(observation_dim, 128),
            nn.ReLU(),
            nn.Linear(128, self.obs_reshape_dim ** 2)
        )
        
        # DHC-SSM model
        self.dhc_model = DHCSSMModel(config)
        
        # Action head with proper initialization
        self.action_head = nn.Sequential(
            nn.Linear(action_dim, action_dim),
            nn.Tanh()  # Bound actions to [-1, 1]
        )
        
        # Temporal context buffer (if enabled)
        if use_temporal_context:
            self.context_buffer = deque(maxlen=sequence_length)
        
        self._initialize_weights()
    
    def _compute_reshape_dim(self, obs_dim: int) -> int:
        """
        Compute optimal reshape dimension for observations.
        Ensures minimum size of 4x4 for CNN processing.
        """
        sqrt_dim = int(np.ceil(np.sqrt(obs_dim)))
        # Ensure minimum dimension of 4 for proper CNN processing
        return max(4, sqrt_dim)
    
    def _initialize_weights(self):
        """Initialize weights for stable training."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass converting observations to actions.
        
        Args:
            obs: Observation tensor of shape (batch_size, observation_dim)
            
        Returns:
            Action tensor of shape (batch_size, action_dim)
        """
        batch_size = obs.shape[0]
        
        # Extract features and reshape
        features = self.feature_extractor(obs)
        
        # Reshape to image format: (batch, 1, H, W)
        obs_reshaped = features.view(batch_size, 1, self.obs_reshape_dim, self.obs_reshape_dim)
        
        # Forward through DHC-SSM
        logits = self.dhc_model(obs_reshaped)
        
        # Apply action head
        actions = self.action_head(logits)
        
        return actions
    
    def reset_context(self):
        """Reset temporal context buffer."""
        if self.use_temporal_context:
            self.context_buffer.clear()


class RLValueAdapter(nn.Module):
    """
    Value function adapter using DHC-SSM for state value estimation.
    """
    
    def __init__(
        self,
        observation_dim: int,
        config: Optional[DHCSSMConfig] = None
    ):
        super().__init__()
        self.observation_dim = observation_dim
        
        # Create config if not provided
        if config is None:
            config = DHCSSMConfig(
                input_channels=1,
                hidden_dim=64,
                state_dim=64,
                output_dim=1  # Single value output
            )
        else:
            config.input_channels = 1
            config.output_dim = 1
        
        # Determine optimal reshaping strategy
        self.obs_reshape_dim = max(4, int(np.ceil(np.sqrt(observation_dim))))
        
        # Feature extraction network
        self.feature_extractor = nn.Sequential(
            nn.Linear(observation_dim, 128),
            nn.ReLU(),
            nn.Linear(128, self.obs_reshape_dim ** 2)
        )
        
        # DHC-SSM model
        self.dhc_model = DHCSSMModel(config)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights for stable training."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to estimate state value.
        
        Args:
            obs: Observation tensor of shape (batch_size, observation_dim)
            
        Returns:
            Value tensor of shape (batch_size, 1)
        """
        batch_size = obs.shape[0]
        
        # Extract features and reshape
        features = self.feature_extractor(obs)
        
        # Reshape to image format: (batch, 1, H, W)
        obs_reshaped = features.view(batch_size, 1, self.obs_reshape_dim, self.obs_reshape_dim)
        
        # Forward through DHC-SSM
        value = self.dhc_model(obs_reshaped)
        
        return value


class RLActorCriticAdapter(nn.Module):
    """
    Actor-Critic adapter combining policy and value networks.
    Shares the DHC-SSM backbone for efficiency.
    """
    
    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        config: Optional[DHCSSMConfig] = None,
        shared_backbone: bool = True
    ):
        super().__init__()
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.shared_backbone = shared_backbone
        
        # Create config if not provided
        if config is None:
            config = DHCSSMConfig(
                input_channels=1,
                hidden_dim=64,
                state_dim=64,
                output_dim=128  # Shared feature dimension
            )
        
        # Determine optimal reshaping strategy
        self.obs_reshape_dim = max(4, int(np.ceil(np.sqrt(observation_dim))))
        
        # Feature extraction network
        self.feature_extractor = nn.Sequential(
            nn.Linear(observation_dim, 128),
            nn.ReLU(),
            nn.Linear(128, self.obs_reshape_dim ** 2)
        )
        
        if shared_backbone:
            # Shared DHC-SSM backbone
            self.backbone = DHCSSMModel(config)
            
            # Separate heads for actor and critic
            self.actor_head = nn.Sequential(
                nn.Linear(config.output_dim, action_dim),
                nn.Tanh()
            )
            
            self.critic_head = nn.Linear(config.output_dim, 1)
        else:
            # Separate networks for actor and critic
            actor_config = DHCSSMConfig(
                input_channels=1,
                hidden_dim=config.hidden_dim,
                state_dim=config.state_dim,
                output_dim=action_dim
            )
            critic_config = DHCSSMConfig(
                input_channels=1,
                hidden_dim=config.hidden_dim,
                state_dim=config.state_dim,
                output_dim=1
            )
            
            self.actor = DHCSSMModel(actor_config)
            self.critic = DHCSSMModel(critic_config)
            
            self.actor_head = nn.Tanh()
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights for stable training."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(
        self,
        obs: torch.Tensor,
        return_value: bool = True
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass for actor-critic.
        
        Args:
            obs: Observation tensor of shape (batch_size, observation_dim)
            return_value: Whether to return value estimate
            
        Returns:
            Tuple of (actions, values) where values is None if return_value=False
        """
        batch_size = obs.shape[0]
        
        # Extract features and reshape
        features = self.feature_extractor(obs)
        obs_reshaped = features.view(batch_size, 1, self.obs_reshape_dim, self.obs_reshape_dim)
        
        if self.shared_backbone:
            # Forward through shared backbone
            shared_features = self.backbone(obs_reshaped)
            
            # Actor output
            actions = self.actor_head(shared_features)
            
            # Critic output
            if return_value:
                values = self.critic_head(shared_features)
            else:
                values = None
        else:
            # Separate forward passes
            actor_output = self.actor(obs_reshaped)
            actions = self.actor_head(actor_output)
            
            if return_value:
                values = self.critic(obs_reshaped)
            else:
                values = None
        
        return actions, values
    
    def get_action(self, obs: torch.Tensor) -> torch.Tensor:
        """Get action without value estimate."""
        actions, _ = self.forward(obs, return_value=False)
        return actions
    
    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        """Get value estimate only."""
        _, values = self.forward(obs, return_value=True)
        return values
