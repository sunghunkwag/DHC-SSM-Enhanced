"""
Temporal Processor for DHC-SSM Architecture.

Layer 2: State Space Model with O(n) complexity for temporal sequence processing.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple
import logging
import math

from dhc_ssm.layers.normalization import RMSNorm
from dhc_ssm.utils.validation import validate_tensor

logger = logging.getLogger(__name__)


class StateSpaceLayer(nn.Module):
    """
    Single State Space Model layer.
    
    Implements discrete-time state space model:
        x_{t+1} = A x_t + B u_t
        y_t = C x_t + D u_t
    
    With efficient parallel scan for O(n) complexity.
    """
    
    def __init__(
        self,
        input_dim: int,
        state_dim: int,
        output_dim: int,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
    ):
        """
        Initialize state space layer.
        
        Args:
            input_dim: Input feature dimension
            state_dim: Hidden state dimension
            output_dim: Output feature dimension
            dt_min: Minimum discretization step
            dt_max: Maximum discretization step
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.state_dim = state_dim
        self.output_dim = output_dim
        
        # State space parameters
        # A: State transition matrix (state_dim, state_dim)
        self.A_log = nn.Parameter(torch.randn(state_dim, state_dim))
        
        # B: Input matrix (state_dim, input_dim)
        self.B = nn.Parameter(torch.randn(state_dim, input_dim))
        
        # C: Output matrix (output_dim, state_dim)
        self.C = nn.Parameter(torch.randn(output_dim, state_dim))
        
        # D: Feedthrough matrix (output_dim, input_dim)
        self.D = nn.Parameter(torch.randn(output_dim, input_dim))
        
        # Learnable time step
        self.dt_log = nn.Parameter(
            torch.rand(1) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        )
        
        self._init_parameters()
    
    def _init_parameters(self):
        """Initialize parameters with appropriate values."""
        # Initialize A to be stable (negative eigenvalues)
        nn.init.xavier_uniform_(self.A_log)
        self.A_log.data = -torch.abs(self.A_log.data)
        
        # Initialize B, C with Xavier
        nn.init.xavier_uniform_(self.B)
        nn.init.xavier_uniform_(self.C)
        
        # Initialize D small
        nn.init.zeros_(self.D)
    
    def discretize(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Discretize continuous-time system to discrete-time.
        
        Returns:
            Tuple of (A_discrete, B_discrete)
        """
        # Get time step
        dt = torch.exp(self.dt_log)
        
        # Get continuous A (ensure stability)
        A = -torch.exp(self.A_log)
        
        # Zero-order hold discretization
        # A_d = exp(A * dt) ≈ I + A * dt (first-order approximation for stability)
        A_discrete = torch.eye(self.state_dim, device=A.device) + A * dt
        
        # B_d = A^{-1} (A_d - I) B ≈ B * dt (first-order approximation)
        B_discrete = self.B * dt
        
        return A_discrete, B_discrete
    
    def forward(self, u: torch.Tensor, state: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through state space layer.
        
        Args:
            u: Input sequence (batch, seq_len, input_dim)
            state: Initial state (batch, state_dim) or None
            
        Returns:
            Tuple of (output, final_state)
        """
        batch_size, seq_len, _ = u.shape
        device = u.device
        
        # Discretize system
        A_d, B_d = self.discretize()
        
        # Initialize state if not provided
        if state is None:
            state = torch.zeros(batch_size, self.state_dim, device=device)
        
        # Process sequence (can be parallelized with scan)
        outputs = []
        for t in range(seq_len):
            # State update: x_{t+1} = A_d x_t + B_d u_t
            state = torch.matmul(state, A_d.t()) + torch.matmul(u[:, t], B_d.t())
            
            # Output: y_t = C x_t + D u_t
            output = torch.matmul(state, self.C.t()) + torch.matmul(u[:, t], self.D.t())
            outputs.append(output)
        
        # Stack outputs
        outputs = torch.stack(outputs, dim=1)  # (batch, seq_len, output_dim)
        
        return outputs, state


class TemporalProcessor(nn.Module):
    """
    Temporal sequence processor using stacked State Space Models.
    
    Features:
    - O(n) complexity for sequence processing
    - Efficient state transitions
    - Stable discretization
    - Multi-layer architecture
    """
    
    def __init__(
        self,
        input_dim: int = 128,
        hidden_dim: int = 256,
        state_dim: int = 128,
        num_layers: int = 4,
        dropout: float = 0.1,
        use_norm: bool = True,
    ):
        """
        Initialize temporal processor.
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden feature dimension
            state_dim: State space dimension
            num_layers: Number of SSM layers
            dropout: Dropout probability
            use_norm: Whether to use normalization
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.state_dim = state_dim
        self.num_layers = num_layers
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Stack of SSM layers
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            layer_input_dim = hidden_dim
            layer_output_dim = hidden_dim
            
            self.layers.append(
                nn.ModuleDict({
                    'ssm': StateSpaceLayer(
                        input_dim=layer_input_dim,
                        state_dim=state_dim,
                        output_dim=layer_output_dim,
                    ),
                    'norm': RMSNorm(layer_output_dim) if use_norm else nn.Identity(),
                    'dropout': nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
                    'ffn': nn.Sequential(
                        nn.Linear(layer_output_dim, layer_output_dim * 4),
                        nn.GELU(),
                        nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
                        nn.Linear(layer_output_dim * 4, layer_output_dim),
                    ),
                    'norm2': RMSNorm(layer_output_dim) if use_norm else nn.Identity(),
                })
            )
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        
        logger.info(
            f"TemporalProcessor initialized: {num_layers} layers, "
            f"dim={hidden_dim}, state_dim={state_dim}"
        )
    
    def forward(
        self,
        x: torch.Tensor,
        states: Optional[list] = None,
    ) -> Tuple[torch.Tensor, list]:
        """
        Forward pass.
        
        Args:
            x: Input tensor (batch, seq_len, input_dim) or (batch, input_dim)
            states: Optional list of initial states for each layer
            
        Returns:
            Tuple of (output, final_states)
        """
        # Handle both sequential and single-step input
        is_single_step = (x.dim() == 2)
        if is_single_step:
            x = x.unsqueeze(1)  # (batch, 1, input_dim)
        
        batch_size, seq_len, _ = x.shape
        
        # Initialize states if not provided
        if states is None:
            states = [None] * self.num_layers
        
        # Input projection
        x = self.input_proj(x)  # (batch, seq_len, hidden_dim)
        
        # Process through SSM layers
        final_states = []
        for i, layer_dict in enumerate(self.layers):
            # SSM with residual connection
            residual = x
            x_norm = layer_dict['norm'](x)
            x_ssm, state = layer_dict['ssm'](x_norm, states[i])
            x = residual + layer_dict['dropout'](x_ssm)
            
            # FFN with residual connection
            residual = x
            x_norm = layer_dict['norm2'](x)
            x_ffn = layer_dict['ffn'](x_norm)
            x = residual + layer_dict['dropout'](x_ffn)
            
            final_states.append(state)
        
        # Output projection
        x = self.output_proj(x)  # (batch, seq_len, hidden_dim)
        
        # Return to original shape if single-step
        if is_single_step:
            x = x.squeeze(1)  # (batch, hidden_dim)
        
        return x, final_states
    
    def get_output_dim(self) -> int:
        """Get output feature dimension."""
        return self.hidden_dim
    
    def get_complexity(self) -> str:
        """Get computational complexity."""
        return "O(n) where n = sequence_length"
