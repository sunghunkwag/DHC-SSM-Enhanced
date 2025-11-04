"""
Strategic Reasoner for DHC-SSM Architecture.

Layer 3: Causal Graph Neural Network for strategic reasoning and planning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import logging

from dhc_ssm.layers.attention import MultiHeadAttention
from dhc_ssm.layers.normalization import LayerNorm

logger = logging.getLogger(__name__)


class CausalMessagePassing(nn.Module):
    """
    Causal message passing layer for graph neural networks.
    
    Implements message passing with causal constraints.
    """
    
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        """
        Initialize causal message passing.
        
        Args:
            in_dim: Input feature dimension
            out_dim: Output feature dimension
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()
        
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        
        # Message computation
        self.message_net = nn.Sequential(
            nn.Linear(in_dim * 2, out_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        # Attention for message aggregation
        self.attention = MultiHeadAttention(
            embed_dim=out_dim,
            num_heads=num_heads,
            dropout=dropout,
        )
        
        # Update network
        self.update_net = nn.Sequential(
            nn.Linear(in_dim + out_dim, out_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
    
    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: Optional[torch.Tensor] = None,
        causal_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            node_features: Node features (batch, num_nodes, in_dim)
            edge_index: Edge connectivity (2, num_edges) or None for fully connected
            causal_mask: Causal mask (num_nodes, num_nodes) or None
            
        Returns:
            Updated node features (batch, num_nodes, out_dim)
        """
        batch_size, num_nodes, _ = node_features.shape
        device = node_features.device
        
        # If no edge_index, create fully connected graph
        if edge_index is None:
            # Fully connected (excluding self-loops initially)
            edge_index = torch.combinations(torch.arange(num_nodes, device=device), r=2).t()
            # Add reverse edges
            edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
        
        # Compute messages
        messages = []
        for b in range(batch_size):
            node_feat = node_features[b]  # (num_nodes, in_dim)
            
            # Simple approach: use attention-based aggregation
            # Treat as sequence-to-sequence with causal masking
            attended = self.attention(
                query=node_feat.unsqueeze(0),  # (1, num_nodes, in_dim)
                key=node_feat.unsqueeze(0),
                value=node_feat.unsqueeze(0),
                mask=causal_mask.unsqueeze(0) if causal_mask is not None else None,
            )  # (1, num_nodes, out_dim)
            
            messages.append(attended.squeeze(0))
        
        messages = torch.stack(messages, dim=0)  # (batch, num_nodes, out_dim)
        
        # Update nodes
        combined = torch.cat([node_features, messages], dim=-1)  # (batch, num_nodes, in_dim + out_dim)
        updated = self.update_net(combined)  # (batch, num_nodes, out_dim)
        
        return updated


class StrategicReasoner(nn.Module):
    """
    Strategic reasoning module using Causal Graph Neural Networks.
    
    Features:
    - Causal message passing
    - Multi-layer reasoning
    - Attention-based aggregation
    - Robust fallback mechanisms
    """
    
    def __init__(
        self,
        input_dim: int = 256,
        hidden_dim: int = 128,
        output_dim: int = 128,
        num_layers: int = 3,
        num_heads: int = 4,
        num_nodes: int = 8,
        dropout: float = 0.1,
        use_causal_mask: bool = True,
    ):
        """
        Initialize strategic reasoner.
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden feature dimension
            output_dim: Output feature dimension
            num_layers: Number of GNN layers
            num_heads: Number of attention heads
            num_nodes: Number of reasoning nodes
            dropout: Dropout probability
            use_causal_mask: Whether to use causal masking
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.num_nodes = num_nodes
        self.use_causal_mask = use_causal_mask
        
        # Input projection to create multiple reasoning nodes
        self.input_proj = nn.Linear(input_dim, hidden_dim * num_nodes)
        
        # Causal message passing layers
        self.gnn_layers = nn.ModuleList()
        for i in range(num_layers):
            layer_in_dim = hidden_dim
            layer_out_dim = hidden_dim
            
            self.gnn_layers.append(
                nn.ModuleDict({
                    'message_passing': CausalMessagePassing(
                        in_dim=layer_in_dim,
                        out_dim=layer_out_dim,
                        num_heads=num_heads,
                        dropout=dropout,
                    ),
                    'norm': LayerNorm(layer_out_dim),
                    'ffn': nn.Sequential(
                        nn.Linear(layer_out_dim, layer_out_dim * 4),
                        nn.GELU(),
                        nn.Dropout(dropout),
                        nn.Linear(layer_out_dim * 4, layer_out_dim),
                    ),
                    'norm2': LayerNorm(layer_out_dim),
                })
            )
        
        # Aggregation and output
        self.aggregation = nn.Sequential(
            nn.Linear(hidden_dim * num_nodes, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        
        # Causal mask (lower triangular)
        if use_causal_mask:
            mask = torch.tril(torch.ones(num_nodes, num_nodes))
            self.register_buffer('causal_mask', mask)
        else:
            self.causal_mask = None
        
        logger.info(
            f"StrategicReasoner initialized: {num_layers} layers, "
            f"{num_nodes} nodes, dim={hidden_dim}"
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor (batch, input_dim)
            
        Returns:
            Strategic reasoning output (batch, output_dim)
        """
        batch_size = x.shape[0]
        
        # Project input to multiple reasoning nodes
        node_features = self.input_proj(x)  # (batch, hidden_dim * num_nodes)
        node_features = node_features.view(
            batch_size, self.num_nodes, self.hidden_dim
        )  # (batch, num_nodes, hidden_dim)
        
        # Apply GNN layers
        for layer_dict in self.gnn_layers:
            # Message passing with residual
            residual = node_features
            node_features_norm = layer_dict['norm'](node_features)
            
            try:
                node_features_mp = layer_dict['message_passing'](
                    node_features_norm,
                    causal_mask=self.causal_mask if self.use_causal_mask else None,
                )
                node_features = residual + node_features_mp
            except Exception as e:
                logger.warning(f"Message passing failed: {e}, using residual")
                node_features = residual
            
            # FFN with residual
            residual = node_features
            node_features_norm = layer_dict['norm2'](node_features)
            node_features_ffn = layer_dict['ffn'](node_features_norm)
            node_features = residual + node_features_ffn
        
        # Aggregate nodes
        node_features_flat = node_features.view(batch_size, -1)  # (batch, hidden_dim * num_nodes)
        aggregated = self.aggregation(node_features_flat)  # (batch, hidden_dim)
        
        # Output projection
        output = self.output_proj(aggregated)  # (batch, output_dim)
        
        return output
    
    def get_output_dim(self) -> int:
        """Get output feature dimension."""
        return self.output_dim
    
    def get_complexity(self) -> str:
        """Get computational complexity."""
        return f"O(n²) where n = {self.num_nodes} (fixed number of reasoning nodes)"
