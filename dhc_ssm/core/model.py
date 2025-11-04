"""
Integrated DHC-SSM Model.

Combines all four layers into a complete architecture:
1. Spatial Encoder (CNN) - O(n)
2. Temporal Processor (SSM) - O(n)
3. Strategic Reasoner (GNN) - O(1) fixed nodes
4. Learning Engine - Deterministic optimization

Overall complexity: O(n) where n = input_size * sequence_length
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple, Any, Union
import logging

from dhc_ssm.core.spatial_encoder import SpatialEncoder
from dhc_ssm.core.temporal_processor import TemporalProcessor
from dhc_ssm.core.strategic_reasoner import StrategicReasoner
from dhc_ssm.core.learning_engine import LearningEngine, DeterministicOptimizer
from dhc_ssm.utils.config import DHCSSMConfig
from dhc_ssm.utils.device import get_device, move_to_device
from dhc_ssm.utils.validation import validate_tensor, check_nan_inf

logger = logging.getLogger(__name__)


class DHCSSMModel(nn.Module):
    """
    Complete DHC-SSM Architecture.
    
    Deterministic Hierarchical Causal State Space Model with:
    - O(n) complexity
    - Deterministic learning
    - Multi-pathway processing
    - Causal reasoning
    """
    
    def __init__(self, config: Optional[DHCSSMConfig] = None):
        """
        Initialize DHC-SSM model.
        
        Args:
            config: Configuration object (uses default if None)
        """
        super().__init__()
        
        if config is None:
            from dhc_ssm.utils.config import get_default_config
            config = get_default_config()
        
        self.config = config
        
        # Layer 1: Spatial Encoder (CNN)
        self.spatial_encoder = SpatialEncoder(
            input_channels=config.input_channels,
            channels=config.cnn_channels,
            output_dim=config.spatial_dim,
            kernel_sizes=config.cnn_kernel_sizes,
            strides=config.cnn_strides,
            use_residual=config.use_residual,
            use_attention=config.use_attention,
            dropout=config.dropout,
        )
        
        # Layer 2: Temporal Processor (SSM)
        self.temporal_processor = TemporalProcessor(
            input_dim=config.spatial_dim,
            hidden_dim=config.temporal_dim,
            state_dim=config.ssm_state_dim,
            num_layers=config.ssm_num_layers,
            dropout=config.dropout,
        )
        
        # Layer 3: Strategic Reasoner (GNN)
        self.strategic_reasoner = StrategicReasoner(
            input_dim=config.temporal_dim,
            hidden_dim=config.strategic_dim,
            output_dim=config.strategic_dim,
            num_layers=config.gnn_num_layers,
            num_heads=config.gnn_heads,
            dropout=config.gnn_dropout,
            use_causal_mask=config.use_causal_mask,
        )
        
        # Layer 4: Learning Engine
        self.learning_engine = LearningEngine(
            input_dim=config.strategic_dim,
            num_objectives=config.num_objectives,
            hidden_dim=config.hidden_dim,
            output_dim=config.output_dim,
            dropout=config.dropout,
        )
        
        # Fusion layer (combine fast and slow pathways)
        self.fusion = nn.Sequential(
            nn.Linear(config.strategic_dim + config.temporal_dim, config.strategic_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )
        
        # Device
        self.device = get_device(config.device)
        self.to(self.device)
        
        # Count parameters
        self.num_parameters = sum(p.numel() for p in self.parameters())
        self.num_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        logger.info(
            f"DHCSSMModel initialized with {self.num_parameters:,} parameters "
            f"({self.num_trainable:,} trainable)"
        )
        logger.info(f"Model device: {self.device}")
        logger.info(f"Overall complexity: O(n) linear")
    
    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """
        Forward pass through the complete architecture.
        
        Args:
            x: Input tensor (batch, channels, height, width)
            return_features: Whether to return intermediate features
            
        Returns:
            Predictions (batch, output_dim) or (predictions, features_dict)
        """
        features = {}
        
        # Layer 1: Spatial encoding
        spatial_features = self.spatial_encoder(x)  # (batch, spatial_dim)
        if return_features:
            features['spatial'] = spatial_features
        
        # Layer 2: Temporal processing
        # Add sequence dimension for temporal processing
        temporal_input = spatial_features.unsqueeze(1)  # (batch, 1, spatial_dim)
        temporal_features, _ = self.temporal_processor(temporal_input)  # (batch, 1, temporal_dim)
        temporal_features = temporal_features.squeeze(1)  # (batch, temporal_dim)
        if return_features:
            features['temporal'] = temporal_features
        
        # Layer 3: Strategic reasoning (slow pathway)
        strategic_features = self.strategic_reasoner(temporal_features)  # (batch, strategic_dim)
        if return_features:
            features['strategic'] = strategic_features
        
        # Multi-pathway fusion (fast + slow)
        # Fast pathway: direct temporal features
        # Slow pathway: strategic reasoning
        combined = torch.cat([temporal_features, strategic_features], dim=-1)
        fused_features = self.fusion(combined)  # (batch, strategic_dim)
        if return_features:
            features['fused'] = fused_features
        
        # Layer 4: Learning and prediction
        predictions = self.learning_engine(fused_features)  # (batch, output_dim)
        
        if return_features:
            return predictions, features
        return predictions
    
    def compute_loss(
        self,
        x: torch.Tensor,
        targets: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute loss for training.
        
        Args:
            x: Input tensor (batch, channels, height, width)
            targets: Target labels (batch,)
            
        Returns:
            Tuple of (loss, loss_dict)
        """
        # Forward pass
        predictions, features = self.forward(x, return_features=True)
        
        # Compute multi-objective loss
        loss, loss_dict = self.learning_engine.compute_loss(
            predictions=predictions,
            targets=targets,
            features=features.get('fused'),
        )
        
        # Check for numerical issues
        if check_nan_inf(loss, name="total_loss", raise_error=False):
            logger.error("NaN/Inf in loss, returning dummy loss")
            loss = torch.tensor(1.0, device=loss.device, requires_grad=True)
            loss_dict['error'] = 1.0
        
        return loss, loss_dict
    
    def train_step(
        self,
        x: torch.Tensor,
        targets: torch.Tensor,
        optimizer: Optional[DeterministicOptimizer] = None,
    ) -> Dict[str, float]:
        """
        Perform a single training step.
        
        Args:
            x: Input tensor
            targets: Target labels
            optimizer: Optimizer instance (if None, only computes loss)
            
        Returns:
            Dictionary of metrics
        """
        self.train()
        
        # Move to device
        x = move_to_device(x, self.device)
        targets = move_to_device(targets, self.device)
        
        # Compute loss
        loss, loss_dict = self.compute_loss(x, targets)
        
        # Optimization step
        if optimizer is not None:
            grad_norm = optimizer.step(loss)
            loss_dict['grad_norm'] = grad_norm
        
        return loss_dict
    
    @torch.no_grad()
    def evaluate_step(
        self,
        x: torch.Tensor,
        targets: torch.Tensor,
    ) -> Dict[str, float]:
        """
        Perform a single evaluation step.
        
        Args:
            x: Input tensor
            targets: Target labels
            
        Returns:
            Dictionary of metrics
        """
        self.eval()
        
        # Move to device
        x = move_to_device(x, self.device)
        targets = move_to_device(targets, self.device)
        
        # Compute loss
        loss, loss_dict = self.compute_loss(x, targets)
        
        # Compute accuracy
        predictions = self.forward(x)
        pred_labels = predictions.argmax(dim=-1)
        accuracy = (pred_labels == targets).float().mean().item()
        loss_dict['accuracy'] = accuracy
        
        return loss_dict
    
    def get_diagnostics(self) -> Dict[str, Any]:
        """
        Get model diagnostics and statistics.
        
        Returns:
            Dictionary of diagnostic information
        """
        diagnostics = {
            'architecture': 'DHC-SSM v3.0',
            'num_parameters': self.num_parameters,
            'num_trainable': self.num_trainable,
            'device': str(self.device),
            'complexity': 'O(n) linear',
            'layers': {
                'spatial_encoder': {
                    'type': 'Enhanced CNN',
                    'output_dim': self.spatial_encoder.get_output_dim(),
                    'complexity': self.spatial_encoder.get_complexity(),
                },
                'temporal_processor': {
                    'type': 'State Space Model',
                    'output_dim': self.temporal_processor.get_output_dim(),
                    'complexity': self.temporal_processor.get_complexity(),
                },
                'strategic_reasoner': {
                    'type': 'Causal GNN',
                    'output_dim': self.strategic_reasoner.get_output_dim(),
                    'complexity': self.strategic_reasoner.get_complexity(),
                },
                'learning_engine': {
                    'type': 'Multi-Objective Optimizer',
                    'output_dim': self.learning_engine.get_output_dim(),
                },
            },
            'config': self.config.to_dict(),
        }
        
        return diagnostics
    
    def save_checkpoint(self, path: str, **kwargs) -> None:
        """
        Save model checkpoint.
        
        Args:
            path: Path to save checkpoint
            **kwargs: Additional data to save
        """
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'config': self.config.to_dict(),
            'num_parameters': self.num_parameters,
            **kwargs
        }
        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved to {path}")
    
    @classmethod
    def load_checkpoint(cls, path: str, device: Optional[str] = None) -> 'DHCSSMModel':
        """
        Load model from checkpoint.
        
        Args:
            path: Path to checkpoint
            device: Device to load model on
            
        Returns:
            Loaded model instance
        """
        checkpoint = torch.load(path, map_location=device or 'cpu')
        
        # Create config
        from dhc_ssm.utils.config import DHCSSMConfig
        config = DHCSSMConfig.from_dict(checkpoint['config'])
        if device is not None:
            config.device = device
        
        # Create model
        model = cls(config)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        logger.info(f"Checkpoint loaded from {path}")
        return model


# Type alias for convenience
