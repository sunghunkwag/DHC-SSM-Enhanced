"""
Configuration management for DHC-SSM Architecture.

Provides configuration classes and preset configurations for different use cases.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional, List
import json
from pathlib import Path


@dataclass
class DHCSSMConfig:
    """
    Configuration for DHC-SSM Architecture.
    
    This dataclass contains all hyperparameters and settings for the model.
    """
    
    # Model architecture
    input_channels: int = 3
    spatial_dim: int = 128
    temporal_dim: int = 256
    strategic_dim: int = 128
    hidden_dim: int = 256
    output_dim: int = 10
    
    # Spatial encoder (CNN)
    cnn_channels: List[int] = field(default_factory=lambda: [64, 128, 256])
    cnn_kernel_sizes: List[int] = field(default_factory=lambda: [3, 3, 3])
    cnn_strides: List[int] = field(default_factory=lambda: [2, 2, 2])
    use_residual: bool = True
    use_attention: bool = True
    
    # Temporal processor (SSM)
    ssm_state_dim: int = 128
    ssm_num_layers: int = 4
    use_parallel_scan: bool = True
    
    # Strategic reasoner (GNN)
    gnn_num_layers: int = 3
    gnn_heads: int = 4
    gnn_dropout: float = 0.1
    use_causal_mask: bool = True
    
    # Learning engine
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    gradient_clip: float = 1.0
    num_objectives: int = 3
    
    # Training settings
    batch_size: int = 32
    num_epochs: int = 100
    warmup_steps: int = 1000
    use_mixed_precision: bool = True
    gradient_accumulation_steps: int = 1
    
    # Optimization
    optimizer: str = "adamw"
    scheduler: str = "cosine"
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-8
    
    # Regularization
    dropout: float = 0.1
    layer_dropout: float = 0.0
    attention_dropout: float = 0.1
    
    # Device and performance
    device: str = "cpu"
    num_workers: int = 4
    pin_memory: bool = True
    use_checkpoint: bool = False
    
    # Logging and checkpointing
    log_interval: int = 100
    eval_interval: int = 1000
    save_interval: int = 5000
    log_dir: str = "./logs"
    checkpoint_dir: str = "./checkpoints"
    
    # Validation
    validate_shapes: bool = True
    check_nan: bool = True
    verbose: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)
    
    def to_json(self, path: Optional[Path] = None) -> str:
        """
        Convert config to JSON string or save to file.
        
        Args:
            path: Optional path to save JSON file
            
        Returns:
            JSON string representation
        """
        json_str = json.dumps(self.to_dict(), indent=2)
        if path is not None:
            Path(path).write_text(json_str)
        return json_str
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "DHCSSMConfig":
        """Create config from dictionary."""
        return cls(**config_dict)
    
    @classmethod
    def from_json(cls, path: Path) -> "DHCSSMConfig":
        """Load config from JSON file."""
        config_dict = json.loads(Path(path).read_text())
        return cls.from_dict(config_dict)
    
    def update(self, **kwargs) -> "DHCSSMConfig":
        """Update config with new values."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Unknown config parameter: {key}")
        return self
    
    def __repr__(self) -> str:
        """String representation of config."""
        lines = ["DHCSSMConfig("]
        for key, value in self.to_dict().items():
            lines.append(f"  {key}={value},")
        lines.append(")")
        return "\n".join(lines)


def get_default_config() -> DHCSSMConfig:
    """
    Get default configuration for general use.
    
    Returns:
        Default DHCSSMConfig instance
    """
    return DHCSSMConfig()


def get_small_config() -> DHCSSMConfig:
    """
    Get small configuration for fast experimentation.
    
    Returns:
        Small DHCSSMConfig instance with reduced dimensions
    """
    return DHCSSMConfig(
        spatial_dim=64,
        temporal_dim=128,
        strategic_dim=64,
        hidden_dim=128,
        cnn_channels=[32, 64, 128],
        ssm_num_layers=2,
        gnn_num_layers=2,
        gnn_heads=2,
        batch_size=16,
    )


def get_large_config() -> DHCSSMConfig:
    """
    Get large configuration for maximum capacity.
    
    Returns:
        Large DHCSSMConfig instance with increased dimensions
    """
    return DHCSSMConfig(
        spatial_dim=256,
        temporal_dim=512,
        strategic_dim=256,
        hidden_dim=512,
        cnn_channels=[128, 256, 512],
        ssm_num_layers=6,
        gnn_num_layers=4,
        gnn_heads=8,
        batch_size=64,
        use_checkpoint=True,
    )


def get_debug_config() -> DHCSSMConfig:
    """
    Get minimal configuration for debugging.
    
    Returns:
        Debug DHCSSMConfig instance with minimal dimensions
    """
    return DHCSSMConfig(
        spatial_dim=32,
        temporal_dim=64,
        strategic_dim=32,
        hidden_dim=64,
        cnn_channels=[16, 32, 64],
        ssm_num_layers=1,
        gnn_num_layers=1,
        gnn_heads=1,
        batch_size=2,
        num_epochs=10,
        log_interval=10,
        validate_shapes=True,
        verbose=True,
    )


def get_cpu_optimized_config() -> DHCSSMConfig:
    """
    Get CPU-optimized configuration.
    
    Returns:
        CPU-optimized DHCSSMConfig instance
    """
    return DHCSSMConfig(
        spatial_dim=96,
        temporal_dim=192,
        strategic_dim=96,
        hidden_dim=192,
        cnn_channels=[48, 96, 192],
        ssm_num_layers=3,
        gnn_num_layers=2,
        gnn_heads=4,
        batch_size=16,
        device="cpu",
        use_mixed_precision=False,
        num_workers=4,
        use_checkpoint=False,
    )


def get_gpu_optimized_config() -> DHCSSMConfig:
    """
    Get GPU-optimized configuration.
    
    Returns:
        GPU-optimized DHCSSMConfig instance
    """
    return DHCSSMConfig(
        spatial_dim=192,
        temporal_dim=384,
        strategic_dim=192,
        hidden_dim=384,
        cnn_channels=[96, 192, 384],
        ssm_num_layers=5,
        gnn_num_layers=3,
        gnn_heads=8,
        batch_size=64,
        device="cuda",
        use_mixed_precision=True,
        num_workers=8,
        pin_memory=True,
        use_checkpoint=True,
    )


# Preset configurations registry
PRESET_CONFIGS = {
    "default": get_default_config,
    "small": get_small_config,
    "large": get_large_config,
    "debug": get_debug_config,
    "cpu": get_cpu_optimized_config,
    "gpu": get_gpu_optimized_config,
}


def get_config(preset: str = "default", **kwargs) -> DHCSSMConfig:
    """
    Get configuration by preset name with optional overrides.
    
    Args:
        preset: Name of preset configuration
        **kwargs: Additional parameters to override
        
    Returns:
        DHCSSMConfig instance
        
    Raises:
        ValueError: If preset name is unknown
    """
    if preset not in PRESET_CONFIGS:
        raise ValueError(
            f"Unknown preset: {preset}. "
            f"Available presets: {list(PRESET_CONFIGS.keys())}"
        )
    
    config = PRESET_CONFIGS[preset]()
    if kwargs:
        config.update(**kwargs)
    
    return config
