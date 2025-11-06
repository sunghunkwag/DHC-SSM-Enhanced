"""
DHC-SSM: Deterministic Hierarchical Causal State Space Model
Enhanced Architecture v3.0

A production-ready deep learning architecture combining:
- Spatial processing with efficient CNNs (O(n))
- Temporal modeling with State Space Models (O(n))
- Strategic reasoning with Causal Graph Neural Networks
- Deterministic learning with multi-objective optimization

Key Features:
- O(n) linear time complexity (vs O(n²) for transformers)
- Deterministic learning without probabilistic sampling
- Multi-pathway processing (fast + slow reasoning)
- Causal understanding through graph-based reasoning
- Production-ready with comprehensive testing

Author: DHC-SSM Development Team
License: MIT
Version: 3.0.0
"""

from dhc_ssm.version import __version__, __author__, __license__, __description__

# Core components - import from model.py where they are actually defined
from dhc_ssm.core.model import (
    DHCSSMModel,
    DHCSSMConfig,
    SpatialEncoder,
    TemporalSSM,
)

# Utilities
from dhc_ssm.utils.config import (
    get_default_config,
    get_small_config,
    get_large_config,
    get_debug_config
)

# Training
from dhc_ssm.training.trainer import Trainer

__all__ = [
    # Version info
    "__version__",
    "__author__",
    "__license__",
    "__description__",
    
    # Core model
    "DHCSSMModel",
    "DHCSSMConfig",
    
    # Architecture components
    "SpatialEncoder",
    "TemporalSSM",
    
    # Configuration
    "get_default_config",
    "get_small_config",
    "get_large_config",
    "get_debug_config",
    
    # Training
    "Trainer",
]
