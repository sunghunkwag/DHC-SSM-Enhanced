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

# Core components
from dhc_ssm.core.model import DHCSSMModel
from dhc_ssm.core.spatial_encoder import SpatialEncoder
from dhc_ssm.core.temporal_processor import TemporalProcessor
from dhc_ssm.core.strategic_reasoner import StrategicReasoner
from dhc_ssm.core.learning_engine import LearningEngine

# Utilities
from dhc_ssm.utils.config import (
    DHCSSMConfig,
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
    
    # Architecture components
    "SpatialEncoder",
    "TemporalProcessor",
    "StrategicReasoner",
    "LearningEngine",
    
    # Configuration
    "DHCSSMConfig",
    "get_default_config",
    "get_small_config",
    "get_large_config",
    "get_debug_config",
    
    # Training
    "Trainer",
]

# Package metadata
__all__.extend([
    "__version__",
    "__author__",
    "__license__",
    "__description__"
])
