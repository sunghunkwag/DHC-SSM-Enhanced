"""Utility modules for DHC-SSM Architecture."""

from dhc_ssm.utils.config import (
    DHCSSMConfig,
    get_default_config,
    get_small_config,
    get_large_config,
    get_debug_config,
    get_cpu_optimized_config,
    get_gpu_optimized_config,
    get_config,
    PRESET_CONFIGS,
)

from dhc_ssm.utils.device import get_device, move_to_device
from dhc_ssm.utils.validation import validate_tensor, validate_shape, check_nan_inf
from dhc_ssm.utils.logging import get_logger, setup_logging
from dhc_ssm.utils.metrics import compute_metrics, MetricsTracker

__all__ = [
    # Configuration
    "DHCSSMConfig",
    "get_default_config",
    "get_small_config",
    "get_large_config",
    "get_debug_config",
    "get_cpu_optimized_config",
    "get_gpu_optimized_config",
    "get_config",
    "PRESET_CONFIGS",
    
    # Device management
    "get_device",
    "move_to_device",
    
    # Validation
    "validate_tensor",
    "validate_shape",
    "check_nan_inf",
    
    # Logging
    "get_logger",
    "setup_logging",
    
    # Metrics
    "compute_metrics",
    "MetricsTracker",
]
