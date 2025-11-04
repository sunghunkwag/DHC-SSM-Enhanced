"""Validation utilities for DHC-SSM Architecture."""

import torch
from typing import Tuple, Optional, Union, List
import logging

logger = logging.getLogger(__name__)


def validate_tensor(
    tensor: torch.Tensor,
    name: str = "tensor",
    expected_dtype: Optional[torch.dtype] = None,
    expected_device: Optional[torch.device] = None,
    check_finite: bool = True,
) -> None:
    """
    Validate a tensor's properties.
    
    Args:
        tensor: Tensor to validate
        name: Name of tensor for error messages
        expected_dtype: Expected data type
        expected_device: Expected device
        check_finite: Whether to check for NaN/Inf values
        
    Raises:
        TypeError: If tensor is not a torch.Tensor
        ValueError: If validation fails
    """
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(tensor)}")
    
    if expected_dtype is not None and tensor.dtype != expected_dtype:
        raise ValueError(
            f"{name} has dtype {tensor.dtype}, expected {expected_dtype}"
        )
    
    if expected_device is not None and tensor.device != expected_device:
        raise ValueError(
            f"{name} is on device {tensor.device}, expected {expected_device}"
        )
    
    if check_finite:
        if torch.isnan(tensor).any():
            raise ValueError(f"{name} contains NaN values")
        if torch.isinf(tensor).any():
            raise ValueError(f"{name} contains Inf values")


def validate_shape(
    tensor: torch.Tensor,
    expected_shape: Tuple[Union[int, None], ...],
    name: str = "tensor",
) -> None:
    """
    Validate tensor shape.
    
    Args:
        tensor: Tensor to validate
        expected_shape: Expected shape (None for any dimension)
        name: Name of tensor for error messages
        
    Raises:
        ValueError: If shape doesn't match
    """
    if len(tensor.shape) != len(expected_shape):
        raise ValueError(
            f"{name} has {len(tensor.shape)} dimensions, "
            f"expected {len(expected_shape)}"
        )
    
    for i, (actual, expected) in enumerate(zip(tensor.shape, expected_shape)):
        if expected is not None and actual != expected:
            raise ValueError(
                f"{name} dimension {i} is {actual}, expected {expected}. "
                f"Full shape: {tensor.shape} vs expected {expected_shape}"
            )


def check_nan_inf(
    tensor: torch.Tensor,
    name: str = "tensor",
    raise_error: bool = True,
) -> bool:
    """
    Check for NaN or Inf values in tensor.
    
    Args:
        tensor: Tensor to check
        name: Name of tensor for logging/error messages
        raise_error: Whether to raise error if found
        
    Returns:
        True if tensor contains NaN or Inf, False otherwise
        
    Raises:
        ValueError: If raise_error=True and NaN/Inf found
    """
    has_nan = torch.isnan(tensor).any().item()
    has_inf = torch.isinf(tensor).any().item()
    
    if has_nan or has_inf:
        msg = f"{name} contains "
        if has_nan:
            msg += "NaN"
        if has_nan and has_inf:
            msg += " and "
        if has_inf:
            msg += "Inf"
        msg += " values"
        
        if raise_error:
            raise ValueError(msg)
        else:
            logger.warning(msg)
        return True
    
    return False


def validate_batch(
    batch: Union[torch.Tensor, Tuple, List],
    batch_size: Optional[int] = None,
    name: str = "batch",
) -> None:
    """
    Validate batch data.
    
    Args:
        batch: Batch data (tensor or tuple/list of tensors)
        batch_size: Expected batch size
        name: Name for error messages
        
    Raises:
        ValueError: If validation fails
    """
    if isinstance(batch, torch.Tensor):
        tensors = [batch]
    elif isinstance(batch, (tuple, list)):
        tensors = list(batch)
    else:
        raise TypeError(f"{name} must be tensor or tuple/list of tensors")
    
    if not tensors:
        raise ValueError(f"{name} is empty")
    
    # Check all tensors have same batch size
    batch_sizes = [t.shape[0] for t in tensors if isinstance(t, torch.Tensor)]
    if len(set(batch_sizes)) > 1:
        raise ValueError(
            f"{name} has inconsistent batch sizes: {batch_sizes}"
        )
    
    # Check against expected batch size
    if batch_size is not None and batch_sizes[0] != batch_size:
        raise ValueError(
            f"{name} has batch size {batch_sizes[0]}, expected {batch_size}"
        )


def validate_config(config: object, required_attrs: List[str]) -> None:
    """
    Validate configuration object has required attributes.
    
    Args:
        config: Configuration object
        required_attrs: List of required attribute names
        
    Raises:
        ValueError: If required attributes are missing
    """
    missing = [attr for attr in required_attrs if not hasattr(config, attr)]
    if missing:
        raise ValueError(
            f"Configuration missing required attributes: {missing}"
        )


def validate_dimensions(
    *tensors: torch.Tensor,
    dim: int = -1,
    names: Optional[List[str]] = None,
) -> None:
    """
    Validate that tensors have matching dimensions.
    
    Args:
        *tensors: Tensors to validate
        dim: Dimension to check
        names: Optional names for error messages
        
    Raises:
        ValueError: If dimensions don't match
    """
    if not tensors:
        return
    
    if names is None:
        names = [f"tensor_{i}" for i in range(len(tensors))]
    
    reference_size = tensors[0].shape[dim]
    for tensor, name in zip(tensors[1:], names[1:]):
        if tensor.shape[dim] != reference_size:
            raise ValueError(
                f"{name} dimension {dim} is {tensor.shape[dim]}, "
                f"expected {reference_size} (from {names[0]})"
            )


def safe_divide(
    numerator: torch.Tensor,
    denominator: torch.Tensor,
    epsilon: float = 1e-8,
) -> torch.Tensor:
    """
    Safely divide tensors, avoiding division by zero.
    
    Args:
        numerator: Numerator tensor
        denominator: Denominator tensor
        epsilon: Small value to add to denominator
        
    Returns:
        Result of division
    """
    return numerator / (denominator + epsilon)


def clip_gradients(
    parameters,
    max_norm: float,
    norm_type: float = 2.0,
) -> float:
    """
    Clip gradients by norm.
    
    Args:
        parameters: Model parameters
        max_norm: Maximum gradient norm
        norm_type: Type of norm (2.0 for L2)
        
    Returns:
        Total norm of gradients
    """
    return torch.nn.utils.clip_grad_norm_(
        parameters,
        max_norm=max_norm,
        norm_type=norm_type
    )
