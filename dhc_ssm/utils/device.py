"""Device management utilities for DHC-SSM Architecture."""

import torch
from typing import Union, Optional, Any
import logging

logger = logging.getLogger(__name__)


def get_device(device: Optional[Union[str, torch.device]] = None) -> torch.device:
    """
    Get the appropriate device for computation.
    
    Args:
        device: Requested device ('cpu', 'cuda', 'cuda:0', etc.) or None for auto-detection
        
    Returns:
        torch.device instance
    """
    if device is None:
        # Auto-detect best available device
        if torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info(f"Auto-selected CUDA device: {torch.cuda.get_device_name(0)}")
        else:
            device = torch.device("cpu")
            logger.info("CUDA not available, using CPU")
    elif isinstance(device, str):
        device = torch.device(device)
        if device.type == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available, falling back to CPU")
            device = torch.device("cpu")
    
    return device


def move_to_device(
    obj: Any,
    device: Union[str, torch.device],
    non_blocking: bool = False
) -> Any:
    """
    Move object (tensor, model, or container) to specified device.
    
    Args:
        obj: Object to move (tensor, nn.Module, dict, list, tuple)
        device: Target device
        non_blocking: Whether to use non-blocking transfer
        
    Returns:
        Object moved to device
    """
    device = get_device(device)
    
    if isinstance(obj, torch.Tensor):
        return obj.to(device, non_blocking=non_blocking)
    elif isinstance(obj, torch.nn.Module):
        return obj.to(device)
    elif isinstance(obj, dict):
        return {k: move_to_device(v, device, non_blocking) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        moved = [move_to_device(item, device, non_blocking) for item in obj]
        return type(obj)(moved)
    else:
        return obj


def get_device_info(device: Optional[Union[str, torch.device]] = None) -> dict:
    """
    Get information about the device.
    
    Args:
        device: Device to query (None for current default)
        
    Returns:
        Dictionary with device information
    """
    device = get_device(device)
    
    info = {
        "device": str(device),
        "type": device.type,
    }
    
    if device.type == "cuda":
        info.update({
            "name": torch.cuda.get_device_name(device),
            "capability": torch.cuda.get_device_capability(device),
            "total_memory": torch.cuda.get_device_properties(device).total_memory,
            "memory_allocated": torch.cuda.memory_allocated(device),
            "memory_reserved": torch.cuda.memory_reserved(device),
            "cuda_version": torch.version.cuda,
        })
    
    return info


def synchronize(device: Optional[Union[str, torch.device]] = None) -> None:
    """
    Synchronize device operations.
    
    Args:
        device: Device to synchronize (None for current default)
    """
    device = get_device(device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def empty_cache(device: Optional[Union[str, torch.device]] = None) -> None:
    """
    Empty the cache for the specified device.
    
    Args:
        device: Device to clear cache for (None for current default)
    """
    device = get_device(device)
    if device.type == "cuda":
        torch.cuda.empty_cache()
        logger.debug(f"Cleared CUDA cache for {device}")


def set_device(device: Union[str, torch.device]) -> torch.device:
    """
    Set the default device for tensor operations.
    
    Args:
        device: Device to set as default
        
    Returns:
        The device that was set
    """
    device = get_device(device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    return device
