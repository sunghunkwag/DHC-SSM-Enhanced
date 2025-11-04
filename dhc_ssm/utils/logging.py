"""Logging utilities for DHC-SSM Architecture."""

import logging
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime


def setup_logging(
    log_file: Optional[Path] = None,
    level: int = logging.INFO,
    format_string: Optional[str] = None,
) -> None:
    """
    Setup logging configuration.
    
    Args:
        log_file: Optional path to log file
        level: Logging level
        format_string: Custom format string
    """
    if format_string is None:
        format_string = (
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
    
    handlers = [logging.StreamHandler(sys.stdout)]
    
    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=level,
        format=format_string,
        handlers=handlers,
        force=True,
    )


def get_logger(name: str, level: Optional[int] = None) -> logging.Logger:
    """
    Get a logger instance.
    
    Args:
        name: Logger name
        level: Optional logging level
        
    Returns:
        Logger instance
    """
    logger = logging.getLogger(name)
    if level is not None:
        logger.setLevel(level)
    return logger


class TensorBoardLogger:
    """Simple TensorBoard logger wrapper."""
    
    def __init__(self, log_dir: Path):
        """
        Initialize TensorBoard logger.
        
        Args:
            log_dir: Directory for TensorBoard logs
        """
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(log_dir)
            self.enabled = True
        except ImportError:
            logging.warning("TensorBoard not available, logging disabled")
            self.writer = None
            self.enabled = False
    
    def log_scalar(self, tag: str, value: float, step: int) -> None:
        """Log scalar value."""
        if self.enabled:
            self.writer.add_scalar(tag, value, step)
    
    def log_scalars(self, main_tag: str, tag_scalar_dict: dict, step: int) -> None:
        """Log multiple scalars."""
        if self.enabled:
            self.writer.add_scalars(main_tag, tag_scalar_dict, step)
    
    def log_histogram(self, tag: str, values, step: int) -> None:
        """Log histogram."""
        if self.enabled:
            self.writer.add_histogram(tag, values, step)
    
    def log_image(self, tag: str, img_tensor, step: int) -> None:
        """Log image."""
        if self.enabled:
            self.writer.add_image(tag, img_tensor, step)
    
    def close(self) -> None:
        """Close the logger."""
        if self.enabled and self.writer is not None:
            self.writer.close()
