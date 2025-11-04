"""Metrics computation and tracking for DHC-SSM Architecture."""

import torch
from typing import Dict, List, Optional, Any
from collections import defaultdict
import numpy as np


def compute_accuracy(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    top_k: int = 1,
) -> float:
    """
    Compute top-k accuracy.
    
    Args:
        predictions: Model predictions (batch_size, num_classes)
        targets: Ground truth labels (batch_size,)
        top_k: Number of top predictions to consider
        
    Returns:
        Accuracy as float
    """
    with torch.no_grad():
        batch_size = targets.size(0)
        _, pred = predictions.topk(top_k, dim=1, largest=True, sorted=True)
        pred = pred.t()
        correct = pred.eq(targets.view(1, -1).expand_as(pred))
        correct_k = correct[:top_k].reshape(-1).float().sum(0, keepdim=True)
        return (correct_k / batch_size).item()


def compute_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    loss: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    """
    Compute comprehensive metrics.
    
    Args:
        predictions: Model predictions
        targets: Ground truth targets
        loss: Optional loss value
        
    Returns:
        Dictionary of metrics
    """
    metrics = {}
    
    # Loss
    if loss is not None:
        metrics["loss"] = loss.item() if torch.is_tensor(loss) else loss
    
    # Accuracy
    if predictions.dim() == 2 and targets.dim() == 1:
        # Classification task
        metrics["accuracy"] = compute_accuracy(predictions, targets, top_k=1)
        if predictions.size(1) >= 5:
            metrics["top5_accuracy"] = compute_accuracy(predictions, targets, top_k=5)
    
    return metrics


class MetricsTracker:
    """Track and aggregate metrics over time."""
    
    def __init__(self):
        """Initialize metrics tracker."""
        self.metrics: Dict[str, List[float]] = defaultdict(list)
        self.steps: List[int] = []
    
    def update(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """
        Update metrics.
        
        Args:
            metrics: Dictionary of metric values
            step: Optional step number
        """
        for key, value in metrics.items():
            if isinstance(value, torch.Tensor):
                value = value.item()
            self.metrics[key].append(float(value))
        
        if step is not None:
            self.steps.append(step)
    
    def get_average(self, key: str, last_n: Optional[int] = None) -> float:
        """
        Get average of a metric.
        
        Args:
            key: Metric name
            last_n: Number of recent values to average (None for all)
            
        Returns:
            Average value
        """
        if key not in self.metrics or not self.metrics[key]:
            return 0.0
        
        values = self.metrics[key]
        if last_n is not None:
            values = values[-last_n:]
        
        return float(np.mean(values))
    
    def get_latest(self, key: str) -> float:
        """
        Get latest value of a metric.
        
        Args:
            key: Metric name
            
        Returns:
            Latest value
        """
        if key not in self.metrics or not self.metrics[key]:
            return 0.0
        return self.metrics[key][-1]
    
    def get_all(self, key: str) -> List[float]:
        """
        Get all values of a metric.
        
        Args:
            key: Metric name
            
        Returns:
            List of all values
        """
        return self.metrics.get(key, [])
    
    def get_summary(self) -> Dict[str, Dict[str, float]]:
        """
        Get summary statistics for all metrics.
        
        Returns:
            Dictionary with mean, std, min, max for each metric
        """
        summary = {}
        for key, values in self.metrics.items():
            if values:
                summary[key] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                    "latest": values[-1],
                }
        return summary
    
    def reset(self) -> None:
        """Reset all metrics."""
        self.metrics.clear()
        self.steps.clear()
    
    def __repr__(self) -> str:
        """String representation."""
        summary = self.get_summary()
        lines = ["MetricsTracker("]
        for key, stats in summary.items():
            lines.append(f"  {key}: {stats['mean']:.4f} ± {stats['std']:.4f}")
        lines.append(")")
        return "\n".join(lines)
