"""
Trainer class for DHC-SSM Architecture.

Provides a complete training loop with:
- Training and validation
- Checkpointing
- Logging and monitoring
- Early stopping
- Learning rate scheduling
"""

import torch
from torch.utils.data import DataLoader
from typing import Optional, Dict, Any
from pathlib import Path
import logging
from tqdm import tqdm

from dhc_ssm.core.model import DHCSSMModel
from dhc_ssm.core.learning_engine import DeterministicOptimizer
from dhc_ssm.utils.metrics import MetricsTracker
from dhc_ssm.utils.logging import TensorBoardLogger

logger = logging.getLogger(__name__)


class Trainer:
    """
    Trainer for DHC-SSM models.
    
    Handles the complete training workflow including:
    - Training loop
    - Validation
    - Checkpointing
    - Logging
    - Early stopping
    """
    
    def __init__(
        self,
        model: DHCSSMModel,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        optimizer: Optional[DeterministicOptimizer] = None,
        checkpoint_dir: str = "./checkpoints",
        log_dir: str = "./logs",
        device: Optional[str] = None,
    ):
        """
        Initialize trainer.
        
        Args:
            model: DHC-SSM model instance
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
            optimizer: Optimizer instance (creates default if None)
            checkpoint_dir: Directory for saving checkpoints
            log_dir: Directory for logs
            device: Device to train on
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        # Create optimizer if not provided
        if optimizer is None:
            optimizer = DeterministicOptimizer(
                model.parameters(),
                lr=model.config.learning_rate,
                weight_decay=model.config.weight_decay,
                gradient_clip=model.config.gradient_clip,
            )
        self.optimizer = optimizer
        
        # Directories
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Metrics tracking
        self.train_metrics = MetricsTracker()
        self.val_metrics = MetricsTracker()
        
        # TensorBoard logging
        self.tb_logger = TensorBoardLogger(self.log_dir)
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        
        logger.info("Trainer initialized")
        logger.info(f"Checkpoint dir: {self.checkpoint_dir}")
        logger.info(f"Log dir: {self.log_dir}")
    
    def train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        epoch_metrics = MetricsTracker()
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}")
        for batch_idx, (inputs, targets) in enumerate(pbar):
            # Training step
            metrics = self.model.train_step(inputs, targets, self.optimizer)
            
            # Update metrics
            epoch_metrics.update(metrics, step=self.global_step)
            self.train_metrics.update(metrics, step=self.global_step)
            
            # Logging
            if self.global_step % self.model.config.log_interval == 0:
                for key, value in metrics.items():
                    self.tb_logger.log_scalar(f"train/{key}", value, self.global_step)
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f"{metrics.get('total', 0):.4f}",
                    'step': self.global_step
                })
            
            self.global_step += 1
        
        return epoch_metrics.get_summary()
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """
        Run validation.
        
        Returns:
            Dictionary of validation metrics
        """
        if self.val_loader is None:
            return {}
        
        self.model.eval()
        val_metrics = MetricsTracker()
        
        for inputs, targets in tqdm(self.val_loader, desc="Validation"):
            metrics = self.model.evaluate_step(inputs, targets)
            val_metrics.update(metrics)
        
        summary = val_metrics.get_summary()
        
        # Log to TensorBoard
        for key, stats in summary.items():
            self.tb_logger.log_scalar(f"val/{key}", stats['mean'], self.current_epoch)
        
        return summary
    
    def save_checkpoint(self, filename: str = "checkpoint.pt", **kwargs) -> None:
        """
        Save training checkpoint.
        
        Args:
            filename: Checkpoint filename
            **kwargs: Additional data to save
        """
        checkpoint_path = self.checkpoint_dir / filename
        
        self.model.save_checkpoint(
            str(checkpoint_path),
            epoch=self.current_epoch,
            global_step=self.global_step,
            optimizer_state=self.optimizer.state_dict(),
            best_val_loss=self.best_val_loss,
            **kwargs
        )
    
    def train(
        self,
        num_epochs: Optional[int] = None,
        early_stopping_patience: int = 10,
    ) -> Dict[str, Any]:
        """
        Run complete training loop.
        
        Args:
            num_epochs: Number of epochs (uses config if None)
            early_stopping_patience: Patience for early stopping
            
        Returns:
            Training history
        """
        if num_epochs is None:
            num_epochs = self.model.config.num_epochs
        
        logger.info(f"Starting training for {num_epochs} epochs")
        
        patience_counter = 0
        history = {
            'train': [],
            'val': [],
        }
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            
            # Train epoch
            train_summary = self.train_epoch()
            history['train'].append(train_summary)
            
            logger.info(f"Epoch {epoch} - Train: {train_summary}")
            
            # Validation
            if self.val_loader is not None:
                val_summary = self.validate()
                history['val'].append(val_summary)
                
                logger.info(f"Epoch {epoch} - Val: {val_summary}")
                
                # Check for improvement
                val_loss = val_summary.get('total', {}).get('mean', float('inf'))
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.save_checkpoint("best_model.pt")
                    logger.info(f"New best model saved (val_loss: {val_loss:.4f})")
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                # Early stopping
                if patience_counter >= early_stopping_patience:
                    logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                    break
            
            # Regular checkpoint
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(f"checkpoint_epoch_{epoch + 1}.pt")
        
        # Final checkpoint
        self.save_checkpoint("final_model.pt")
        
        # Close logger
        self.tb_logger.close()
        
        logger.info("Training completed")
        return history
