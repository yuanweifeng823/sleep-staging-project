"""Base trainer for all models"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import time
import logging
from typing import Dict, Any, Optional, Callable, Union
import json

from .metrics import MetricsTracker, compute_metrics
from ..utils.config import ExperimentConfig
from ..utils.logger import setup_logger

logger = setup_logger(__name__)


class BaseTrainer:
    """
    Base trainer class for all models
    
    Provides common training functionality.
    All member trainers should inherit from this.
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: Union[Dict[str, Any], ExperimentConfig],
        device: Optional[str] = None
    ):
        """
        Args:
            model: PyTorch model
            config: Training configuration (dict or ExperimentConfig)
            device: Device to use ('cuda' or 'cpu')
        """
        self.model = model

        # Handle config
        if isinstance(config, ExperimentConfig):
            self.config = config.training
            self.experiment_config = config
        else:
            self.config = config
            self.experiment_config = None

        # Set device
        if device is None:
            if self.experiment_config and self.experiment_config.device != 'auto':
                self.device = torch.device(self.experiment_config.device)
            else:
                self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        self.model = self.model.to(self.device)

        # Training components
        self.criterion = self._create_criterion()
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        
        # Metrics tracking
        self.metrics_tracker = MetricsTracker()

        # Save/checkpoint defaults
        if hasattr(self.config, 'save_dir') and self.config.save_dir:
            self.save_dir = Path(self.config.save_dir)
        elif self.experiment_config is not None and hasattr(self.experiment_config, 'output_dir'):
            self.save_dir = Path(self.experiment_config.output_dir)
        else:
            self.save_dir = Path('./checkpoints')
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Early stopping
        self.early_stopping = getattr(self.config, 'early_stopping', True)
        self.patience = getattr(self.config, 'patience', 10)
        self.no_improve_epochs = 0

        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')

        logger.info(f"Trainer initialized on device: {self.device}")
        logger.info(f"Model parameters: {self.model.count_parameters():,}")

    
    def _create_criterion(self) -> nn.Module:
        """Create loss function"""
        return nn.CrossEntropyLoss()
    
    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer"""
        # config may be ExperimentConfig or raw dict
        if hasattr(self.config, 'learning_rate'):
            lr = self.config.learning_rate
        else:
            lr = self.config.get('learning_rate', 0.001)

        if hasattr(self.config, 'weight_decay'):
            weight_decay = self.config.weight_decay
        else:
            weight_decay = self.config.get('weight_decay', 0.0001)
        
        return torch.optim.Adam(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
    
    def _create_scheduler(self) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        """Create learning rate scheduler"""
        if hasattr(self.config, 'scheduler'):
            scheduler_type = self.config.scheduler
        else:
            scheduler_type = self.config.get('scheduler', 'none')
        
        if scheduler_type == 'cosine':
            if hasattr(self.config, 'epochs'):
                t_max = self.config.epochs
            else:
                t_max = self.config.get('epochs', 50)

            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=t_max
            )
        elif scheduler_type == 'step':
            if hasattr(self.config, 'step_size'):
                step_size = self.config.step_size
            else:
                step_size = self.config.get('step_size', 10)

            if hasattr(self.config, 'gamma'):
                gamma = self.config.gamma
            else:
                gamma = self.config.get('gamma', 0.1)

            return torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=step_size,
                gamma=gamma
            )
        else:
            return None
    
    def train_epoch(
        self,
        train_loader: DataLoader
    ) -> Dict[str, float]:
        """
        Train for one epoch
        
        Returns:
            Dictionary of metrics
        """
        self.model.train()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(self.device), y.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            output = self.model(x)
            loss = self.criterion(output, y)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if hasattr(self.config, 'grad_clip'):
                clip_grad = self.config.grad_clip
            else:
                clip_grad = self.config.get('grad_clip', None)
            if clip_grad:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip_grad)
            
            self.optimizer.step()
            
            # Track metrics
            total_loss += loss.item()
            all_preds.extend(output.argmax(1).cpu().numpy())
            all_labels.extend(y.cpu().numpy())
        
        # Compute epoch metrics
        avg_loss = total_loss / len(train_loader)
        metrics = compute_metrics(all_labels, all_preds)
        metrics['loss'] = avg_loss
        
        return metrics
    
    def validate(
        self,
        val_loader: DataLoader
    ) -> Dict[str, float]:
        """
        Validate model
        
        Returns:
            Dictionary of metrics
        """
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(self.device), y.to(self.device)
                
                output = self.model(x)
                loss = self.criterion(output, y)
                
                total_loss += loss.item()
                all_preds.extend(output.argmax(1).cpu().numpy())
                all_labels.extend(y.cpu().numpy())
        
        # Compute metrics
        avg_loss = total_loss / len(val_loader)
        metrics = compute_metrics(all_labels, all_preds)
        metrics['loss'] = avg_loss
        
        return metrics
    
    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Main training loop

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of epochs (overrides config)
        
        Returns:
            Training history
        """
        if epochs is None:
            if hasattr(self.config, 'epochs'):
                epochs = self.config.epochs
            else:
                epochs = self.config.get('epochs', 50)

        logger.info(f"Starting training for {epochs} epochs")

        for epoch in range(epochs):
            self.current_epoch = epoch

            # Training
            start_time = time.time()
            train_metrics = self.train_epoch(train_loader)
            
            # Validation
            val_metrics = self.validate(val_loader)
            
            # Update scheduler
            if self.scheduler:
                self.scheduler.step()
            
            # Track metrics
            self.metrics_tracker.update(
                epoch=epoch,
                train_metrics=train_metrics,
                val_metrics=val_metrics
            )
            
            # Log progress
            epoch_time = time.time() - start_time
            logger.info(
                f"Epoch {epoch+1}/{epochs} - "
                f"Train Loss: {train_metrics['loss']:.4f}, "
                f"Train Acc: {train_metrics['accuracy']:.4f}, "
                f"Val Loss: {val_metrics['loss']:.4f}, "
                f"Val Acc: {val_metrics['accuracy']:.4f}, "
                f"Time: {epoch_time:.2f}s"
            )
            
            # Save best model
            if val_metrics['loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['loss']
                self.no_improve_epochs = 0
                self.save_checkpoint('best_model.pth', val_metrics)
            else:
                self.no_improve_epochs += 1

            if self.early_stopping and self.no_improve_epochs >= self.patience:
                logger.info(f"Early stopping at epoch {epoch+1} (no improvement for {self.patience} epochs)")
                break
        
        # Save final model
        self.save_checkpoint('final_model.pth', val_metrics)
        
        return self.metrics_tracker.history
    
    def save_checkpoint(self, filename: str, metrics: Dict[str, float]):
        """Save model checkpoint"""
        save_dir = self.save_dir
        save_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
            'config': self.config
        }
        
        path = save_dir / filename
        torch.save(checkpoint, path)
        logger.debug(f"Checkpoint saved to {path}")
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        
        logger.info(f"Checkpoint loaded from {path}")
    
    def test(
        self,
        test_loader: DataLoader,
        save_predictions: bool = True
    ) -> Dict[str, Any]:
        """
        Test model
        
        Args:
            test_loader: Test data loader
            save_predictions: Whether to save predictions
        
        Returns:
            Test metrics and predictions
        """
        self.model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for x, y in test_loader:
                x = x.to(self.device)
                output = self.model(x)
                
                preds = output.argmax(1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(y.numpy())
        
        # Compute metrics
        metrics = compute_metrics(all_labels, all_preds)
        
        # Save predictions
        if save_predictions:
            save_dir = Path(self.config.get('save_dir', './results'))
            save_dir.mkdir(parents=True, exist_ok=True)
            
            np.save(save_dir / 'predictions.npy', np.array(all_preds))
            np.save(save_dir / 'labels.npy', np.array(all_labels))
            
            # Save metrics as JSON
            with open(save_dir / 'test_metrics.json', 'w') as f:
                json.dump(metrics, f, indent=2)
        
        return {
            'metrics': metrics,
            'predictions': all_preds,
            'labels': all_labels
        }