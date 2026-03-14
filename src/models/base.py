"""Base model class for all sleep staging models"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import json
from pathlib import Path


class BaseSleepModel(nn.Module, ABC):
    """
    Abstract base class for all sleep staging models
    
    All member models should inherit from this class
    to ensure consistent interface.
    """
    
    def __init__(self, n_classes: int = 5):
        super().__init__()
        self.n_classes = n_classes
    
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor, shape depends on model
        
        Returns:
            Output logits, shape (batch_size, n_classes)
        """
        pass
    
    @abstractmethod
    def get_config(self) -> Dict[str, Any]:
        """
        Get model configuration for reproducibility
        
        Returns:
            Dictionary with model configuration
        """
        pass
    
    def save(self, path: str):
        """
        Save model weights and config
        
        Args:
            path: Path to save directory or file
        """
        path = Path(path)
        
        if path.suffix == '.pth':
            # Save only weights
            torch.save(self.state_dict(), path)
        else:
            # Save weights and config
            path.mkdir(parents=True, exist_ok=True)
            
            # Save weights
            torch.save(self.state_dict(), path / 'model.pth')
            
            # Save config
            with open(path / 'config.json', 'w') as f:
                json.dump(self.get_config(), f, indent=2)
    
    def load(self, path: str):
        """
        Load model weights
        
        Args:
            path: Path to weights file
        """
        device = next(self.parameters()).device
        self.load_state_dict(torch.load(path, map_location=device))
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict class labels
        
        Args:
            x: Input tensor
        
        Returns:
            Predicted class indices
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            predictions = torch.argmax(logits, dim=1)
        return predictions
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict class probabilities
        
        Args:
            x: Input tensor
        
        Returns:
            Class probabilities
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probabilities = torch.softmax(logits, dim=1)
        return probabilities
    
    def get_input_shape(self) -> tuple:
        """
        Get expected input shape
        
        Returns:
            Tuple describing input shape (without batch dimension)
        """
        return self.input_shape if hasattr(self, 'input_shape') else None
    
    def count_parameters(self) -> int:
        """Count trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def summary(self) -> str:
        """Get model summary"""
        lines = []
        lines.append(f"Model: {self.__class__.__name__}")
        lines.append(f"Parameters: {self.count_parameters():,}")
        lines.append(f"Classes: {self.n_classes}")
        
        config = self.get_config()
        if config:
            lines.append("Configuration:")
            for key, value in config.items():
                lines.append(f"  {key}: {value}")
        
        return "\n".join(lines)