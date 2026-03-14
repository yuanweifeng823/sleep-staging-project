"""Evaluation metrics utilities"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import json


def compute_metrics(
    y_true: List[int],
    y_pred: List[int],
    average: str = 'macro'
) -> Dict[str, float]:
    """
    Compute classification metrics
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        average: Averaging method for F1/precision/recall
    
    Returns:
        Dictionary of metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'f1_macro': f1_score(y_true, y_pred, average='macro'),
        'f1_weighted': f1_score(y_true, y_pred, average='weighted'),
        'precision_macro': precision_score(y_true, y_pred, average='macro'),
        'recall_macro': recall_score(y_true, y_pred, average='macro'),
    }
    
    # Per-class metrics
    labels = [0, 1, 2, 3, 4]
    f1_per_class = f1_score(y_true, y_pred, labels=labels, average=None)
    
    for i, f1 in enumerate(f1_per_class):
        stage_names = ['W', 'N1', 'N2', 'N3', 'REM']
        metrics[f'f1_{stage_names[i]}'] = f1
    
    return metrics


def plot_confusion_matrix(
    y_true: List[int],
    y_pred: List[int],
    class_names: List[str] = ['W', 'N1', 'N2', 'N3', 'REM'],
    normalize: bool = True,
    figsize: Tuple[int, int] = (8, 6),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot confusion matrix
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        class_names: Class names
        normalize: Whether to normalize rows
        figsize: Figure size
        save_path: Path to save figure
    
    Returns:
        Matplotlib figure
    """
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
        title = 'Normalized Confusion Matrix'
    else:
        fmt = 'd'
        title = 'Confusion Matrix'
    
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        xticklabels=class_names,
        yticklabels=class_names,
        cmap='Blues',
        ax=ax
    )
    
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(title)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_training_history(
    history: Dict[str, List],
    metrics: List[str] = ['loss', 'accuracy'],
    figsize: Tuple[int, int] = (12, 4),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot training history
    
    Args:
        history: Training history dictionary
        metrics: Metrics to plot
        figsize: Figure size
        save_path: Path to save figure
    
    Returns:
        Matplotlib figure
    """
    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=figsize)
    
    if n_metrics == 1:
        axes = [axes]
    
    for ax, metric in zip(axes, metrics):
        train_key = f'train_{metric}'
        val_key = f'val_{metric}'
        
        if train_key in history:
            ax.plot(history[train_key], label='Train', marker='o')
        if val_key in history:
            ax.plot(history[val_key], label='Validation', marker='s')
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel(metric.capitalize())
        ax.set_title(f'{metric.capitalize()} over Epochs')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


class MetricsTracker:
    """Track and save metrics during training"""
    
    def __init__(self):
        self.history = {
            'train_loss': [],
            'train_accuracy': [],
            'train_f1': [],
            'val_loss': [],
            'val_accuracy': [],
            'val_f1': []
        }
        self.best_metrics = {}
    
    def update(
        self,
        epoch: int,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float]
    ):
        """Update metrics for an epoch"""
        self.history['train_loss'].append(train_metrics['loss'])
        self.history['train_accuracy'].append(train_metrics['accuracy'])
        self.history['train_f1'].append(train_metrics['f1_macro'])
        
        self.history['val_loss'].append(val_metrics['loss'])
        self.history['val_accuracy'].append(val_metrics['accuracy'])
        self.history['val_f1'].append(val_metrics['f1_macro'])
        
        # Update best metrics
        if val_metrics['f1_macro'] > self.best_metrics.get('best_f1', 0):
            self.best_metrics = {
                'best_epoch': epoch,
                'best_f1': val_metrics['f1_macro'],
                'best_accuracy': val_metrics['accuracy']
            }
    
    def get_summary(self) -> Dict[str, float]:
        """Get training summary"""
        return {
            'best_epoch': self.best_metrics.get('best_epoch', -1),
            'best_f1': self.best_metrics.get('best_f1', 0),
            'best_accuracy': self.best_metrics.get('best_accuracy', 0),
            'final_train_loss': self.history['train_loss'][-1],
            'final_val_loss': self.history['val_loss'][-1]
        }
    
    def save(self, path: str):
        """Save metrics to file"""
        with open(path, 'w') as f:
            json.dump({
                'history': self.history,
                'best_metrics': self.best_metrics
            }, f, indent=2)