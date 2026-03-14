"""Logging utilities"""

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logger(
    name: str,
    log_file: Optional[str] = None,
    level: int = logging.INFO,
    console: bool = True
) -> logging.Logger:
    """
    Setup logger with file and console handlers
    
    Args:
        name: Logger name
        log_file: Path to log file
        level: Logging level
        console: Whether to log to console
    
    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Remove existing handlers
    logger.handlers.clear()
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # File handler
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    # Console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    return logger


class TrainingLogger:
    """Logger for training progress"""
    
    def __init__(self, log_dir: str, experiment_name: str):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.log_file = self.log_dir / f"{experiment_name}.log"
        self.logger = setup_logger(
            experiment_name,
            log_file=str(self.log_file)
        )
        
        # CSV logger for metrics
        self.csv_file = self.log_dir / f"{experiment_name}_metrics.csv"
        self.csv_initialized = False
    
    def log_epoch(self, epoch: int, train_metrics: dict, val_metrics: dict):
        """Log epoch metrics"""
        self.logger.info(
            f"Epoch {epoch} - "
            f"Train Loss: {train_metrics['loss']:.4f}, "
            f"Train Acc: {train_metrics['accuracy']:.4f}, "
            f"Val Loss: {val_metrics['loss']:.4f}, "
            f"Val Acc: {val_metrics['accuracy']:.4f}"
        )
        
        # Write to CSV
        if not self.csv_initialized:
            with open(self.csv_file, 'w') as f:
                header = ['epoch', 'train_loss', 'train_acc', 
                         'val_loss', 'val_acc']
                f.write(','.join(header) + '\n')
            self.csv_initialized = True
        
        with open(self.csv_file, 'a') as f:
            row = [epoch, train_metrics['loss'], train_metrics['accuracy'],
                  val_metrics['loss'], val_metrics['accuracy']]
            f.write(','.join(map(str, row)) + '\n')
    
    def log_config(self, config: dict):
        """Log configuration"""
        self.logger.info("Configuration:")
        for key, value in config.items():
            self.logger.info(f"  {key}: {value}")