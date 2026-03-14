#!/usr/bin/env python
"""Training script for Member A (1D-CNN)"""

import sys
from pathlib import Path
import argparse

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
from torch.utils.data import DataLoader

from src.data.dataset import SleepEDFDataset
from src.training.trainer import BaseTrainer
from src.utils.config import config_manager
from src.utils.logger import setup_logger
from src.utils.helpers import set_seed
from src.utils.paths import paths

logger = setup_logger(__name__)


def load_data(config):
    """Load training and validation data"""
    # Load processed data
    data_path = paths.data_processed

    # Load data arrays
    eeg_data = torch.load(data_path / 'eeg_epochs.pt')
    eog_data = torch.load(data_path / 'eog_epochs.pt')
    labels = torch.load(data_path / 'labels.pt')
    splits = torch.load(data_path / 'splits.pt')

    # Create datasets
    train_mask = splits == 'train'
    val_mask = splits == 'val'

    train_dataset = SleepEDFDataset(
        eeg_data[train_mask],
        eog_data[train_mask],
        labels[train_mask],
        modality='eeg'
    )

    val_dataset = SleepEDFDataset(
        eeg_data[val_mask],
        eog_data[val_mask],
        labels[val_mask],
        modality='eeg'
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=0
    )

    return train_loader, val_loader


def create_model(config):
    """Create the model (placeholder - implement actual model)"""
    # This is a placeholder - you need to implement your actual model
    # For now, we'll use a simple CNN as an example

    class Simple1DCNN(torch.nn.Module):
        def __init__(self, config):
            super().__init__()
            self.conv1 = torch.nn.Conv1d(1, 64, kernel_size=5, stride=1, padding=2)
            self.conv2 = torch.nn.Conv1d(64, 128, kernel_size=5, stride=1, padding=2)
            self.conv3 = torch.nn.Conv1d(128, 256, kernel_size=5, stride=1, padding=2)
            self.pool = torch.nn.MaxPool1d(2)
            self.dropout = torch.nn.Dropout(config.model.dropout)
            self.fc = torch.nn.Linear(256 * 150, config.model.n_classes)  # Adjust based on input size

        def forward(self, x):
            x = torch.relu(self.conv1(x))
            x = self.pool(x)
            x = torch.relu(self.conv2(x))
            x = self.pool(x)
            x = torch.relu(self.conv3(x))
            x = self.pool(x)
            x = x.view(x.size(0), -1)
            x = self.dropout(x)
            x = self.fc(x)
            return x

    return Simple1DCNN(config)


def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='Train Member A model')
    parser.add_argument('--config', '-c', type=str,
                       help='Path to config file (default: use member config)')
    parser.add_argument('--resume', '-r', type=str,
                       help='Path to checkpoint to resume from')

    args = parser.parse_args()

    # Load configuration
    if args.config:
        config_path = Path(args.config)
    else:
        config_path = paths.get_member_experiments_path('A') / 'train_config.yaml'

    logger.info(f"Loading config from: {config_path}")
    config = config_manager.load_config(config_path)

    # Validate config
    if not config_manager.validate_config(config):
        logger.error("Invalid configuration, aborting...")
        return

    # Set random seed
    set_seed(config.seed)

    logger.info(f"Starting experiment: {config.name}")

    # Create model
    model = create_model(config)
    logger.info(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")

    # Create trainer
    trainer = BaseTrainer(model, config)

    # Load data
    logger.info("Loading training data...")
    train_loader, val_loader = load_data(config)

    # Training
    logger.info("Starting training...")
    history = trainer.fit(train_loader, val_loader, config.training.epochs)

    # Save final model
    save_path = paths.get_member_results_path('A') / 'final_model.pth'
    trainer.save_checkpoint(save_path)
    logger.info(f"Final model saved to: {save_path}")

    logger.info("Training completed!")


if __name__ == "__main__":
    main()