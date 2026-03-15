#!/usr/bin/env python
"""Training script for Member A (1D-CNN)"""

import sys
from pathlib import Path
import argparse

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import torch
import numpy as np
from torch.utils.data import DataLoader

from src.data.dataset import SleepEDFDataset
from src.models.memberA_1dcnn.model import MemberA1DCNN
from src.training.trainer import BaseTrainer
from src.utils.config import config_manager
from src.utils.logger import setup_logger
from src.utils.helpers import set_seed
from src.utils.paths import paths

logger = setup_logger(__name__)


def _build_augmentation_transform(config):
    """Creates a lightweight EEG augmentation transform."""
    noise_std = getattr(config.data, 'noise_std', 0.01)
    noise_prob = getattr(config.data, 'noise_prob', 0.5)
    scale_std = getattr(config.data, 'scale_std', 0.05)
    scale_prob = getattr(config.data, 'scale_prob', 0.5)

    def transform(x):
        if torch.rand(1).item() < noise_prob:
            x = x + torch.randn_like(x) * noise_std

        if torch.rand(1).item() < scale_prob:
            scale = 1.0 + torch.randn(1).item() * scale_std
            x = x * scale

        return x

    return transform


def load_data(config):
    """Load training and validation data"""
    data_path = Path(paths.data_processed)

    # Handle both .pt and .npy outputs gracefully
    def _load_array(name):
        pt_path = data_path / f"{name}.pt"
        np_path = data_path / f"{name}.npy"
        if pt_path.exists():
            val = torch.load(pt_path)
            if name == 'splits':
                return val.numpy() if isinstance(val, torch.Tensor) else np.array(val)
            return val
        if np_path.exists():
            np_val = np.load(np_path, allow_pickle=True)
            if name == 'splits':
                return np_val
            # for numeric arrays convert to tensor when needed
            if np.issubdtype(np_val.dtype, np.number):
                return torch.from_numpy(np_val)
            return np_val
        raise FileNotFoundError(f"No data file found for {name}: {pt_path} or {np_path}")

    eeg_data = _load_array('eeg_epochs')
    eog_data = _load_array('eog_epochs')
    labels = _load_array('labels')
    splits = _load_array('splits')

    # Ensure splits are numpy array
    if isinstance(splits, torch.Tensor):
        splits = splits.numpy()
    if isinstance(splits, np.ndarray) and splits.dtype != bool:
        try:
            splits = splits.astype('U')
        except Exception:
            splits = np.array(splits, dtype=object)

    # Create datasets
    train_mask = splits == 'train'
    val_mask = splits == 'val'

    if isinstance(eeg_data, torch.Tensor):
        eeg_data = eeg_data.numpy()
    if isinstance(eog_data, torch.Tensor):
        eog_data = eog_data.numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.numpy()

    # Ensure mask is boolean
    train_mask = np.array(train_mask, dtype=bool)
    val_mask = np.array(val_mask, dtype=bool)

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

    # Data augmentation only on training split
    train_dataset.transform = _build_augmentation_transform(config)

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

    class_weights = train_dataset.get_class_weights()
    return train_loader, val_loader, class_weights


def create_model(config):
    """Create MemberA1DCNN model using config values"""
    model_cfg = config.model

    return MemberA1DCNN(
        n_classes=getattr(model_cfg, 'n_classes', 5),
        input_channels=getattr(model_cfg, 'input_channels', 1),
        hidden_dims=tuple(getattr(model_cfg, 'hidden_dims', [64, 128, 256])),
        dropout=getattr(model_cfg, 'dropout', 0.5)
    )


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

    # Ensure save directory is in config
    if not getattr(config.training, 'save_dir', None):
        config.training.save_dir = str(paths.get_member_results_path('A'))

    # Create trainer
    trainer = BaseTrainer(model, config)

    # Load data
    logger.info("Loading training data...")
    train_loader, val_loader, class_weights = load_data(config)

    # Use class weights to mitigate imbalance
    device = trainer.device
    trainer.criterion = torch.nn.CrossEntropyLoss(weight=class_weights.to(device))

    # Training
    logger.info("Starting training...")
    history = trainer.fit(train_loader, val_loader, config.training.epochs)

    # Final checkpoint already saved by trainer.fit
    logger.info(f"Final model saved to: {config.training.save_dir}/final_model.pth")

    logger.info("Training completed!")


if __name__ == "__main__":
    main()