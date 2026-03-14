#!/usr/bin/env python
"""Evaluation script for Member A (1D-CNN)"""

import sys
from pathlib import Path
import argparse

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import torch
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

from src.data.dataset import SleepEDFDataset
from src.models.memberA_1dcnn.model import MemberA1DCNN
from src.training.trainer import BaseTrainer
from src.utils.config import config_manager
from src.utils.logger import setup_logger
from src.utils.paths import paths

logger = setup_logger(__name__)


def load_test_data(config):
    data_path = paths.data_processed

    eeg_data = torch.load(data_path / 'eeg_epochs.pt') if (data_path / 'eeg_epochs.pt').exists() else torch.from_numpy(np.load(data_path / 'eeg_epochs.npy'))
    eog_data = torch.load(data_path / 'eog_epochs.pt') if (data_path / 'eog_epochs.pt').exists() else torch.from_numpy(np.load(data_path / 'eog_epochs.npy'))
    labels = torch.load(data_path / 'labels.pt') if (data_path / 'labels.pt').exists() else torch.from_numpy(np.load(data_path / 'labels.npy'))
    splits = torch.load(data_path / 'splits.pt') if (data_path / 'splits.pt').exists() else np.load(data_path / 'splits.npy', allow_pickle=True)

    test_mask = splits == 'test'

    test_dataset = SleepEDFDataset(
        eeg_data[test_mask].numpy() if isinstance(eeg_data, torch.Tensor) else eeg_data[test_mask],
        eog_data[test_mask].numpy() if isinstance(eog_data, torch.Tensor) else eog_data[test_mask],
        labels[test_mask].numpy() if isinstance(labels, torch.Tensor) else labels[test_mask],
        modality='eeg'
    )

    test_loader = DataLoader(test_dataset, batch_size=config.training.batch_size, shuffle=False, num_workers=0)
    return test_loader, test_dataset


def main():
    parser = argparse.ArgumentParser(description='Evaluate Member A model')
    parser.add_argument('--config', '-c', type=str, default=None,
                        help='Path to config file (default uses member config)')
    parser.add_argument('--checkpoint', '-k', type=str, default=None,
                        help='Path to model checkpoint (best_model.pth or final_model.pth)')

    args = parser.parse_args()

    if args.config:
        config_path = Path(args.config)
    else:
        config_path = paths.get_member_experiments_path('A') / 'train_config.yaml'

    config = config_manager.load_config(config_path)
    if not config_manager.validate_config(config):
        logger.error('Invalid config, exit')
        return

    model_cfg = config.model
    model = MemberA1DCNN(
        n_classes=model_cfg.get('n_classes', 5),
        input_channels=model_cfg.get('input_channels', 1),
        hidden_dims=tuple(model_cfg.get('hidden_dims', [64, 128, 256])),
        dropout=model_cfg.get('dropout', 0.5)
    )

    trainer = BaseTrainer(model, config)

    checkpoint_path = args.checkpoint or paths.get_member_results_path('A') / 'best_model.pth'
    if not Path(checkpoint_path).exists():
        checkpoint_path = paths.get_member_results_path('A') / 'final_model.pth'

    trainer.load_checkpoint(str(checkpoint_path))
    logger.info(f'Loaded checkpoint: {checkpoint_path}')

    test_loader, test_dataset = load_test_data(config)

    # Evaluate on test set
    results = trainer.test(test_loader, save_predictions=False)

    y_true = np.array(results['labels'])
    y_pred = np.array(results['predictions'])

    report = classification_report(y_true, y_pred, target_names=['W', 'N1', 'N2', 'N3', 'REM'])
    cm = confusion_matrix(y_true, y_pred)

    logger.info('\n' + report)

    result_dir = paths.results / 'memberA_1dcnn'
    result_dir.mkdir(parents=True, exist_ok=True)
    np.save(result_dir / 'test_predictions.npy', y_pred)
    np.save(result_dir / 'test_labels.npy', y_true)

    with open(result_dir / 'classification_report.txt', 'w') as f:
        f.write(report)

    np.save(result_dir / 'confusion_matrix.npy', cm)

    plt.figure(figsize=(6, 6))
    plt.imshow(cm, cmap='Blues')
    plt.title('MemberA Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.colorbar()
    plt.xticks(range(5), ['W', 'N1', 'N2', 'N3', 'REM'])
    plt.yticks(range(5), ['W', 'N1', 'N2', 'N3', 'REM'])

    for i in range(5):
        for j in range(5):
            plt.text(j, i, cm[i, j], ha='center', va='center', color='white' if cm[i, j] > cm.max() / 2 else 'black')

    plt.savefig(result_dir / 'confusion_matrix.png', dpi=150)
    plt.close()

    logger.info(f'Evaluation results saved to {result_dir}')


if __name__ == '__main__':
    main()
