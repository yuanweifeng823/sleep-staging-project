"""PyTorch Dataset for Sleep-EDFx"""

import torch
from torch.utils.data import Dataset
import numpy as np
from typing import Optional, Callable, Tuple, Union


class SleepEDFDataset(Dataset):
    """
    PyTorch Dataset for sleep staging data
    
    This unified dataset class should be used by all team members
    to ensure consistent data loading.
    """
    
    def __init__(
        self,
        eeg_epochs: np.ndarray,
        eog_epochs: np.ndarray,
        labels: np.ndarray,
        modality: str = 'eeg',
        transform: Optional[Callable] = None,
        subject_ids: Optional[np.ndarray] = None
    ):
        """
        Args:
            eeg_epochs: EEG epochs, shape (n_epochs, n_samples)
            eog_epochs: EOG epochs, shape (n_epochs, n_samples)
            labels: Sleep stage labels (0-4)
            modality: Input modality:
                - 'eeg': only EEG
                - 'eog': only EOG
                - 'both': both EEG and EOG (stacked)
            transform: Optional transform to apply
            subject_ids: Optional subject identifiers
        """
        self.eeg = torch.FloatTensor(eeg_epochs)
        self.eog = torch.FloatTensor(eog_epochs)
        self.labels = torch.LongTensor(labels)
        self.modality = modality
        self.transform = transform
        
        if subject_ids is not None:
            self.subject_ids = subject_ids
        else:
            self.subject_ids = np.array(['unknown'] * len(labels))
    
    def __len__(self) -> int:
        return len(self.labels)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Get data sample by index"""
        if self.modality == 'eeg':
            x = self.eeg[idx].unsqueeze(0)  # (1, n_samples)
        elif self.modality == 'eog':
            x = self.eog[idx].unsqueeze(0)  # (1, n_samples)
        elif self.modality == 'both':
            # Stack EEG and EOG along channel dimension
            x = torch.stack([self.eeg[idx], self.eog[idx]], dim=0)  # (2, n_samples)
        else:
            raise ValueError(f"Unknown modality: {self.modality}")
        
        y = self.labels[idx]
        
        if self.transform:
            x = self.transform(x)
        
        return x, y
    
    def get_subject_ids(self) -> np.ndarray:
        """Get subject IDs for all samples"""
        return self.subject_ids
    
    @staticmethod
    def collate_fn(batch):
        """Custom collate function for batching"""
        x_batch = torch.stack([item[0] for item in batch])
        y_batch = torch.tensor([item[1] for item in batch])
        return x_batch, y_batch
    
    def get_class_weights(self) -> torch.Tensor:
        """Calculate class weights for imbalanced dataset"""
        class_counts = torch.bincount(self.labels)
        total = len(self.labels)
        weights = total / (len(class_counts) * class_counts.float())
        return weights
    
    def split_by_subject(
        self, 
        train_subjects: list, 
        val_subjects: list, 
        test_subjects: list
    ) -> Tuple['SleepEDFDataset', 'SleepEDFDataset', 'SleepEDFDataset']:
        """Split dataset by subject"""
        train_mask = np.isin(self.subject_ids, train_subjects)
        val_mask = np.isin(self.subject_ids, val_subjects)
        test_mask = np.isin(self.subject_ids, test_subjects)
        
        train_dataset = SleepEDFDataset(
            self.eeg[train_mask].numpy(),
            self.eog[train_mask].numpy(),
            self.labels[train_mask].numpy(),
            self.modality,
            self.transform,
            self.subject_ids[train_mask]
        )
        
        val_dataset = SleepEDFDataset(
            self.eeg[val_mask].numpy(),
            self.eog[val_mask].numpy(),
            self.labels[val_mask].numpy(),
            self.modality,
            self.transform,
            self.subject_ids[val_mask]
        )
        
        test_dataset = SleepEDFDataset(
            self.eeg[test_mask].numpy(),
            self.eog[test_mask].numpy(),
            self.labels[test_mask].numpy(),
            self.modality,
            self.transform,
            self.subject_ids[test_mask]
        )
        
        return train_dataset, val_dataset, test_dataset