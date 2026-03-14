"""Data splitting utilities"""

import numpy as np
from sklearn.model_selection import GroupKFold, train_test_split
from typing import List, Tuple, Optional, Dict
import logging

logger = logging.getLogger(__name__)


def create_data_splits(
    subject_ids: List[str],
    n_folds: int = 5,
    test_fold: int = 0,
    val_size: float = 0.1,
    random_seed: int = 42
) -> Dict[str, List[str]]:
    """
    Create subject-level data splits
    
    All members should use the same split for fair comparison.
    
    Args:
        subject_ids: List of all subject IDs
        n_folds: Number of cross-validation folds
        test_fold: Which fold to use as test set
        val_size: Proportion of training data for validation
        random_seed: Random seed for reproducibility
    
    Returns:
        Dictionary with 'train', 'val', 'test' subject lists
    """
    np.random.seed(random_seed)
    subject_ids = np.array(subject_ids)
    n_subjects = len(subject_ids)
    
    # Create fold indices
    fold_indices = np.array_split(np.random.permutation(n_subjects), n_folds)
    
    # Test set is the specified fold
    test_idx = fold_indices[test_fold]
    test_subjects = subject_ids[test_idx].tolist()
    
    # Remaining subjects for training/validation
    train_val_idx = np.concatenate([fold_indices[i] for i in range(n_folds) if i != test_fold])
    train_val_subjects = subject_ids[train_val_idx]
    
    # Split train/val
    train_idx, val_idx = train_test_split(
        np.arange(len(train_val_subjects)),
        test_size=val_size,
        random_state=random_seed,
        shuffle=True
    )
    
    train_subjects = train_val_subjects[train_idx].tolist()
    val_subjects = train_val_subjects[val_idx].tolist()
    
    logger.info(f"Data split: {len(train_subjects)} train, "
                f"{len(val_subjects)} val, {len(test_subjects)} test subjects")
    
    return {
        'train': train_subjects,
        'val': val_subjects,
        'test': test_subjects
    }


def get_subject_folds(
    subject_ids: List[str],
    n_folds: int = 5
) -> List[Tuple[List[str], List[str]]]:
    """
    Get cross-validation folds
    
    Returns:
        List of (train_ids, test_ids) tuples
    """
    groups = np.arange(len(subject_ids))
    gkf = GroupKFold(n_splits=n_folds)
    
    folds = []
    for train_idx, test_idx in gkf.split(subject_ids, groups=groups):
        train_ids = [subject_ids[i] for i in train_idx]
        test_ids = [subject_ids[i] for i in test_idx]
        folds.append((train_ids, test_ids))
    
    return folds


def save_splits(splits: Dict, save_path: str):
    """Save splits to file"""
    np.save(save_path, splits)
    logger.info(f"Splits saved to {save_path}")


def load_splits(load_path: str) -> Dict:
    """Load splits from file"""
    splits = np.load(load_path, allow_pickle=True).item()
    logger.info(f"Splits loaded from {load_path}")
    return splits