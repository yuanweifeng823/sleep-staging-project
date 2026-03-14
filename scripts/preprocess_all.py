#!/usr/bin/env python
"""Script to preprocess all subjects"""

import sys
from pathlib import Path

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import logging
from src.data.loader import DataLoader
from src.data.preprocess import Preprocessor
from src.data.split import create_data_splits, save_splits
from src.utils.logger import setup_logger
from src.utils.paths import paths

logger = setup_logger(__name__)


def main():
    """Main preprocessing function"""
    # Initialize components
    loader = DataLoader(str(paths.data_raw))
    preprocessor = Preprocessor()
    
    # Get all subjects
    subjects = loader.get_available_subjects()
    logger.info(f"Found {len(subjects)} subjects")
    
    # Create data splits
    splits = create_data_splits(subjects, n_folds=5, test_fold=0)
    
    # Process all subjects
    all_data = []
    for subject_id in subjects:
        logger.info(f"Processing {subject_id}...")
        
        # Load raw data
        raw_data = loader.load_subject(subject_id)
        
        # Preprocess
        processed = preprocessor.process_subject(raw_data)
        
        # Add split information
        for split_name, split_subjects in splits.items():
            if subject_id in split_subjects:
                processed['split'] = split_name
                break
        
        all_data.append(processed)
    
    # Save processed data
    save_dir = paths.data_processed
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Save as numpy arrays
    eeg_all = np.concatenate([d['eeg'] for d in all_data])
    eog_all = np.concatenate([d['eog'] for d in all_data])
    labels_all = np.concatenate([d['labels'] for d in all_data])
    splits_all = np.concatenate([[d['split']] * len(d['labels']) 
                                 for d in all_data])
    
    np.save(save_dir / 'eeg_epochs.npy', eeg_all)
    np.save(save_dir / 'eog_epochs.npy', eog_all)
    np.save(save_dir / 'labels.npy', labels_all)
    np.save(save_dir / 'splits.npy', splits_all)
    
    # Save subject information
    subject_info = {
        'subject_ids': [d['subject_id'] for d in all_data],
        'n_epochs': [d['n_epochs'] for d in all_data]
    }
    np.save(save_dir / 'subject_info.npy', subject_info)
    
    # Save splits
    save_splits(splits, save_dir / 'splits_config.npy')
    
    logger.info(f"Preprocessing complete!")
    logger.info(f"Total epochs: {len(labels_all)}")
    logger.info(f"Class distribution: {np.bincount(labels_all)}")


if __name__ == "__main__":
    main()