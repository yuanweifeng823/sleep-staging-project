"""Data loading utilities for Sleep-EDFx dataset"""

import mne
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union
from ..utils.logger import setup_logger

logger = setup_logger(__name__)


def load_sleep_edf(
    subject_id: str, 
    data_path: Union[str, Path] = None,
    verbose: bool = False
) -> Dict:
    """
    Load PSG and annotation data for a single subject
    
    Args:
        subject_id: Subject identifier (e.g., 'SC4001E0')
        data_path: Path to raw data directory
        verbose: Whether to print MNE verbose output
    
    Returns:
        Dictionary containing:
            - eeg: EEG signal array
            - eog: EOG signal array
            - hypnogram: Sleep stage annotations
            - sfreq: Sampling frequency
    """
    data_path = Path(data_path)
    
    # File paths
    psg_path = data_path / f'{subject_id}-PSG.edf'
    hyp_path = data_path / f'{subject_id}-Hypnogram.edf'
    
    if not psg_path.exists():
        raise FileNotFoundError(f"PSG file not found: {psg_path}")

    # Hypnogram filename may not use the same suffix (e.g., SC4001E0 vs SC4001EC)
    if not hyp_path.exists():
        fallback_prefix = subject_id[:6]
        hyp_candidates = sorted(data_path.glob(f"{fallback_prefix}*-Hypnogram.edf"))
        if len(hyp_candidates) == 1:
            hyp_path = hyp_candidates[0]
            logger.warning(
                f"Hypnogram file for {subject_id} not found exactly; using fallback {hyp_path.name}"
            )
        elif len(hyp_candidates) > 1:
            hyp_path = hyp_candidates[0]
            logger.warning(
                f"Multiple hypnogram candidates found for {subject_id}: "
                f"{[p.name for p in hyp_candidates]}. Using {hyp_path.name}"
            )
        else:
            raise FileNotFoundError(f"Hypnogram file not found: {hyp_path}")
    
    # Read PSG data
    logger.info(f"Loading PSG data for {subject_id}")
    raw = mne.io.read_raw_edf(psg_path, preload=True, verbose=False)
    
    # Select channels (unified to use Fpz-Cz as EEG)
    ch_names = raw.ch_names
    eeg_ch = 'EEG Fpz-Cz' if 'EEG Fpz-Cz' in ch_names else ch_names[0]
    eog_ch = 'EOG horizontal' if 'EOG horizontal' in ch_names else ch_names[1]
    
    if verbose:
        logger.info(f"Channels found: {ch_names}")
        logger.info(f"Using EEG: {eeg_ch}, EOG: {eog_ch}")
    
    # Extract signals
    eeg_data = raw.copy().pick_channels([eeg_ch]).get_data()[0]
    eog_data = raw.copy().pick_channels([eog_ch]).get_data()[0]
    sfreq = raw.info['sfreq']
    
    # Read annotations (prefer read_annotations for hypnogram-only EDFs)
    try:
        annot = mne.read_annotations(str(hyp_path))
        if len(annot) == 0:
            raise ValueError("No annotations found via read_annotations")
    except Exception:
        raw_hyp = mne.io.read_raw_edf(hyp_path, preload=True, verbose=False)
        annot = raw_hyp.annotations

    # Convert annotations to epoch labels
    def _map_stage(desc: str) -> int:
        stage = desc.strip().upper().replace('SLEEP STAGE', '').replace('STAGE', '').strip()
        if stage in {'W', 'WAKE'}:
            return 0
        if stage in {'1', 'N1'}:
            return 1
        if stage in {'2', 'N2'}:
            return 2
        if stage in {'3', '4', 'N3'}:
            return 3
        if stage in {'R', 'REM'}:
            return 4
        return -1

    hypnogram = []
    for desc in annot.description:
        hypnogram.append(_map_stage(desc))
    
    return {
        'eeg': eeg_data,
        'eog': eog_data,
        'hypnogram': np.array(hypnogram),
        'sfreq': sfreq,
        'subject_id': subject_id
    }


class DataLoader:
    """Batch data loader for multiple subjects"""
    
    def __init__(self, data_path: Union[str, Path] = None):
        if data_path is None:
            from ..utils.paths import paths
            self.data_path = paths.data_raw
        else:
            self.data_path = Path(data_path)
        self.cache = {}
    
    def get_available_subjects(self) -> List[str]:
        """Get list of available subject IDs"""
        psg_files = list(self.data_path.glob('*-PSG.edf'))
        subjects = [f.name.replace('-PSG.edf', '') for f in psg_files]
        return sorted(subjects)
    
    def load_subject(self, subject_id: str, use_cache: bool = True) -> Dict:
        """Load a single subject with optional caching"""
        if use_cache and subject_id in self.cache:
            return self.cache[subject_id]
        
        data = load_sleep_edf(subject_id, self.data_path)
        
        if use_cache:
            self.cache[subject_id] = data
        
        return data
    
    def load_all_subjects(
        self, 
        subject_ids: Optional[List[str]] = None,
        use_cache: bool = True
    ) -> List[Dict]:
        """Load multiple subjects"""
        if subject_ids is None:
            subject_ids = self.get_available_subjects()
        
        all_data = []
        for sid in subject_ids:
            try:
                data = self.load_subject(sid, use_cache)
                all_data.append(data)
                logger.info(f"Loaded {sid}")
            except Exception as e:
                logger.error(f"Failed to load {sid}: {e}")
        
        return all_data