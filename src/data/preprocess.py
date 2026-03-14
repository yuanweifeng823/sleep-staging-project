"""Signal preprocessing utilities"""

import numpy as np
from scipy import signal
from typing import Tuple, Optional, Dict, Union


def preprocess_signal(
    data: np.ndarray,
    sfreq: float,
    ch_type: str = 'eeg',
    lowcut: Optional[float] = None,
    highcut: Optional[float] = None,
    order: int = 4
) -> np.ndarray:
    """
    Apply bandpass filter to signal
    
    Args:
        data: Raw signal array
        sfreq: Sampling frequency
        ch_type: Channel type ('eeg' or 'eog')
        lowcut: Low cutoff frequency (if None, use default based on ch_type)
        highcut: High cutoff frequency (if None, use default based on ch_type)
        order: Filter order
    
    Returns:
        Filtered signal
    """
    # Set default frequency bands
    if lowcut is None or highcut is None:
        if ch_type == 'eeg':
            lowcut = 0.5
            highcut = 30.0
        else:  # eog
            lowcut = 0.5
            highcut = 10.0
    
    # Design Butterworth filter
    nyquist = 0.5 * sfreq
    low = lowcut / nyquist
    high = highcut / nyquist
    
    b, a = signal.butter(order, [low, high], btype='band')
    
    # Apply filter
    filtered_data = signal.filtfilt(b, a, data)
    
    return filtered_data


def create_epochs(
    eeg: np.ndarray,
    eog: np.ndarray,
    hypnogram: np.ndarray,
    sfreq: float,
    epoch_len: int = 30,
    remove_invalid: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create 30-second epochs from continuous signals
    
    Args:
        eeg: EEG signal array
        eog: EOG signal array
        hypnogram: Sleep stage annotations (per epoch)
        sfreq: Sampling frequency
        epoch_len: Epoch length in seconds
        remove_invalid: Whether to remove epochs with invalid labels
    
    Returns:
        Tuple of (eeg_epochs, eog_epochs, labels)
    """
    samples_per_epoch = int(sfreq * epoch_len)
    n_epochs_possible = len(eeg) // samples_per_epoch
    
    # Truncate to match hypnogram length
    n_epochs = min(n_epochs_possible, len(hypnogram))
    
    eeg_epochs = []
    eog_epochs = []
    labels = []
    
    for i in range(n_epochs):
        start = i * samples_per_epoch
        end = start + samples_per_epoch

        label = hypnogram[i]

        if not remove_invalid or label != -1:
            eeg_epochs.append(eeg[start:end])
            eog_epochs.append(eog[start:end])
            labels.append(label)

    if len(eeg_epochs) == 0:
        return (
            np.empty((0, samples_per_epoch), dtype=eeg.dtype),
            np.empty((0, samples_per_epoch), dtype=eog.dtype),
            np.empty((0,), dtype=int)
        )

    return np.array(eeg_epochs), np.array(eog_epochs), np.array(labels)


def normalize_epochs(
    epochs: np.ndarray,
    method: str = 'zscore',
    eps: float = 1e-8
) -> np.ndarray:
    """
    Normalize epochs
    
    Args:
        epochs: Array of shape (n_epochs, n_samples)
        method: Normalization method ('zscore' or 'minmax')
        eps: Small constant to avoid division by zero
    
    Returns:
        Normalized epochs
    """
    normalized = np.zeros_like(epochs)
    
    for i in range(epochs.shape[0]):
        epoch = epochs[i]
        
        if method == 'zscore':
            mean = np.mean(epoch)
            std = np.std(epoch)
            normalized[i] = (epoch - mean) / (std + eps)
        
        elif method == 'minmax':
            min_val = np.min(epoch)
            max_val = np.max(epoch)
            normalized[i] = (epoch - min_val) / (max_val - min_val + eps)
        
        else:
            raise ValueError(f"Unknown normalization method: {method}")
    
    return normalized


class Preprocessor:
    """Complete preprocessing pipeline"""
    
    def __init__(
        self,
        eeg_band: Tuple[float, float] = (0.5, 30.0),
        eog_band: Tuple[float, float] = (0.5, 10.0),
        epoch_len: int = 30,
        normalization: str = 'zscore'
    ):
        self.eeg_band = eeg_band
        self.eog_band = eog_band
        self.epoch_len = epoch_len
        self.normalization = normalization
    
    def process_subject(self, data: Dict) -> Dict:
        """
        Process all data for a single subject
        
        Args:
            data: Dictionary from load_sleep_edf
        
        Returns:
            Dictionary with processed epochs
        """
        # Filter signals
        eeg_filtered = preprocess_signal(
            data['eeg'], 
            data['sfreq'], 
            ch_type='eeg',
            lowcut=self.eeg_band[0],
            highcut=self.eeg_band[1]
        )
        
        eog_filtered = preprocess_signal(
            data['eog'],
            data['sfreq'],
            ch_type='eog',
            lowcut=self.eog_band[0],
            highcut=self.eog_band[1]
        )
        
        # Create epochs
        eeg_epochs, eog_epochs, labels = create_epochs(
            eeg_filtered,
            eog_filtered,
            data['hypnogram'],
            data['sfreq'],
            self.epoch_len
        )
        
        # Normalize
        eeg_epochs = normalize_epochs(eeg_epochs, self.normalization)
        eog_epochs = normalize_epochs(eog_epochs, self.normalization)
        
        return {
            'eeg': eeg_epochs,
            'eog': eog_epochs,
            'labels': labels,
            'subject_id': data['subject_id'],
            'n_epochs': len(labels)
        }