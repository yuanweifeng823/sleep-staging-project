"""PSD (Power Spectral Density) feature extraction"""

import numpy as np
from scipy import signal
from typing import Dict, List, Tuple, Optional


def extract_psd_features(
    eeg_epoch: np.ndarray,
    sfreq: int = 100,
    nperseg: int = 256,
    bands: Optional[Dict[str, Tuple[float, float]]] = None
) -> np.ndarray:
    """
    Extract power spectral density features
    
    Args:
        eeg_epoch: Single EEG epoch, shape (n_samples,)
        sfreq: Sampling frequency
        nperseg: Window length for Welch's method
        bands: Dictionary of frequency bands
    
    Returns:
        Feature array
    """
    if bands is None:
        bands = {
            'delta': (0.5, 4),
            'theta': (4, 8),
            'alpha': (8, 12),
            'sigma': (12, 16),
            'beta': (16, 30)
        }
    
    # Compute PSD using Welch's method
    freqs, psd = signal.welch(eeg_epoch, fs=sfreq, nperseg=nperseg)
    
    # Extract band powers
    features = []
    band_powers = {}
    
    for band_name, (low, high) in bands.items():
        mask = (freqs >= low) & (freqs <= high)
        band_power = np.mean(psd[mask])
        band_powers[band_name] = band_power
        features.append(band_power)
    
    # Calculate ratio features
    theta_beta = band_powers['theta'] / (band_powers['beta'] + 1e-8)
    delta_alpha = band_powers['delta'] / (band_powers['alpha'] + 1e-8)
    alpha_beta = band_powers['alpha'] / (band_powers['beta'] + 1e-8)
    
    features.extend([theta_beta, delta_alpha, alpha_beta])
    
    return np.array(features)


def extract_psd_features_batch(
    eeg_epochs: np.ndarray,
    sfreq: int = 100,
    nperseg: int = 256
) -> np.ndarray:
    """
    Extract PSD features for a batch of epochs
    
    Args:
        eeg_epochs: Batch of EEG epochs, shape (n_epochs, n_samples)
        sfreq: Sampling frequency
        nperseg: Window length
    
    Returns:
        Feature matrix, shape (n_epochs, n_features)
    """
    features = []
    for i in range(eeg_epochs.shape[0]):
        feat = extract_psd_features(eeg_epochs[i], sfreq, nperseg)
        features.append(feat)
    
    return np.array(features)


class PSDExtractor:
    """PSD feature extractor with caching"""
    
    def __init__(
        self,
        sfreq: int = 100,
        nperseg: int = 256,
        bands: Optional[Dict[str, Tuple[float, float]]] = None
    ):
        self.sfreq = sfreq
        self.nperseg = nperseg
        self.bands = bands or {
            'delta': (0.5, 4),
            'theta': (4, 8),
            'alpha': (8, 12),
            'sigma': (12, 16),
            'beta': (16, 30)
        }
        self.cache = {}
    
    def extract(self, eeg_epochs: np.ndarray) -> np.ndarray:
        """
        Extract features with caching
        
        Args:
            eeg_epochs: EEG epochs, shape (n_epochs, n_samples)
        
        Returns:
            Feature matrix
        """
        # Simple hash for caching (use with caution for large datasets)
        data_hash = hash(eeg_epochs.tobytes())
        
        if data_hash in self.cache:
            return self.cache[data_hash].copy()
        
        features = extract_psd_features_batch(
            eeg_epochs,
            self.sfreq,
            self.nperseg
        )
        
        self.cache[data_hash] = features.copy()
        return features
    
    def get_feature_names(self) -> List[str]:
        """Get names of extracted features"""
        names = list(self.bands.keys())
        names.extend(['theta_beta', 'delta_alpha', 'alpha_beta'])
        return names
    
    def get_n_features(self) -> int:
        """Get number of features"""
        return len(self.bands) + 3