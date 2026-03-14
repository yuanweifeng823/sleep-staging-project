"""STFT (Short-Time Fourier Transform) utilities"""

import numpy as np
from scipy import signal
import torch
from typing import Optional, Tuple


def compute_stft(
    eeg_epoch: np.ndarray,
    sfreq: int = 100,
    nperseg: int = 256,
    noverlap: Optional[int] = None,
    freq_range: Tuple[float, float] = (0, 30)
) -> np.ndarray:
    """
    Compute STFT for a single epoch
    
    Args:
        eeg_epoch: Single EEG epoch, shape (n_samples,)
        sfreq: Sampling frequency
        nperseg: Length of each segment
        noverlap: Number of points to overlap (if None, use nperseg//2)
        freq_range: Frequency range to keep (min, max)
    
    Returns:
        Spectrogram: shape (n_freqs, n_times)
    """
    if noverlap is None:
        noverlap = nperseg // 2
    
    # Compute STFT
    f, t, Zxx = signal.stft(
        eeg_epoch,
        fs=sfreq,
        nperseg=nperseg,
        noverlap=noverlap
    )
    
    # Keep only specified frequency range
    freq_mask = (f >= freq_range[0]) & (f <= freq_range[1])
    spectrogram = np.abs(Zxx[freq_mask, :])
    
    # Log transform for better visualization
    spectrogram = np.log1p(spectrogram)
    
    return spectrogram


def batch_stft(
    eeg_batch: torch.Tensor,
    sfreq: int = 100,
    nperseg: int = 256
) -> torch.Tensor:
    """
    Compute STFT for a batch of EEG epochs
    
    Args:
        eeg_batch: Batch of EEG epochs, shape (batch, 1, n_samples)
        sfreq: Sampling frequency
        nperseg: STFT window size
    
    Returns:
        Batch of spectrograms, shape (batch, 1, n_freqs, n_times)
    """
    batch_size = eeg_batch.shape[0]
    specs = []
    
    for i in range(batch_size):
        # Remove channel dimension
        epoch = eeg_batch[i, 0].numpy()
        spec = compute_stft(epoch, sfreq, nperseg)
        specs.append(spec)
    
    # Stack and add channel dimension
    specs_array = np.array(specs)
    specs_tensor = torch.FloatTensor(specs_array).unsqueeze(1)
    
    return specs_tensor


class STFTTransformer:
    """Transform EEG epochs to spectrograms"""
    
    def __init__(
        self,
        sfreq: int = 100,
        nperseg: int = 256,
        noverlap: Optional[int] = None,
        freq_range: Tuple[float, float] = (0, 30)
    ):
        self.sfreq = sfreq
        self.nperseg = nperseg
        self.noverlap = noverlap if noverlap is not None else nperseg // 2
        self.freq_range = freq_range
    
    def __call__(self, eeg_epoch: torch.Tensor) -> torch.Tensor:
        """
        Transform a single epoch
        
        Args:
            eeg_epoch: Tensor of shape (1, n_samples)
        
        Returns:
            Spectrogram tensor of shape (1, n_freqs, n_times)
        """
        epoch_np = eeg_epoch.numpy().squeeze()
        spec = compute_stft(
            epoch_np,
            self.sfreq,
            self.nperseg,
            self.noverlap,
            self.freq_range
        )
        return torch.FloatTensor(spec).unsqueeze(0)
    
    def get_output_shape(self, input_length: int) -> Tuple[int, int]:
        """Get output spectrogram shape"""
        n_freqs = self._get_n_freqs()
        n_times = self._get_n_times(input_length)
        return (n_freqs, n_times)
    
    def _get_n_freqs(self) -> int:
        """Calculate number of frequency bins"""
        # Rough estimate: nperseg//2 + 1 within frequency range
        total_freqs = self.nperseg // 2 + 1
        max_freq_idx = int(total_freqs * self.freq_range[1] / (self.sfreq / 2))
        min_freq_idx = int(total_freqs * self.freq_range[0] / (self.sfreq / 2))
        return max_freq_idx - min_freq_idx
    
    def _get_n_times(self, input_length: int) -> int:
        """Calculate number of time bins"""
        return 1 + (input_length - self.nperseg) // (self.nperseg - self.noverlap)