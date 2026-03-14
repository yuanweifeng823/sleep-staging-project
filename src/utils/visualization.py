"""Visualization utilities for sleep data"""

import matplotlib.pyplot as plt
import numpy as np
from typing import Optional, List, Tuple
import seaborn as sns


def plot_waveform(
    eeg_signal: np.ndarray,
    eog_signal: Optional[np.ndarray] = None,
    sfreq: int = 100,
    duration: Optional[float] = None,
    title: str = "EEG/EOG Waveform",
    figsize: Tuple[int, int] = (12, 6),
    save_path: Optional[str] = None
):
    """
    Plot EEG/EOG waveform
    
    Args:
        eeg_signal: EEG signal array
        eog_signal: EOG signal array (optional)
        sfreq: Sampling frequency
        duration: Duration to plot (seconds), if None plot all
        title: Plot title
        figsize: Figure size
        save_path: Path to save figure
    """
    if duration is not None:
        n_samples = int(duration * sfreq)
        eeg_signal = eeg_signal[:n_samples]
        if eog_signal is not None:
            eog_signal = eog_signal[:n_samples]
    
    time = np.arange(len(eeg_signal)) / sfreq
    
    if eog_signal is not None:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)
        
        ax1.plot(time, eeg_signal, color='black', linewidth=0.5)
        ax1.set_ylabel('EEG (μV)')
        ax1.set_title('EEG')
        ax1.grid(True, alpha=0.3)
        
        ax2.plot(time, eog_signal, color='blue', linewidth=0.5)
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('EOG (μV)')
        ax2.set_title('EOG')
        ax2.grid(True, alpha=0.3)
    else:
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(time, eeg_signal, color='black', linewidth=0.5)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('EEG (μV)')
        ax.grid(True, alpha=0.3)
    
    plt.suptitle(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_hypnogram(
    stages: np.ndarray,
    epoch_length: int = 30,
    title: str = "Hypnogram",
    figsize: Tuple[int, int] = (12, 4),
    save_path: Optional[str] = None
):
    """
    Plot sleep hypnogram
    
    Args:
        stages: Sleep stage labels (0-4)
        epoch_length: Length of each epoch in seconds
        title: Plot title
        figsize: Figure size
        save_path: Path to save figure
    """
    stage_names = ['W', 'N1', 'N2', 'N3', 'REM']
    stage_colors = ['#FF9999', '#66B2FF', '#99FF99', '#FFCC99', '#FF99FF']
    
    time_hours = np.arange(len(stages)) * epoch_length / 3600
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot each stage segment
    for i in range(len(stages)):
        ax.barh(0, width=epoch_length/3600, left=time_hours[i],
                color=stage_colors[stages[i]], edgecolor='none', height=0.5)
    
    # Add stage labels
    ax.set_yticks([])
    ax.set_xlabel('Time (hours)')
    ax.set_title(title)
    
    # Add legend
    legend_elements = [plt.Rectangle((0,0),1,1, facecolor=color, label=name)
                      for name, color in zip(stage_names, stage_colors)]
    ax.legend(handles=legend_elements, loc='upper right')
    
    ax.set_xlim(0, time_hours[-1])
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_stage_distribution(
    labels: np.ndarray,
    title: str = "Sleep Stage Distribution",
    figsize: Tuple[int, int] = (8, 5),
    save_path: Optional[str] = None
):
    """
    Plot distribution of sleep stages
    
    Args:
        labels: Sleep stage labels
        title: Plot title
        figsize: Figure size
        save_path: Path to save figure
    """
    stage_names = ['W', 'N1', 'N2', 'N3', 'REM']
    colors = ['#FF9999', '#66B2FF', '#99FF99', '#FFCC99', '#FF99FF']
    
    counts = np.bincount(labels, minlength=5)
    percentages = counts / len(labels) * 100
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Bar plot
    bars = ax1.bar(stage_names, counts, color=colors)
    ax1.set_xlabel('Sleep Stage')
    ax1.set_ylabel('Count')
    ax1.set_title('Stage Counts')
    
    # Add count labels
    for bar, count in zip(bars, counts):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                str(count), ha='center', va='bottom')
    
    # Pie chart
    ax2.pie(percentages, labels=stage_names, colors=colors, autopct='%1.1f%%',
            startangle=90)
    ax2.set_title('Stage Percentages')
    
    plt.suptitle(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig