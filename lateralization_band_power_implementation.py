"""
Improved lateralization calculation using band power (RECOMMENDED).

This implementation calculates lateralization index using mu (8-13 Hz) and beta (13-30 Hz)
band power instead of raw amplitude, which is the standard approach for motor tasks.

Author: Analysis Review
Date: 2025-11-11
"""

import numpy as np
import mne
from typing import Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


def extract_lateralization_band_power(
    epochs_dict: Dict,
    ch_names: list = ['C3', 'C4'],
    bands: Dict[str, Tuple[float, float]] = None,
    method: str = 'welch'
) -> Dict:
    """
    Extract lateralization features using band power (RECOMMENDED for motor tasks).
    
    Parameters:
    -----------
    epochs_dict : Dict
        Dictionary with keys like 'left_real', 'right_real', 'left_imagined', 'right_imagined'
        Each value should be an MNE Epochs object
    ch_names : list
        Channel names to use (default: ['C3', 'C4'])
    bands : Dict[str, Tuple[float, float]]
        Frequency bands to use. Default: {'mu': (8, 13), 'beta': (13, 30)}
    method : str
        Method for power calculation: 'welch' (PSD) or 'morlet' (time-frequency)
    
    Returns:
    --------
    Dict with lateralization indices for each condition and band
    """
    if bands is None:
        bands = {
            'mu': (8, 13),      # Mu rhythm (motor cortex)
            'beta': (13, 30)    # Beta band (motor cortex)
        }
    
    features = {
        'lateralization_index': {
            'real': {'left': {}, 'right': {}},
            'imagined': {'left': {}, 'right': {}}
        },
        'times': None
    }
    
    # Check if we have the required channels
    sample_epochs = None
    for key in ['left_real', 'right_real', 'left_imagined', 'right_imagined']:
        if key in epochs_dict and epochs_dict[key] is not None:
            sample_epochs = epochs_dict[key]
            break
    
    if sample_epochs is None:
        logger.warning("No epochs found for lateralization calculation")
        return features
    
    # Verify channels exist
    available_chs = [ch for ch in ch_names if ch in sample_epochs.ch_names]
    if len(available_chs) < 2:
        logger.warning(f"Required channels {ch_names} not found. Available: {sample_epochs.ch_names}")
        return features
    
    c3_idx = sample_epochs.ch_names.index('C3') if 'C3' in sample_epochs.ch_names else None
    c4_idx = sample_epochs.ch_names.index('C4') if 'C4' in sample_epochs.ch_names else None
    
    if c3_idx is None or c4_idx is None:
        logger.warning("C3 or C4 not found in channels")
        return features
    
    times = sample_epochs.times
    features['times'] = times
    
    # Process each condition
    for condition in ['real', 'imagined']:
        for hand in ['left', 'right']:
            key = f'{hand}_{condition}'
            
            if key not in epochs_dict or epochs_dict[key] is None:
                continue
            
            epochs = epochs_dict[key]
            
            # Calculate band power for each band
            for band_name, (fmin, fmax) in bands.items():
                if method == 'welch':
                    # Use Welch's method (simpler, faster)
                    # Compute PSD for each epoch, then average
                    psd_list = []
                    for epoch_idx in range(len(epochs)):
                        # Extract single epoch
                        single_epoch = epochs[epoch_idx:epoch_idx+1]
                        
                        # Compute PSD
                        psd = single_epoch.compute_psd(
                            method='welch',
                            fmin=fmin,
                            fmax=fmax,
                            n_fft=min(256, len(single_epoch.times) * int(single_epoch.info['sfreq'])),
                            n_overlap=None,
                            verbose=False
                        )
                        psd_data, freqs = psd.get_data(return_freqs=True)
                        
                        # Extract C3 and C4 power, average across frequencies
                        c3_power = psd_data[0, c3_idx, :].mean()  # Average across frequencies
                        c4_power = psd_data[0, c4_idx, :].mean()
                        
                        psd_list.append({'c3': c3_power, 'c4': c4_power})
                    
                    # Average across epochs
                    c3_mean = np.mean([p['c3'] for p in psd_list])
                    c4_mean = np.mean([p['c4'] for p in psd_list])
                    
                    # Calculate lateralization index
                    li = (c3_mean - c4_mean) / (c3_mean + c4_mean + 1e-10)
                    
                    # Store single value (time-averaged)
                    features['lateralization_index'][condition][hand][band_name] = li
                
                elif method == 'morlet':
                    # Use Morlet wavelets (time-frequency, more accurate)
                    freqs = np.arange(fmin, fmax + 1, 1)
                    
                    # Compute time-frequency representation
                    power = mne.time_frequency.tfr_morlet(
                        epochs,
                        freqs=freqs,
                        n_cycles=freqs / 2,
                        use_fft=True,
                        return_itc=False,
                        average=False,  # Keep individual epochs
                        n_jobs=1,
                        verbose=False
                    )
                    
                    # Extract C3 and C4 power
                    c3_power = power.data[:, c3_idx, :, :]  # [epochs, freqs, times]
                    c4_power = power.data[:, c4_idx, :, :]
                    
                    # Average across epochs and frequencies
                    c3_mean = c3_power.mean(axis=(0, 1))  # [times]
                    c4_mean = c4_power.mean(axis=(0, 1))
                    
                    # Calculate lateralization index over time
                    li_time = (c3_mean - c4_mean) / (c3_mean + c4_mean + 1e-10)
                    
                    # Store time series
                    features['lateralization_index'][condition][hand][band_name] = li_time
                    features['times'] = power.times
    
    # Calculate combined lateralization (weighted average of mu and beta)
    for condition in ['real', 'imagined']:
        for hand in ['left', 'right']:
            if 'mu' in features['lateralization_index'][condition][hand] and \
               'beta' in features['lateralization_index'][condition][hand]:
                
                mu_li = features['lateralization_index'][condition][hand]['mu']
                beta_li = features['lateralization_index'][condition][hand]['beta']
                
                # Handle both scalar and array cases
                if isinstance(mu_li, np.ndarray) and isinstance(beta_li, np.ndarray):
                    # Time series: weighted average
                    combined = 0.6 * mu_li + 0.4 * beta_li
                else:
                    # Scalar: weighted average
                    combined = 0.6 * mu_li + 0.4 * beta_li
                
                features['lateralization_index'][condition][hand]['combined'] = combined
    
    return features


def extract_lateralization_erd(
    epochs_dict: Dict,
    baseline_epochs: Optional[mne.Epochs] = None,
    ch_names: list = ['C3', 'C4'],
    bands: Dict[str, Tuple[float, float]] = None
) -> Dict:
    """
    Extract lateralization using ERD/ERS (MOST ACCURATE for motor tasks).
    
    ERD (Event-Related Desynchronization) = power decrease relative to baseline
    This directly measures motor cortex activation.
    
    Parameters:
    -----------
    epochs_dict : Dict
        Dictionary with task epochs
    baseline_epochs : mne.Epochs, optional
        Baseline epochs for ERD calculation. If None, uses pre-movement period (-1 to 0s)
    ch_names : list
        Channel names to use
    bands : Dict[str, Tuple[float, float]]
        Frequency bands (default: mu + beta)
    
    Returns:
    --------
    Dict with ERD-based lateralization indices
    """
    if bands is None:
        bands = {
            'mu': (8, 13),
            'beta': (13, 30)
        }
    
    features = {
        'lateralization_index': {
            'real': {'left': {}, 'right': {}},
            'imagined': {'left': {}, 'right': {}}
        },
        'times': None
    }
    
    # Get sample epochs to check channels
    sample_epochs = None
    for key in ['left_real', 'right_real']:
        if key in epochs_dict and epochs_dict[key] is not None:
            sample_epochs = epochs_dict[key]
            break
    
    if sample_epochs is None:
        return features
    
    c3_idx = sample_epochs.ch_names.index('C3') if 'C3' in sample_epochs.ch_names else None
    c4_idx = sample_epochs.ch_names.index('C4') if 'C4' in sample_epochs.ch_names else None
    
    if c3_idx is None or c4_idx is None:
        return features
    
    # Process each condition
    for condition in ['real', 'imagined']:
        for hand in ['left', 'right']:
            key = f'{hand}_{condition}'
            
            if key not in epochs_dict or epochs_dict[key] is None:
                continue
            
            epochs = epochs_dict[key]
            
            # Use pre-movement baseline if no baseline provided
            if baseline_epochs is None:
                baseline = epochs.copy().crop(tmin=-1.0, tmax=0.0)
            else:
                baseline = baseline_epochs
            
            # Compute time-frequency for task and baseline
            all_freqs = []
            for fmin, fmax in bands.values():
                all_freqs.extend(np.arange(fmin, fmax + 1, 1))
            freqs = np.unique(all_freqs)
            
            # Task power
            task_power = mne.time_frequency.tfr_morlet(
                epochs,
                freqs=freqs,
                n_cycles=freqs / 2,
                use_fft=True,
                return_itc=False,
                average=True,
                n_jobs=1,
                verbose=False
            )
            
            # Baseline power
            baseline_power = mne.time_frequency.tfr_morlet(
                baseline,
                freqs=freqs,
                n_cycles=freqs / 2,
                use_fft=True,
                return_itc=False,
                average=True,
                n_jobs=1,
                verbose=False
            )
            
            # Calculate ERD: (task - baseline) / baseline * 100
            baseline_mean = baseline_power.data.mean(axis=-1, keepdims=True)
            erd = ((task_power.data - baseline_mean) / baseline_mean) * 100
            
            # Extract C3 and C4 ERD
            c3_erd = erd[c3_idx, :, :].mean(axis=0)  # Average across frequencies [times]
            c4_erd = erd[c4_idx, :, :].mean(axis=0)
            
            # Lateralization index on ERD
            # Negative ERD = activation, so for left hand: C4 should be more negative
            # LI = (C3_ERD - C4_ERD) / (|C3_ERD| + |C4_ERD|)
            li_erd = (c3_erd - c4_erd) / (np.abs(c3_erd) + np.abs(c4_erd) + 1e-10)
            
            features['lateralization_index'][condition][hand]['erd'] = li_erd
            features['times'] = task_power.times
    
    return features


# Example usage function
def compare_lateralization_methods(
    left_real_epochs: mne.Epochs,
    right_real_epochs: mne.Epochs,
    left_imag_epochs: mne.Epochs,
    right_imag_epochs: mne.Epochs,
    baseline_epochs: Optional[mne.Epochs] = None
) -> Dict:
    """
    Compare different lateralization calculation methods.
    
    Returns results from:
    1. Raw amplitude (current method)
    2. Band power (recommended)
    3. ERD-based (most accurate)
    """
    epochs_dict = {
        'left_real': left_real_epochs,
        'right_real': right_real_epochs,
        'left_imagined': left_imag_epochs,
        'right_imagined': right_imag_epochs
    }
    
    results = {}
    
    # Method 1: Band power (Welch)
    results['band_power_welch'] = extract_lateralization_band_power(
        epochs_dict, method='welch'
    )
    
    # Method 2: Band power (Morlet - time-frequency)
    results['band_power_morlet'] = extract_lateralization_band_power(
        epochs_dict, method='morlet'
    )
    
    # Method 3: ERD-based
    if baseline_epochs is not None:
        results['erd_based'] = extract_lateralization_erd(
            epochs_dict, baseline_epochs=baseline_epochs
        )
    
    return results

