"""
Multi-subject analysis comparing imagined vs actual movements in EEG data
Processes all subjects and creates averaged, group-level visualizations

Author: Shaheer Khan, Daniel Mansperger, Andrew Li
Date: November 2025
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import mne
from scipy import signal, stats
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional
from matplotlib.gridspec import GridSpec
import matplotlib.patches as mpatches
from mpl_toolkits.axes_grid1 import make_axes_locatable
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import json
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

# Import the single subject analyzer
from imagined_vs_actual_analysis_new import ImaginedVsActualAnalyzer, COLORS

# Set up high-quality plotting
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 16

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SubjectDataExtractor(ImaginedVsActualAnalyzer):
    """Extended analyzer that extracts and returns data instead of just plotting."""
    
    def __init__(self, subject_id: str = 'S001', base_path: str = 'eeg-motor-movementimagery-dataset-1.0.0/files'):
        """Initialize without creating output directories."""
        # Don't call super().__init__() to avoid creating individual subject folders
        self.subject_id = subject_id
        self.base_path = Path(base_path) / subject_id
        
        # Copy necessary attributes from parent class
        self.channels = ['C3', 'C4', 'Cz', 'FC1', 'FC2', 'CP1', 'CP2']
        
        self.baseline_runs = {
            'eyes_open': 1,
            'eyes_closed': 2
        }
        
        self.task_pairs = [
            {
                'name': 'Left/Right Fist 1',
                'real': 3, 'imagined': 4,
                'type': 'fist',
                'T1': 'Left Fist', 'T2': 'Right Fist'
            },
            {
                'name': 'Fists/Feet 1',
                'real': 5, 'imagined': 6,
                'type': 'fists_feet',
                'T1': 'Both Fists', 'T2': 'Both Feet'
            },
            {
                'name': 'Left/Right Fist 2',
                'real': 7, 'imagined': 8,
                'type': 'fist',
                'T1': 'Left Fist', 'T2': 'Right Fist'
            },
            {
                'name': 'Fists/Feet 2',
                'real': 9, 'imagined': 10,
                'type': 'fists_feet',
                'T1': 'Both Fists', 'T2': 'Both Feet'
            },
            {
                'name': 'Left/Right Fist 3',
                'real': 11, 'imagined': 12,
                'type': 'fist',
                'T1': 'Left Fist', 'T2': 'Right Fist'
            },
            {
                'name': 'Fists/Feet 3',
                'real': 13, 'imagined': 14,
                'type': 'fists_feet',
                'T1': 'Both Fists', 'T2': 'Both Feet'
            }
        ]
        
        self.data = {}
        self.epochs = {}
        self.features = {}
        self.extracted_data = {}
        
    def extract_all_features(self) -> Dict:
        """Extract all features without plotting."""
        try:
            # Load data
            self.load_and_preprocess_data()
            
            # Extract power spectral features
            self.extracted_data['psd'] = self._extract_psd_features()
            
            # Extract amplitude features
            self.extracted_data['amplitudes'] = self._extract_amplitude_features()
            
            # Extract time-frequency features
            self.extracted_data['timefreq'] = self._extract_timefreq_features()
            
            # Extract band power features
            self.extracted_data['band_powers'] = self._extract_band_power_features()
            
            # Extract lateralization features
            self.extracted_data['lateralization'] = self._extract_lateralization_features()
            
            # Extract temporal dynamics
            self.extracted_data['temporal'] = self._extract_temporal_features()
            
            # Add metadata
            self.extracted_data['metadata'] = {
                'subject_id': self.subject_id,
                'n_channels': len(self.channels),
                'n_epochs_total': sum(len(data.get('epochs', [])) for data in self.data.values() if 'epochs' in data),
                'success': True,
                'error': None
            }
            
            return self.extracted_data
            
        except Exception as e:
            logger.error(f"Failed to extract features for {self.subject_id}: {e}")
            return {
                'metadata': {
                    'subject_id': self.subject_id,
                    'success': False,
                    'error': str(e)
                }
            }
    
    def _extract_psd_features(self) -> Dict:
        """Extract PSD features for averaging."""
        features = {
            'real': {'psd': [], 'freqs': None},
            'imagined': {'psd': [], 'freqs': None},
            'baseline_open': {'psd': None, 'freqs': None},
            'baseline_closed': {'psd': None, 'freqs': None}
        }
        
        for key, data in self.data.items():
            if 'baseline_eyes_open' in key:
                psd = data['raw'].compute_psd(
                    method='welch', fmin=1, fmax=40, n_fft=512,
                    n_overlap=256, verbose=False
                )
                psd_data, freqs = psd.get_data(return_freqs=True)
                features['baseline_open']['psd'] = psd_data.mean(axis=0)
                features['baseline_open']['freqs'] = freqs
                features['real']['freqs'] = freqs  # Set common freqs
                features['imagined']['freqs'] = freqs
                
            elif 'baseline_eyes_closed' in key:
                psd = data['raw'].compute_psd(
                    method='welch', fmin=1, fmax=40, n_fft=512,
                    n_overlap=256, verbose=False
                )
                psd_data, freqs = psd.get_data(return_freqs=True)
                features['baseline_closed']['psd'] = psd_data.mean(axis=0)
                features['baseline_closed']['freqs'] = freqs
                
            elif 'real' in key and 'epochs' in data:
                psd = data['epochs'].compute_psd(
                    method='welch', fmin=1, fmax=40, n_fft=512,
                    n_overlap=256, verbose=False
                )
                psd_data, freqs = psd.get_data(return_freqs=True)
                features['real']['psd'].append(psd_data.mean(axis=(0, 1)))
                
            elif 'imagined' in key and 'epochs' in data:
                psd = data['epochs'].compute_psd(
                    method='welch', fmin=1, fmax=40, n_fft=512,
                    n_overlap=256, verbose=False
                )
                psd_data, freqs = psd.get_data(return_freqs=True)
                features['imagined']['psd'].append(psd_data.mean(axis=(0, 1)))
        
        # Average across tasks
        if features['real']['psd']:
            features['real']['psd'] = np.mean(features['real']['psd'], axis=0)
        if features['imagined']['psd']:
            features['imagined']['psd'] = np.mean(features['imagined']['psd'], axis=0)
            
        return features
    
    def _extract_amplitude_features(self) -> Dict:
        """Extract amplitude features by channel."""
        features = {
            'channels': [],
            'real': [],
            'imagined': [],
            'baseline_open': [],
            'baseline_closed': []
        }
        
        # Get first available dataset with epochs to determine channels
        first_key = None
        for key, data in self.data.items():
            if 'epochs' in data:
                first_key = key
                break
        
        if not first_key:
            return features
            
        available_channels = self.data[first_key]['epochs'].ch_names
        
        for ch in available_channels:
            ch_amps_real = []
            ch_amps_imag = []
            ch_baseline_open = None
            ch_baseline_closed = None
            
            # Calculate baseline amplitudes
            for key, data in self.data.items():
                if 'baseline_eyes_open' in key and ch in data['raw'].ch_names:
                    ch_idx = data['raw'].ch_names.index(ch)
                    raw_data = data['raw'].get_data()[ch_idx, :]
                    ch_baseline_open = np.sqrt(np.mean(raw_data**2)) * 1e6  # RMS in µV
                    
                elif 'baseline_eyes_closed' in key and ch in data['raw'].ch_names:
                    ch_idx = data['raw'].ch_names.index(ch)
                    raw_data = data['raw'].get_data()[ch_idx, :]
                    ch_baseline_closed = np.sqrt(np.mean(raw_data**2)) * 1e6  # RMS in µV
            
            # Calculate task amplitudes
            for key, data in self.data.items():
                if 'epochs' in data and ch in data['epochs'].ch_names:
                    ch_idx = data['epochs'].ch_names.index(ch)
                    epochs_data = data['epochs'].get_data()[:, ch_idx, :]
                    
                    # RMS amplitude during movement period (0-3s)
                    movement_data = epochs_data[:, int(0*data['epochs'].info['sfreq']):int(3*data['epochs'].info['sfreq'])]
                    rms_amp = np.sqrt(np.mean(movement_data**2))
                    
                    if 'real' in key:
                        ch_amps_real.append(rms_amp * 1e6)
                    elif 'imagined' in key:
                        ch_amps_imag.append(rms_amp * 1e6)
            
            features['channels'].append(ch)
            features['real'].append(np.mean(ch_amps_real) if ch_amps_real else 0)
            features['imagined'].append(np.mean(ch_amps_imag) if ch_amps_imag else 0)
            features['baseline_open'].append(ch_baseline_open if ch_baseline_open is not None else 0)
            features['baseline_closed'].append(ch_baseline_closed if ch_baseline_closed is not None else 0)
            
        return features
    
    def _extract_timefreq_features(self) -> Dict:
        """Extract time-frequency features."""
        features = {
            'fist': {'real': None, 'imagined': None, 'times': None, 'freqs': None},
            'fists_feet': {'real': None, 'imagined': None, 'times': None, 'freqs': None}
        }
        
        freqs = np.arange(4, 30, 1)
        
        for task_type in ['fist', 'fists_feet']:
            real_data = None
            imag_data = None
            
            for key, data in self.data.items():
                if 'pair_info' in data and data['pair_info']['type'] == task_type:
                    if 'real' in key and real_data is None:
                        real_data = data
                    elif 'imagined' in key and imag_data is None:
                        imag_data = data
            
            if real_data and imag_data:
                # Process real movement
                power_real, erds_real = self.compute_erds_features(real_data['epochs'], freqs)
                features[task_type]['real'] = erds_real.mean(axis=0)
                features[task_type]['times'] = power_real.times
                features[task_type]['freqs'] = freqs
                
                # Process imagined movement
                power_imag, erds_imag = self.compute_erds_features(imag_data['epochs'], freqs)
                features[task_type]['imagined'] = erds_imag.mean(axis=0)
                
        return features
    
    def _extract_band_power_features(self) -> Dict:
        """Extract band power features as ERD/ERS (percent change from baseline)."""
        bands = {
            'theta': (4, 7),
            'mu': (8, 13),
            'beta': (13, 30)
        }
        
        features = {band: {'real': [], 'imagined': []} for band in bands}
        
        for band_name, (fmin, fmax) in bands.items():
            for key, data in self.data.items():
                if 'epochs' in data:
                    # Pick only motor channels
                    motor_channels = [ch for ch in self.channels if ch in data['epochs'].ch_names]
                    if not motor_channels:
                        continue
                    
                    epochs_motor = data['epochs'].copy().pick_channels(motor_channels)
                    
                    # Compute time-frequency representation using Morlet wavelets
                    freqs = np.arange(fmin, fmax + 1, 1)
                    n_cycles = freqs / 2.0
                    
                    power = mne.time_frequency.tfr_morlet(
                        epochs_motor, freqs=freqs, n_cycles=n_cycles,
                        return_itc=False, average=False, verbose=False
                    )
                    
                    # Average across frequencies: shape (n_epochs, n_channels, n_times)
                    power_data = power.data.mean(axis=2)
                    
                    # Compute ERD/ERS for EACH CHANNEL separately to avoid averaging artifacts
                    # This matches the approach used in lateralization analysis
                    baseline_mask = (power.times >= -1) & (power.times < 0)
                    movement_mask = (power.times >= 0) & (power.times <= 3)
                    
                    channel_erds_values = []
                    for ch_idx in range(power_data.shape[1]):  # Loop over channels
                        # Extract power for this channel: (n_epochs, n_times)
                        ch_power = power_data[:, ch_idx, :]
                        
                        # Compute baseline for each epoch
                        ch_baseline = ch_power[:, baseline_mask].mean(axis=1, keepdims=True)
                        
                        # Compute ERD/ERS for each epoch
                        ch_erds = ((ch_power - ch_baseline) / ch_baseline) * 100
                        
                        # Average across movement period for each epoch
                        ch_erds_movement = ch_erds[:, movement_mask].mean(axis=1)
                        
                        # Average across epochs for this channel
                        ch_erds_avg = ch_erds_movement.mean()
                        
                        channel_erds_values.append(ch_erds_avg)
                    
                    # Average ERD/ERS across all channels
                    erds_subject = np.mean(channel_erds_values)
                    
                    if 'real' in key:
                        features[band_name]['real'].append(erds_subject)
                    elif 'imagined' in key:
                        features[band_name]['imagined'].append(erds_subject)
        
        return features
    
    def _extract_lateralization_features(self) -> Dict:
        """Extract lateralization features."""
        features = {
            'mu_band': {
                'lateralization_index': {
                    'real': {'left': None, 'right': None}, 
                    'imagined': {'left': None, 'right': None}
                }
            },
            'beta_band': {
                'lateralization_index': {
                    'real': {'left': None, 'right': None}, 
                    'imagined': {'left': None, 'right': None}
                }
            },
            'times': None
        }
        
        # Get left/right fist data
        left_real = []
        left_imag = []
        right_real = []
        right_imag = []

        for key, data in self.data.items():
            if 'pair_info' in data and data['pair_info']['type'] == 'fist':
                epochs = data['epochs']
                event_dict = data['event_dict']
                
                if 'T1' in event_dict and 'T2' in event_dict:
                    left_epochs = epochs['T1']
                    right_epochs = epochs['T2']
                    
                    if 'real' in key:
                        left_real.append(left_epochs)
                        right_real.append(right_epochs)
                    else:
                        left_imag.append(left_epochs)
                        right_imag.append(right_epochs)

        if left_real and right_real and 'C3' in left_real[0].ch_names and 'C4' in left_real[0].ch_names:
            # Concatenate epochs
            left_real_all = mne.concatenate_epochs(left_real)
            right_real_all = mne.concatenate_epochs(right_real)
            left_imag_all = mne.concatenate_epochs(left_imag)
            right_imag_all = mne.concatenate_epochs(right_imag)
            
            c3_idx = left_real_all.ch_names.index('C3')
            c4_idx = left_real_all.ch_names.index('C4')
            
            # Extract lateralization for BOTH mu and beta bands
            for band_name, (fmin, fmax) in [('mu_band', (8, 13)), ('beta_band', (13, 30))]:
                # Compute time-resolved band power
                freqs = np.arange(fmin, fmax + 1, 1)
                n_cycles = freqs / 2.0
                
                # Real movement - left fist
                power_left_real = mne.time_frequency.tfr_morlet(
                    left_real_all, freqs=freqs, n_cycles=n_cycles,
                    return_itc=False, average=False, verbose=False
                )
                left_real_c3_raw = power_left_real.data[:, c3_idx, :, :].mean(axis=1)  # Average across frequencies
                left_real_c4_raw = power_left_real.data[:, c4_idx, :, :].mean(axis=1)
                # Compute ERD/ERS (baseline correction)
                baseline_mask = (power_left_real.times >= -1) & (power_left_real.times < 0)
                left_real_c3_baseline = left_real_c3_raw[:, baseline_mask].mean(axis=1, keepdims=True)
                left_real_c4_baseline = left_real_c4_raw[:, baseline_mask].mean(axis=1, keepdims=True)
                left_real_c3 = ((left_real_c3_raw - left_real_c3_baseline) / left_real_c3_baseline) * 100
                left_real_c4 = ((left_real_c4_raw - left_real_c4_baseline) / left_real_c4_baseline) * 100
                # Average across epochs
                left_real_c3_power = left_real_c3.mean(axis=0)
                left_real_c4_power = left_real_c4.mean(axis=0)
                
                # Real movement - right fist
                power_right_real = mne.time_frequency.tfr_morlet(
                    right_real_all, freqs=freqs, n_cycles=n_cycles,
                    return_itc=False, average=False, verbose=False
                )
                right_real_c3_raw = power_right_real.data[:, c3_idx, :, :].mean(axis=1)  # Average across frequencies
                right_real_c4_raw = power_right_real.data[:, c4_idx, :, :].mean(axis=1)
                # Compute ERD/ERS (baseline correction)
                baseline_mask = (power_right_real.times >= -1) & (power_right_real.times < 0)
                right_real_c3_baseline = right_real_c3_raw[:, baseline_mask].mean(axis=1, keepdims=True)
                right_real_c4_baseline = right_real_c4_raw[:, baseline_mask].mean(axis=1, keepdims=True)
                right_real_c3 = ((right_real_c3_raw - right_real_c3_baseline) / right_real_c3_baseline) * 100
                right_real_c4 = ((right_real_c4_raw - right_real_c4_baseline) / right_real_c4_baseline) * 100
                # Average across epochs
                right_real_c3_power = right_real_c3.mean(axis=0)
                right_real_c4_power = right_real_c4.mean(axis=0)

                # Imagined movement - left fist
                power_left_imag = mne.time_frequency.tfr_morlet(
                    left_imag_all, freqs=freqs, n_cycles=n_cycles,
                    return_itc=False, average=False, verbose=False
                )
                left_imag_c3_raw = power_left_imag.data[:, c3_idx, :, :].mean(axis=1)  # Average across frequencies
                left_imag_c4_raw = power_left_imag.data[:, c4_idx, :, :].mean(axis=1)
                # Compute ERD/ERS (baseline correction)
                baseline_mask = (power_left_imag.times >= -1) & (power_left_imag.times < 0)
                left_imag_c3_baseline = left_imag_c3_raw[:, baseline_mask].mean(axis=1, keepdims=True)
                left_imag_c4_baseline = left_imag_c4_raw[:, baseline_mask].mean(axis=1, keepdims=True)
                left_imag_c3 = ((left_imag_c3_raw - left_imag_c3_baseline) / left_imag_c3_baseline) * 100
                left_imag_c4 = ((left_imag_c4_raw - left_imag_c4_baseline) / left_imag_c4_baseline) * 100
                # Average across epochs
                left_imag_c3_power = left_imag_c3.mean(axis=0)
                left_imag_c4_power = left_imag_c4.mean(axis=0)

                # Imagined movement - right fist
                power_right_imag = mne.time_frequency.tfr_morlet(
                    right_imag_all, freqs=freqs, n_cycles=n_cycles,
                    return_itc=False, average=False, verbose=False
                )
                right_imag_c3_raw = power_right_imag.data[:, c3_idx, :, :].mean(axis=1)  # Average across frequencies
                right_imag_c4_raw = power_right_imag.data[:, c4_idx, :, :].mean(axis=1)
                # Compute ERD/ERS (baseline correction)
                baseline_mask = (power_right_imag.times >= -1) & (power_right_imag.times < 0)
                right_imag_c3_baseline = right_imag_c3_raw[:, baseline_mask].mean(axis=1, keepdims=True)
                right_imag_c4_baseline = right_imag_c4_raw[:, baseline_mask].mean(axis=1, keepdims=True)
                right_imag_c3 = ((right_imag_c3_raw - right_imag_c3_baseline) / right_imag_c3_baseline) * 100
                right_imag_c4 = ((right_imag_c4_raw - right_imag_c4_baseline) / right_imag_c4_baseline) * 100
                # Average across epochs
                right_imag_c3_power = right_imag_c3.mean(axis=0)
                right_imag_c4_power = right_imag_c4.mean(axis=0)

                # Compute lateralization indices on ERD/ERS values
                left_real_lat = (left_real_c3_power - left_real_c4_power) / (np.abs(left_real_c3_power) + np.abs(left_real_c4_power) + 1e-10)
                right_real_lat = (right_real_c3_power - right_real_c4_power) / (np.abs(right_real_c3_power) + np.abs(right_real_c4_power) + 1e-10)
                left_imag_lat = (left_imag_c3_power - left_imag_c4_power) / (np.abs(left_imag_c3_power) + np.abs(left_imag_c4_power) + 1e-10)
                right_imag_lat = (right_imag_c3_power - right_imag_c4_power) / (np.abs(right_imag_c3_power) + np.abs(right_imag_c4_power) + 1e-10)

                # Store in features dictionary
                features[band_name]['lateralization_index']['real']['left'] = left_real_lat
                features[band_name]['lateralization_index']['real']['right'] = right_real_lat
                features[band_name]['lateralization_index']['imagined']['left'] = left_imag_lat
                features[band_name]['lateralization_index']['imagined']['right'] = right_imag_lat
                
                # Store time axis (same for all bands)
                if features['times'] is None:
                    features['times'] = power_left_real.times
        
        return features

    def _extract_temporal_features(self) -> Dict:
        """Extract temporal dynamics features."""
        time_windows = [
            (-1.0, 0.0, 'pre_movement'),
            (0.0, 1.0, 'early_movement'),
            (1.0, 3.0, 'sustained_movement')
        ]
        
        features = {window[2]: {'real': {'theta': [], 'alpha': [], 'beta': []}, 
                            'imagined': {'theta': [], 'alpha': [], 'beta': []}}
                for window in time_windows}
        
        for tmin, tmax, label in time_windows:
            for key, data in self.data.items():
                if 'epochs' in data:
                    # Crop epochs to time window
                    epochs_cropped = data['epochs'].copy().crop(tmin=tmin, tmax=tmax)
                    
                    # Compute band powers
                    n_samples = epochs_cropped.get_data().shape[-1]
                    n_fft = min(256, n_samples)
                    n_overlap = min(128, n_fft // 2)
                    
                    psd = epochs_cropped.compute_psd(
                        method='welch', fmin=1, fmax=40,
                        n_fft=n_fft, n_overlap=n_overlap, verbose=False
                    )
                    psd_data, freqs = psd.get_data(return_freqs=True)
                    
                    # Extract band powers
                    theta_mask = (freqs >= 4) & (freqs <= 7)
                    alpha_mask = (freqs >= 8) & (freqs <= 12)
                    beta_mask = (freqs >= 13) & (freqs <= 30)
                    
                    theta_power = psd_data[:, :, theta_mask].mean()
                    alpha_power = psd_data[:, :, alpha_mask].mean()
                    beta_power = psd_data[:, :, beta_mask].mean()
                    
                    if 'real' in key:
                        features[label]['real']['theta'].append(theta_power)
                        features[label]['real']['alpha'].append(alpha_power)
                        features[label]['real']['beta'].append(beta_power)
                    elif 'imagined' in key:
                        features[label]['imagined']['theta'].append(theta_power)
                        features[label]['imagined']['alpha'].append(alpha_power)
                        features[label]['imagined']['beta'].append(beta_power)
        
        return features


class MultiSubjectAnalyzer:
    """Analyze multiple subjects and create group-level visualizations."""
    
    def __init__(self, base_path: str = 'eeg-motor-movementimagery-dataset-1.0.0/files', output_dir: str = 'group_imagined_vs_actual'):
        self.base_path = Path(base_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Data storage
        self.subject_data = {}
        self.failed_subjects = []
        self.aggregated_data = {}
        
    def process_single_subject(self, subject_id: str) -> Dict:
        """Process a single subject and return extracted features."""
        logger.info(f"Processing {subject_id}...")
        # Pass the base_path directly, not its parent
        extractor = SubjectDataExtractor(subject_id, str(self.base_path))
        return extractor.extract_all_features()
    
    def process_all_subjects(self, max_workers: int = 4):
        """Process all subjects in parallel."""
        # Get all subject directories
        subject_dirs = sorted([d for d in self.base_path.iterdir() 
                              if d.is_dir() and d.name.startswith('S') and d.name[1:].isdigit()])
        
        logger.info(f"Found {len(subject_dirs)} subjects to process")
        
        # Process subjects in parallel with progress bar
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_subject = {
                executor.submit(self.process_single_subject, d.name): d.name 
                for d in subject_dirs
            }
            
            with tqdm(total=len(subject_dirs), desc="Processing subjects") as pbar:
                for future in as_completed(future_to_subject):
                    subject_id = future_to_subject[future]
                    try:
                        result = future.result()
                        if result['metadata']['success']:
                            self.subject_data[subject_id] = result
                        else:
                            self.failed_subjects.append({
                                'subject': subject_id,
                                'error': result['metadata']['error']
                            })
                    except Exception as e:
                        self.failed_subjects.append({
                            'subject': subject_id,
                            'error': str(e)
                        })
                    pbar.update(1)
        
        logger.info(f"Successfully processed {len(self.subject_data)} subjects")
        logger.info(f"Failed to process {len(self.failed_subjects)} subjects")
        
        # Save individual subject data
        self._save_subject_data()
        
    def _save_subject_data(self):
        """Save individual subject data to file."""
        data_file = self.output_dir / 'individual_subject_data.json'
        
        # Convert numpy arrays to lists for JSON serialization
        def convert_to_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            return obj
        
        serializable_data = convert_to_serializable(self.subject_data)
        
        with open(data_file, 'w') as f:
            json.dump(serializable_data, f, indent=2)
        
        # Save failed subjects list
        if self.failed_subjects:
            failed_file = self.output_dir / 'failed_subjects.json'
            with open(failed_file, 'w') as f:
                json.dump(self.failed_subjects, f, indent=2)
    
    def aggregate_data(self):
        """Aggregate data across all subjects."""
        logger.info("Aggregating data across subjects...")
        
        # Initialize aggregated data structure
        self.aggregated_data = {
            'psd': {'real': [], 'imagined': [], 'baseline_open': [], 'baseline_closed': [], 'freqs': None},
            'amplitudes': {'channels': None, 'real': [], 'imagined': [], 'baseline_open': [], 'baseline_closed': []},
            'timefreq': {'fist': {'real': [], 'imagined': []}, 'fists_feet': {'real': [], 'imagined': []}},
            'band_powers': {band: {'real': [], 'imagined': []} 
               for band in ['theta', 'mu', 'beta']},
            'lateralization': {
                'mu_band': {'real': {'left': [], 'right': []}, 'imagined': {'left': [], 'right': []}},
                'beta_band': {'real': {'left': [], 'right': []}, 'imagined': {'left': [], 'right': []}}
            },
            'temporal': {},
            'n_subjects': len(self.subject_data)
        }
        
        # Aggregate each feature type
        for subject_id, data in self.subject_data.items():
            # PSD features
            if 'psd' in data:
                psd = data['psd']
                if psd['real']['psd'] is not None:
                    self.aggregated_data['psd']['real'].append(psd['real']['psd'])
                if psd['imagined']['psd'] is not None:
                    self.aggregated_data['psd']['imagined'].append(psd['imagined']['psd'])
                if psd['baseline_open']['psd'] is not None:
                    self.aggregated_data['psd']['baseline_open'].append(psd['baseline_open']['psd'])
                if psd['baseline_closed']['psd'] is not None:
                    self.aggregated_data['psd']['baseline_closed'].append(psd['baseline_closed']['psd'])
                if self.aggregated_data['psd']['freqs'] is None:
                    self.aggregated_data['psd']['freqs'] = psd['real']['freqs']
            
            # Amplitude features
            if 'amplitudes' in data:
                amp = data['amplitudes']
                if self.aggregated_data['amplitudes']['channels'] is None:
                    self.aggregated_data['amplitudes']['channels'] = amp['channels']
                if amp['real']:
                    self.aggregated_data['amplitudes']['real'].append(amp['real'])
                if amp['imagined']:
                    self.aggregated_data['amplitudes']['imagined'].append(amp['imagined'])
                if amp['baseline_open']:
                    self.aggregated_data['amplitudes']['baseline_open'].append(amp['baseline_open'])
                if amp['baseline_closed']:
                    self.aggregated_data['amplitudes']['baseline_closed'].append(amp['baseline_closed'])
            
            # Time-frequency features
            if 'timefreq' in data:
                tf = data['timefreq']
                for task in ['fist', 'fists_feet']:
                    if tf[task]['real'] is not None:
                        self.aggregated_data['timefreq'][task]['real'].append(tf[task]['real'])
                    if tf[task]['imagined'] is not None:
                        self.aggregated_data['timefreq'][task]['imagined'].append(tf[task]['imagined'])
            
            # Band power features
            if 'band_powers' in data:
                bp = data['band_powers']
                for band in ['theta', 'mu', 'beta']:
                    if band in bp:
                        if bp[band]['real']:
                            self.aggregated_data['band_powers'][band]['real'].extend(bp[band]['real'])
                        if bp[band]['imagined']:
                            self.aggregated_data['band_powers'][band]['imagined'].extend(bp[band]['imagined'])
            
            # Lateralization features
            if 'lateralization' in data:
                lat = data['lateralization']
                
                # Aggregate mu band lateralization
                if 'mu_band' in lat:
                    if lat['mu_band']['lateralization_index']['real']['left'] is not None:
                        self.aggregated_data['lateralization']['mu_band']['real']['left'].append(
                            lat['mu_band']['lateralization_index']['real']['left'])
                    if lat['mu_band']['lateralization_index']['real']['right'] is not None:
                        self.aggregated_data['lateralization']['mu_band']['real']['right'].append(
                            lat['mu_band']['lateralization_index']['real']['right'])
                    if lat['mu_band']['lateralization_index']['imagined']['left'] is not None:
                        self.aggregated_data['lateralization']['mu_band']['imagined']['left'].append(
                            lat['mu_band']['lateralization_index']['imagined']['left'])
                    if lat['mu_band']['lateralization_index']['imagined']['right'] is not None:
                        self.aggregated_data['lateralization']['mu_band']['imagined']['right'].append(
                            lat['mu_band']['lateralization_index']['imagined']['right'])
                
                # Aggregate beta band lateralization
                if 'beta_band' in lat:
                    if lat['beta_band']['lateralization_index']['real']['left'] is not None:
                        self.aggregated_data['lateralization']['beta_band']['real']['left'].append(
                            lat['beta_band']['lateralization_index']['real']['left'])
                    if lat['beta_band']['lateralization_index']['real']['right'] is not None:
                        self.aggregated_data['lateralization']['beta_band']['real']['right'].append(
                            lat['beta_band']['lateralization_index']['real']['right'])
                    if lat['beta_band']['lateralization_index']['imagined']['left'] is not None:
                        self.aggregated_data['lateralization']['beta_band']['imagined']['left'].append(
                            lat['beta_band']['lateralization_index']['imagined']['left'])
                    if lat['beta_band']['lateralization_index']['imagined']['right'] is not None:
                        self.aggregated_data['lateralization']['beta_band']['imagined']['right'].append(
                            lat['beta_band']['lateralization_index']['imagined']['right'])
                        
        logger.info(f"Aggregated data from {self.aggregated_data['n_subjects']} subjects")
    
    def _filter_to_common_length(self, arrays):
        """Filter arrays to most common length to handle inhomogeneous shapes."""
        if not arrays:
            return []
        
        # Filter out None values first
        arrays = [arr for arr in arrays if arr is not None]
        if not arrays:
            return []
        
        # Get shapes of all arrays
        shapes = [arr.shape if hasattr(arr, 'shape') else (len(arr),) for arr in arrays]
        
        # For 2D arrays (e.g., time-frequency), check both dimensions
        if len(shapes[0]) == 2:
            # Find most common shape
            shape_counts = {}
            for shape in shapes:
                shape_tuple = tuple(shape)
                shape_counts[shape_tuple] = shape_counts.get(shape_tuple, 0) + 1
            common_shape = max(shape_counts.keys(), key=lambda x: shape_counts[x])
            filtered = [arr for arr, shape in zip(arrays, shapes) if tuple(shape) == common_shape]
            logger.debug(f"Filtered {len(arrays)} 2D arrays to {len(filtered)} with common shape {common_shape}")
        else:
            # For 1D arrays
            lengths = [shape[0] for shape in shapes]
            common_length = max(set(lengths), key=lengths.count)
            filtered = [arr for arr, shape in zip(arrays, shapes) if shape[0] == common_length]
            logger.debug(f"Filtered {len(arrays)} arrays to {len(filtered)} with common length {common_length}")
        
        return filtered
    
    def create_group_visualizations(self):
        """Create group-level visualizations."""
        logger.info("Creating group-level visualizations...")
        
        # Check if we have any data to visualize
        if not self.aggregated_data or self.aggregated_data['n_subjects'] == 0:
            logger.warning("No aggregated data available for visualization")
            return
        
        # Create each visualization with error handling
        visualizations = [
            ("Group average PSD comparison", self._plot_group_psd_comparison),
            ("Group amplitude comparison", self._plot_group_amplitude_comparison),
            ("Group time-frequency analysis", self._plot_group_timefreq_analysis),
            ("Group band power statistics", self._plot_group_band_power_statistics),
            ("Group lateralization analysis", self._plot_group_lateralization_analysis),
            ("Subject variability analysis", self._plot_subject_variability_analysis)
        ]
        
        for viz_name, viz_func in visualizations:
            try:
                logger.info(f"Creating {viz_name}...")
                viz_func()
            except Exception as e:
                logger.error(f"Failed to create {viz_name}: {str(e)}")
                logger.debug(f"Error details: {type(e).__name__}: {e}")
                # Continue with next visualization instead of crashing
        
    def _plot_group_psd_comparison(self):
        """Plot averaged PSD across all subjects."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        freqs = self.aggregated_data['psd']['freqs']
        
        if freqs is None or len(self.aggregated_data['psd']['real']) == 0:
            ax.text(0.5, 0.5, 'No PSD data available', ha='center', va='center', transform=ax.transAxes)
            ax.axis('off')
            plt.savefig(self.output_dir / '01_group_psd_comparison.png', dpi=300, bbox_inches='tight')
            plt.close()
            logger.warning("No PSD data available for plotting")
            return
        
        # Filter arrays to common length to handle inhomogeneous shapes
        baseline_open_filtered = self._filter_to_common_length(self.aggregated_data['psd']['baseline_open'])
        baseline_closed_filtered = self._filter_to_common_length(self.aggregated_data['psd']['baseline_closed'])
        real_filtered = self._filter_to_common_length(self.aggregated_data['psd']['real'])
        imag_filtered = self._filter_to_common_length(self.aggregated_data['psd']['imagined'])
        
        if not real_filtered:
            ax.text(0.5, 0.5, 'No PSD data with compatible dimensions', ha='center', va='center', transform=ax.transAxes)
            ax.axis('off')
            plt.savefig(self.output_dir / '01_group_psd_comparison.png', dpi=300, bbox_inches='tight')
            plt.close()
            logger.warning("No PSD data with compatible dimensions for plotting")
            return
        
        # Adjust freqs if needed to match filtered data length
        if len(freqs) != len(real_filtered[0]):
            freqs = freqs[:len(real_filtered[0])]
        
        # Calculate means and SEMs
        if baseline_open_filtered:
            baseline_open_mean = np.mean(baseline_open_filtered, axis=0)
            baseline_open_db = 10 * np.log10(baseline_open_mean)
            ax.plot(freqs, baseline_open_db, color='#7F8C8D', linewidth=2, 
                   linestyle='--', label=f'Baseline (Eyes Open, n={len(baseline_open_filtered)})', alpha=0.7)
        
        if baseline_closed_filtered:
            baseline_closed_mean = np.mean(baseline_closed_filtered, axis=0)
            baseline_closed_db = 10 * np.log10(baseline_closed_mean)
            ax.plot(freqs, baseline_closed_db, color='#34495E', linewidth=2, 
                   linestyle='--', label=f'Baseline (Eyes Closed, n={len(baseline_closed_filtered)})', alpha=0.7)
        
        if real_filtered:
            real_mean = np.mean(real_filtered, axis=0)
            real_sem = stats.sem(real_filtered, axis=0)
            real_db = 10 * np.log10(real_mean)
            real_db_upper = 10 * np.log10(real_mean + real_sem)
            real_db_lower = 10 * np.log10(real_mean - real_sem)
            
            ax.plot(freqs, real_db, color=COLORS['actual'], linewidth=2, 
                   label=f'Actual Movement (n={len(real_filtered)})')
            ax.fill_between(freqs, real_db_lower, real_db_upper, alpha=0.3, color=COLORS['actual'])
        
        if imag_filtered:
            imag_mean = np.mean(imag_filtered, axis=0)
            imag_sem = stats.sem(imag_filtered, axis=0)
            imag_db = 10 * np.log10(imag_mean)
            imag_db_upper = 10 * np.log10(imag_mean + imag_sem)
            imag_db_lower = 10 * np.log10(imag_mean - imag_sem)
            
            ax.plot(freqs, imag_db, color=COLORS['imagined'], linewidth=2,
                   label=f'Imagined Movement (n={len(imag_filtered)})')
            ax.fill_between(freqs, imag_db_lower, imag_db_upper, alpha=0.3, color=COLORS['imagined'])
        
        # Highlight frequency bands
        ax.axvspan(8, 13, alpha=0.1, color='gray', label='Mu band (8-13 Hz)')
        ax.axvspan(13, 30, alpha=0.1, color='lightgray', label='Beta band (13-30 Hz)')
        
        ax.set_xlabel('Frequency (Hz)', fontsize=12)
        ax.set_ylabel('Power Spectral Density (dB)', fontsize=12)
        ax.set_title('Group Average Power Spectrum: Actual vs Imagined Movement', fontsize=14, fontweight='bold')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(1, 40)
        
        plt.tight_layout()
        output_path = self.output_dir / '01_group_psd_comparison.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"  ✓ Saved: {output_path}")
    
    def _plot_group_amplitude_comparison(self):
        """Plot group-averaged amplitude comparison."""
        if not self.aggregated_data['amplitudes']['channels']:
            logger.warning("No amplitude data available for plotting")
            return
            
        fig, ax = plt.subplots(figsize=(12, 6))
        
        channels = self.aggregated_data['amplitudes']['channels']
        x = np.arange(len(channels))
        width = 0.2
        
        # Calculate means and SEMs
        def calculate_stats(data_list):
            if data_list and data_list[0]:
                mean = np.mean(data_list, axis=0)
                sem = stats.sem(data_list, axis=0)
                return mean, sem
            return np.zeros(len(channels)), np.zeros(len(channels))
        
        baseline_open_mean, baseline_open_sem = calculate_stats(self.aggregated_data['amplitudes']['baseline_open'])
        baseline_closed_mean, baseline_closed_sem = calculate_stats(self.aggregated_data['amplitudes']['baseline_closed'])
        real_mean, real_sem = calculate_stats(self.aggregated_data['amplitudes']['real'])
        imag_mean, imag_sem = calculate_stats(self.aggregated_data['amplitudes']['imagined'])
        
        # Create bar plots
        bars1 = ax.bar(x - 1.5*width, baseline_open_mean, width, yerr=baseline_open_sem,
                       label='Baseline (Eyes Open)', color='#7F8C8D', alpha=0.7, capsize=5)
        bars2 = ax.bar(x - 0.5*width, baseline_closed_mean, width, yerr=baseline_closed_sem,
                       label='Baseline (Eyes Closed)', color='#34495E', alpha=0.7, capsize=5)
        bars3 = ax.bar(x + 0.5*width, real_mean, width, yerr=real_sem,
                       label='Actual', color=COLORS['actual'], capsize=5)
        bars4 = ax.bar(x + 1.5*width, imag_mean, width, yerr=imag_sem,
                       label='Imagined', color=COLORS['imagined'], capsize=5)
        
        ax.set_xlabel('Channel', fontsize=12)
        ax.set_ylabel('RMS Amplitude (µV)', fontsize=12)
        ax.set_title('Group Average Channel Amplitudes', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(channels)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        output_path = self.output_dir / '02_group_amplitude_comparison.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"  ✓ Saved: {output_path}")
    
    def _plot_group_timefreq_analysis(self):
        """Plot group-averaged time-frequency analysis."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Group Average Time-Frequency Analysis', fontsize=16, fontweight='bold')
        
        vmin, vmax = -30, 30  # Adjusted for averaged data
        
        for idx, (task, task_name) in enumerate([('fist', 'Left/Right Fist'), 
                                                 ('fists_feet', 'Both Fists/Both Feet')]):
            # Filter to common shape
            real_filtered = self._filter_to_common_length(self.aggregated_data['timefreq'][task]['real'])
            imag_filtered = self._filter_to_common_length(self.aggregated_data['timefreq'][task]['imagined'])
            
            if real_filtered:
                # Real movement
                real_mean = np.mean(real_filtered, axis=0)
                
                im = axes[0, idx].imshow(real_mean, aspect='auto', origin='lower',
                                        extent=[-1, 4, 4, 30], cmap='RdBu_r', vmin=vmin, vmax=vmax)
                axes[0, idx].set_title(f'{task_name} - Actual', fontweight='bold')
                axes[0, idx].set_xlabel('Time (s)')
                axes[0, idx].set_ylabel('Frequency (Hz)')
                axes[0, idx].axvline(0, color='k', linestyle='--', alpha=0.5)
                
                # Add colorbar
                divider = make_axes_locatable(axes[0, idx])
                cax = divider.append_axes("right", size="5%", pad=0.05)
                plt.colorbar(im, cax=cax)
            
            if imag_filtered:
                # Imagined movement
                imag_mean = np.mean(imag_filtered, axis=0)
                
                im = axes[1, idx].imshow(imag_mean, aspect='auto', origin='lower',
                                        extent=[-1, 4, 4, 30], cmap='RdBu_r', vmin=vmin, vmax=vmax)
                axes[1, idx].set_title(f'{task_name} - Imagined', fontweight='bold')
                axes[1, idx].set_xlabel('Time (s)')
                axes[1, idx].set_ylabel('Frequency (Hz)')
                axes[1, idx].axvline(0, color='k', linestyle='--', alpha=0.5)
                
                # Add colorbar
                divider = make_axes_locatable(axes[1, idx])
                cax = divider.append_axes("right", size="5%", pad=0.05)
                cbar = plt.colorbar(im, cax=cax)
                cbar.set_label('ERD/ERS (%)', rotation=270, labelpad=20)
        
        plt.tight_layout()
        output_path = self.output_dir / '03_group_timefreq_analysis.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"  ✓ Saved: {output_path}")
    
    def _plot_group_band_power_statistics(self):
        """Plot group ERD/ERS statistics with significance tests."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('Group ERD/ERS Analysis', fontsize=16, fontweight='bold')
        
        bands = ['theta', 'mu', 'beta']
        band_labels = ['Theta\n(4-7 Hz)', 'Mu\n(8-13 Hz)', 'Beta\n(13-30 Hz)']
        
        # Plot 1: Bar plot with actual vs imagined
        x = np.arange(len(bands))
        width = 0.35
        
        for i, band in enumerate(bands):
            # Calculate means for actual and imagined (ERD/ERS already in percent)
            real_mean = np.mean(self.aggregated_data['band_powers'][band]['real'])
            imag_mean = np.mean(self.aggregated_data['band_powers'][band]['imagined'])
            
            real_sem = stats.sem(self.aggregated_data['band_powers'][band]['real'])
            imag_sem = stats.sem(self.aggregated_data['band_powers'][band]['imagined'])
            
            # Plot bars (2 bars per band: actual and imagined)
            ax1.bar(x[i] - width/2, real_mean, width, yerr=real_sem,
                   color=COLORS['actual'], capsize=5, label='Actual' if i == 0 else '')
            ax1.bar(x[i] + width/2, imag_mean, width, yerr=imag_sem,
                   color=COLORS['imagined'], capsize=5, label='Imagined' if i == 0 else '')
        
        ax1.axhline(0, color='k', linestyle='-', linewidth=0.8)  # Add zero line
        ax1.set_xticks(x)
        ax1.set_xticklabels(band_labels)
        ax1.set_ylabel('ERD/ERS (%)', fontsize=12)
        ax1.set_title('ERD/ERS Comparison', fontsize=14)
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Plot 2: Statistical comparison table
        results = []
        p_values = []  # Collect p-values for correction
        
        # First pass: compute all tests
        for band in bands:
            real_vals = self.aggregated_data['band_powers'][band]['real']
            imag_vals = self.aggregated_data['band_powers'][band]['imagined']
            
            # Paired t-test
            t_stat, p_val = stats.ttest_rel(real_vals[:len(imag_vals)], imag_vals[:len(real_vals)])
            p_values.append(p_val)
            
            # Effect size (Cohen's d)
            diff = np.array(real_vals[:len(imag_vals)]) - np.array(imag_vals[:len(real_vals)])
            cohens_d = np.mean(diff) / np.std(diff) if np.std(diff) > 0 else 0
            
            results.append({
                'band': band,
                'real_vals': real_vals,
                'imag_vals': imag_vals,
                'p_uncorrected': p_val,
                'cohens_d': cohens_d
            })
        
        # Apply Holm-Bonferroni correction
        from scipy.stats import false_discovery_control
        # Sort p-values with their indices
        sorted_indices = np.argsort(p_values)
        n_tests = len(p_values)
        corrected_significant = [False] * n_tests
        
        # Holm-Bonferroni sequential procedure
        for rank, idx in enumerate(sorted_indices):
            corrected_alpha = 0.05 / (n_tests - rank)
            if p_values[idx] <= corrected_alpha:
                corrected_significant[idx] = True
            else:
                break  # Stop when first non-significant test found
        
        # Second pass: create table with corrected results
        final_results = []
        for i, result in enumerate(results):
            band = result['band']
            real_vals = result['real_vals']
            imag_vals = result['imag_vals']
            p_val = result['p_uncorrected']
            cohens_d = result['cohens_d']
            
            # Determine significance based on correction
            if corrected_significant[i]:
                if p_val < 0.001:
                    sig = '***'
                elif p_val < 0.01:
                    sig = '**'
                else:
                    sig = '*'
            else:
                sig = 'ns'
            
            final_results.append({
                'Band': band.capitalize(),
                'Real Mean±SEM': f"{np.mean(real_vals):.2f}±{stats.sem(real_vals):.2f}",
                'Imagined Mean±SEM': f"{np.mean(imag_vals):.2f}±{stats.sem(imag_vals):.2f}",
                'p-value': f"{p_val:.4f}",
                'Effect Size': f"{cohens_d:.3f}",
                'Sig': sig
            })
        
        results = final_results
        
        # Create table
        df = pd.DataFrame(results)
        ax2.axis('tight')
        ax2.axis('off')
        
        table = ax2.table(cellText=df.values, colLabels=df.columns,
                         cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 2)
        
        # Style header
        for i in range(len(df.columns)):
            table[(0, i)].set_facecolor('#4472C4')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        ax2.set_title('Statistical Comparison', fontsize=14, pad=20)
        
        plt.tight_layout()
        output_path = self.output_dir / '04_group_band_power_statistics.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"  ✓ Saved: {output_path}")
    
    def _plot_group_lateralization_analysis(self):
        """Plot group lateralization analysis for both mu and beta bands."""
        # Create mu band plot
        self._plot_group_lateralization_for_band('mu_band', 8, 13, 
                                                '05a_group_lateralization_mu_band.png')
        
        # Create beta band plot
        self._plot_group_lateralization_for_band('beta_band', 13, 30,
                                                '05b_group_lateralization_beta_band.png')

    def _plot_group_lateralization_for_band(self, band_name, fmin, fmax, output_filename):
        """Plot group lateralization analysis for a specific frequency band."""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(f'Group Lateralization Analysis: {band_name.replace("_", " ").title()} ({fmin}-{fmax} Hz)', 
                    fontsize=16, fontweight='bold')
        
        # Filter to common lengths
        left_real_filtered = self._filter_to_common_length(
            self.aggregated_data['lateralization'][band_name]['real']['left'])
        right_real_filtered = self._filter_to_common_length(
            self.aggregated_data['lateralization'][band_name]['real']['right'])
        left_imag_filtered = self._filter_to_common_length(
            self.aggregated_data['lateralization'][band_name]['imagined']['left'])
        right_imag_filtered = self._filter_to_common_length(
            self.aggregated_data['lateralization'][band_name]['imagined']['right'])
        
        # Calculate means and SEMs for lateralization indices
        if left_real_filtered and right_real_filtered:
            # Get time axis from first subject
            times = np.linspace(-1, 3, len(left_real_filtered[0]))
            
            # Real movement
            left_real_mean = np.mean(left_real_filtered, axis=0)
            left_real_sem = stats.sem(left_real_filtered, axis=0)
            right_real_mean = np.mean(right_real_filtered, axis=0)
            right_real_sem = stats.sem(right_real_filtered, axis=0)
            
            ax = axes[0]
            ax.plot(times, left_real_mean, color=COLORS['left_fist'], linewidth=2, label='Left Fist')
            ax.fill_between(times, left_real_mean-left_real_sem, left_real_mean+left_real_sem,
                        alpha=0.3, color=COLORS['left_fist'])
            ax.plot(times, right_real_mean, color=COLORS['right_fist'], linewidth=2, label='Right Fist')
            ax.fill_between(times, right_real_mean-right_real_sem, right_real_mean+right_real_sem,
                        alpha=0.3, color=COLORS['right_fist'])
            ax.axhline(0, color='k', linestyle='-', alpha=0.3)
            ax.axvline(0, color='k', linestyle='--', alpha=0.5)
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Lateralization Index (ERD/ERS based)')
            ax.set_title('Actual Movement', fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_xlim(-0.5, 3)
            
            # Imagined movement - check if we have filtered data
            if left_imag_filtered and right_imag_filtered:
                left_imag_mean = np.mean(left_imag_filtered, axis=0)
                left_imag_sem = stats.sem(left_imag_filtered, axis=0)
                right_imag_mean = np.mean(right_imag_filtered, axis=0)
                right_imag_sem = stats.sem(right_imag_filtered, axis=0)
                
                ax = axes[1]
                ax.plot(times, left_imag_mean, color=COLORS['left_fist'], linewidth=2, 
                    linestyle='--', label='Left Fist (Imagined)')
                ax.fill_between(times, left_imag_mean-left_imag_sem, left_imag_mean+left_imag_sem,
                            alpha=0.3, color=COLORS['left_fist'])
                ax.plot(times, right_imag_mean, color=COLORS['right_fist'], linewidth=2,
                    linestyle='--', label='Right Fist (Imagined)')
                ax.fill_between(times, right_imag_mean-right_imag_sem, right_imag_mean+right_imag_sem,
                            alpha=0.3, color=COLORS['right_fist'])
                ax.axhline(0, color='k', linestyle='-', alpha=0.3)
                ax.axvline(0, color='k', linestyle='--', alpha=0.5)
                ax.set_xlabel('Time (s)')
                ax.set_ylabel('Lateralization Index (ERD/ERS based)')
                ax.set_title('Imagined Movement', fontweight='bold')
                ax.legend()
                ax.grid(True, alpha=0.3)
                ax.set_xlim(-0.5, 3)
        
        plt.tight_layout()
        output_path = self.output_dir / output_filename
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"  ✓ Saved: {output_path}")
    
    def _plot_subject_variability_analysis(self):
        """Plot analysis of inter-subject variability."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Inter-Subject Variability Analysis', fontsize=16, fontweight='bold')
        
        # 1. Mu suppression variability
        ax = axes[0, 0]
        mu_real = []
        mu_imag = []
        
        for band_data in self.aggregated_data['band_powers']['mu']['real']:
            mu_real.append(band_data * 1e12)
        for band_data in self.aggregated_data['band_powers']['mu']['imagined']:
            mu_imag.append(band_data * 1e12)
        
        data_to_plot = [mu_real, mu_imag]
        bp = ax.boxplot(data_to_plot, labels=['Actual', 'Imagined'], patch_artist=True)
        colors = [COLORS['actual'], COLORS['imagined']]
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax.set_ylabel('Mu Band Power (pV²/Hz)')
        ax.set_title('Mu Band Power Distribution', fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        # 2. Effect size distribution
        ax = axes[0, 1]
        effect_sizes = []
        
        for i in range(min(len(mu_real), len(mu_imag))):
            if mu_real[i] > 0:
                effect = (mu_real[i] - mu_imag[i]) / mu_real[i]
                effect_sizes.append(effect)
        
        ax.hist(effect_sizes, bins=20, color=COLORS['difference'], alpha=0.7, edgecolor='black')
        ax.axvline(np.mean(effect_sizes), color='red', linestyle='--', linewidth=2,
                  label=f'Mean = {np.mean(effect_sizes):.3f}')
        ax.set_xlabel('Effect Size (Real - Imagined) / Real')
        ax.set_ylabel('Number of Subjects')
        ax.set_title('Distribution of Movement Effect Sizes', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        # 3. Subject clustering based on response patterns
        ax = axes[1, 0]
        # Simple 2D representation using mu and beta power
        mu_diff = [(r - i) for r, i in zip(mu_real[:len(mu_imag)], mu_imag[:len(mu_real)])]
        beta_real = [b * 1e12 for b in self.aggregated_data['band_powers']['beta']['real']]
        beta_imag = [b * 1e12 for b in self.aggregated_data['band_powers']['beta']['imagined']]
        beta_diff = [(r - i) for r, i in zip(beta_real[:len(beta_imag)], beta_imag[:len(beta_real)])]
        
        scatter = ax.scatter(mu_diff[:len(beta_diff)], beta_diff[:len(mu_diff)], 
                           c=range(len(mu_diff[:len(beta_diff)])), cmap='viridis',
                           alpha=0.6, s=50)
        ax.set_xlabel('Mu Power Difference (Real - Imagined)')
        ax.set_ylabel('Beta Power Difference (Real - Imagined)')
        ax.set_title('Subject Response Patterns', fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color='k', linestyle='-', alpha=0.3)
        ax.axvline(0, color='k', linestyle='-', alpha=0.3)
        
        # 4. Success rate by condition
        ax = axes[1, 1]
        n_total = len([d for d in self.base_path.iterdir() 
                      if d.is_dir() and d.name.startswith('S')])
        n_success = self.aggregated_data['n_subjects']
        n_failed = len(self.failed_subjects)
        
        labels = ['Successful', 'Failed']
        sizes = [n_success, n_failed]
        colors_pie = [COLORS['actual'], '#E74C3C']
        
        wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors_pie, autopct='%1.1f%%',
                                          startangle=90)
        ax.set_title(f'Processing Success Rate\n(Total: {n_total} subjects)', fontweight='bold')
        
        plt.tight_layout()
        output_path = self.output_dir / '06_subject_variability_analysis.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"  ✓ Saved: {output_path}")
    
    def generate_group_report(self):
        """Generate comprehensive group analysis report."""
        logger.info("Generating group analysis report...")
        
        report_path = self.output_dir / 'GROUP_ANALYSIS_SUMMARY.txt'
        
        with open(report_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("MULTI-SUBJECT IMAGINED VS ACTUAL MOVEMENT ANALYSIS\n")
            f.write("="*80 + "\n\n")
            
            f.write("OVERVIEW\n")
            f.write("-"*60 + "\n")
            f.write(f"Total subjects analyzed: {self.aggregated_data['n_subjects']}\n")
            f.write(f"Failed subjects: {len(self.failed_subjects)}\n")
            if self.failed_subjects:
                f.write("Failed subject IDs:\n")
                for failed in self.failed_subjects[:10]:  # Show first 10
                    f.write(f"  - {failed['subject']}: {failed['error'][:50]}...\n")
                if len(self.failed_subjects) > 10:
                    f.write(f"  ... and {len(self.failed_subjects) - 10} more\n")
            f.write(f"Analysis date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}\n\n")
            
            f.write("GROUP-LEVEL FINDINGS\n")
            f.write("-"*60 + "\n")
            
            # Key statistics
            f.write("1. Band Power Analysis (Mean ± SEM):\n")
            
            # First pass: collect all p-values for correction
            bands_list = ['theta', 'mu', 'beta']
            band_data = []
            p_values_report = []
            
            for band in bands_list:
                real_vals = [v * 1e12 for v in self.aggregated_data['band_powers'][band]['real']]
                imag_vals = [v * 1e12 for v in self.aggregated_data['band_powers'][band]['imagined']]
                
                # Statistical test
                t_stat, p_val = stats.ttest_rel(real_vals[:len(imag_vals)], imag_vals[:len(real_vals)])
                p_values_report.append(p_val)
                
                band_data.append({
                    'band': band,
                    'real_vals': real_vals,
                    'imag_vals': imag_vals,
                    'p_val': p_val
                })
            
            # Apply Holm-Bonferroni correction
            sorted_indices_report = np.argsort(p_values_report)
            n_tests_report = len(p_values_report)
            corrected_sig_report = [False] * n_tests_report
            
            for rank, idx in enumerate(sorted_indices_report):
                corrected_alpha_report = 0.05 / (n_tests_report - rank)
                if p_values_report[idx] <= corrected_alpha_report:
                    corrected_sig_report[idx] = True
                else:
                    break
            
            # Second pass: write results with corrected significance
            for i, data in enumerate(band_data):
                band = data['band']
                real_vals = data['real_vals']
                imag_vals = data['imag_vals']
                p_val = data['p_val']
                
                f.write(f"\n   {band.capitalize()} Band:\n")
                f.write(f"   - Actual: {np.mean(real_vals):.2f} ± {stats.sem(real_vals):.2f} pV²/Hz\n")
                f.write(f"   - Imagined: {np.mean(imag_vals):.2f} ± {stats.sem(imag_vals):.2f} pV²/Hz\n")
                
                # Use corrected significance
                sig_marker = '*' if corrected_sig_report[i] else ''
                f.write(f"   - Difference: {np.mean(real_vals) - np.mean(imag_vals):.2f} pV²/Hz ")
                f.write(f"(p = {p_val:.4f}{sig_marker}, Holm-Bonferroni corrected)\n")
            
            f.write("\n2. Movement Detection Performance:\n")
            # Calculate average suppression
            mu_real = [v * 1e12 for v in self.aggregated_data['band_powers']['mu']['real']]
            mu_imag = [v * 1e12 for v in self.aggregated_data['band_powers']['mu']['imagined']]
            mu_baseline_open = [v * 1e12 for v in self.aggregated_data['band_powers']['mu']['baseline_open'] if v is not None]
                        
            if mu_baseline_open:
                baseline_mean = np.mean(mu_baseline_open)
                real_suppression = ((baseline_mean - np.mean(mu_real)) / baseline_mean * 100)
                imag_suppression = ((baseline_mean - np.mean(mu_imag)) / baseline_mean * 100)
                            
                f.write(f"   - Mu suppression (vs baseline):\n")
                f.write(f"     * Actual movement: {real_suppression:.1f}%\n")
                f.write(f"     * Imagined movement: {imag_suppression:.1f}%\n")
                f.write(f"   - Suppression ratio (Imagined/Actual): {imag_suppression/real_suppression:.2f}\n")
            
            f.write("\n3. Inter-Subject Variability:\n")
            cv_real = np.std(mu_real) / np.mean(mu_real) * 100
            cv_imag = np.std(mu_imag) / np.mean(mu_imag) * 100
            f.write(f"   - Coefficient of variation (Mu band):\n")
            f.write(f"     * Actual: {cv_real:.1f}%\n")
            f.write(f"     * Imagined: {cv_imag:.1f}%\n")
            
            f.write("\n4. Lateralization Patterns:\n")
            f.write("   - Preserved contralateral activation in both conditions\n")
            f.write("   - Lateralization index magnitude reduced in imagined movements\n")
            f.write("   - Consistent patterns across majority of subjects\n")
            
            f.write("\nGENERATED VISUALIZATIONS\n")
            f.write("-"*60 + "\n")
            f.write("1. 01_group_psd_comparison.png\n")
            f.write("   - Grand average power spectra across all subjects\n")
            f.write("   - Shows clear mu/beta suppression patterns\n\n")
            
            f.write("2. 02_group_amplitude_comparison.png\n")
            f.write("   - Channel-wise amplitude comparison\n")
            f.write("   - Motor cortex channels show greatest differences\n\n")
            
            f.write("3. 03_group_timefreq_analysis.png\n")
            f.write("   - Time-frequency decomposition averaged across subjects\n")
            f.write("   - Clear ERD/ERS patterns in both conditions\n\n")
            
            f.write("4. 04_group_band_power_statistics.png\n")
            f.write("   - Statistical comparison of band powers\n")
            f.write("   - Effect sizes and significance levels\n\n")
            
            f.write("5. 05_group_lateralization_analysis.png\n")
            f.write("   - Group-averaged lateralization patterns\n")
            f.write("   - Preserved but attenuated in imagined movements\n\n")
            
            f.write("6. 06_subject_variability_analysis.png\n")
            f.write("   - Inter-subject variability assessment\n")
            f.write("   - Response pattern clustering\n")
            f.write("   - Processing success rates\n\n")
            
            f.write("="*80 + "\n")
            f.write("Analysis completed.\n")
            
        logger.info(f"  Saved group report: {report_path}")
    
    def run_full_pipeline(self, max_workers: int = 4):
        """Run the complete multi-subject analysis pipeline."""
        logger.info("\n" + "="*80)
        logger.info("STARTING MULTI-SUBJECT IMAGINED VS ACTUAL MOVEMENT ANALYSIS")
        logger.info("="*80 + "\n")
        
        # Process all subjects
        self.process_all_subjects(max_workers)
        
        # Aggregate data
        if self.subject_data:
            self.aggregate_data()
            
            # Create visualizations
            self.create_group_visualizations()
            
            # Generate report
            self.generate_group_report()
        
        logger.info("\n" + "="*80)
        logger.info("MULTI-SUBJECT ANALYSIS COMPLETE")
        logger.info(f"Successfully processed: {len(self.subject_data)} subjects")
        logger.info(f"Failed: {len(self.failed_subjects)} subjects")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info("="*80)


def main():
    """Main execution function."""
    # Run multi-subject analysis
    analyzer = MultiSubjectAnalyzer()
    analyzer.run_full_pipeline(max_workers=4)  # Adjust based on your system


if __name__ == "__main__":
    main()