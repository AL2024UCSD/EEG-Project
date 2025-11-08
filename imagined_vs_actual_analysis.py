"""
EEG Analysis - Imagined vs Actual Movements
Trying to figure out if imagined movements look like real ones in brain signals

Shaheer Khan - Oct 2025
Started: Oct 15, revised Oct 20
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import mne
from scipy import signal, stats
import pandas as pd
import logging
from typing import Dict, List, Tuple
from matplotlib.gridspec import GridSpec
import matplotlib.patches as mpatches
from mpl_toolkits.axes_grid1 import make_axes_locatable

# TODO: Clean this up later
# TODO: Add more frequency bands?
# NOTE: Some plots might need tweaking - check with advisor

# Plotting settings - adjusted after first few runs looked ugly
plt.rcParams['figure.dpi'] = 300  
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
# plt.rcParams['font.family'] = 'serif'  # Changed back to sans-serif, serif looked weird
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['figure.titlesize'] = 16

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Color palette - spent way too long picking these
# Original colors looked bad, these are better
COLORS = {
    'actual': '#2E86AB',      
    'imagined': '#A23B72',    
    'difference': '#F18F01',  
    'left_fist': '#0077BE',   
    'right_fist': '#DC143C',  
    'both_fists': '#228B22',  
    'both_feet': '#FF8C00'    
}

class ImaginedVsActualAnalyzer:
    """Main analyzer class - does all the heavy lifting"""
    
    def __init__(self, subject_id='S001', base_path='raw_data'):
        """Set up the analyzer"""
        self.subject_id = subject_id
        self.base_path = Path(base_path) / subject_id
        self.output_dir = Path(f'{subject_id}_imagined_vs_actual')
        self.output_dir.mkdir(exist_ok=True)
        
        # Motor cortex channels - these are the important ones
        self.channels = ['C3', 'C4', 'Cz', 'FC1', 'FC2', 'CP1', 'CP2']
        
        # Baseline runs - eyes open/closed
        self.baseline_runs = {
            'eyes_open': 1,
            'eyes_closed': 2
        }
        
        # Task pairs - real vs imagined for each task type
        # Run numbers from PhysioNet dataset docs
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
        
    def load_and_preprocess_data(self):
        """Load all the EEG data and clean it up"""
        print("="*70)
        print(f"Loading data for {self.subject_id}...")
        print("="*70)
        
        # Load baseline runs first
        for baseline_type, run_id in self.baseline_runs.items():
            filename = self.base_path / f"{self.subject_id}R{run_id:02d}.edf"
            print(f"\nLoading baseline: {baseline_type} (Run {run_id})")
            
            try:
                raw = mne.io.read_raw_edf(filename, preload=True, verbose=False)
                raw.rename_channels(lambda x: x.strip('.'))  # Remove dots from channel names
                
                # Pick motor channels
                available_channels = [ch for ch in self.channels if ch in raw.ch_names]
                if not available_channels:
                    print(f"WARNING: No motor channels found in {filename}")
                    continue
                
                raw.pick_channels(available_channels)
                
                # Bandpass filter - 0.5-40 Hz is standard for motor imagery
                raw.filter(0.5, 40, fir_design='firwin', verbose=False)
                
                self.data[f'baseline_{baseline_type}'] = {
                    'raw': raw,
                    'type': 'baseline',
                    'condition': baseline_type
                }
                
                print(f"  ✓ Loaded: {raw.times[-1]:.1f}s, {len(available_channels)} channels")
                
            except Exception as e:
                print(f"ERROR loading baseline {baseline_type}: {e}")
        
        # Load task runs
        for pair in self.task_pairs:
            for condition, run_id in [('real', pair['real']), ('imagined', pair['imagined'])]:
                key = f"{pair['name']}_{condition}"
                filename = self.base_path / f"{self.subject_id}R{run_id:02d}.edf"
                
                print(f"\nLoading {key} (Run {run_id})")
                
                try:
                    raw = mne.io.read_raw_edf(filename, preload=True, verbose=False)
                    raw.rename_channels(lambda x: x.strip('.'))
                    
                    available_channels = [ch for ch in self.channels if ch in raw.ch_names]
                    if not available_channels:
                        print(f"WARNING: No motor channels in {filename}")
                        continue
                    
                    raw.pick_channels(available_channels)
                    raw.filter(0.5, 40, fir_design='firwin', verbose=False)
                    
                    # Get events - these mark when each movement started
                    events, event_dict = mne.events_from_annotations(raw, verbose=False)
                    
                    # Create epochs - chunks of data around each movement
                    # -1 to 4s window, baseline correction from -1 to 0
                    epochs = mne.Epochs(raw, events, event_id=event_dict,
                                       tmin=-1.0, tmax=4.0, baseline=(-1, 0),
                                       preload=True, verbose=False)
                    
                    self.data[key] = {
                        'raw': raw,
                        'events': events,
                        'event_dict': event_dict,
                        'epochs': epochs,
                        'pair_info': pair
                    }
                    
                    print(f"  ✓ Loaded: {len(epochs)} epochs, {len(available_channels)} channels")
                    
                except Exception as e:
                    print(f"ERROR loading {key}: {e}")
        
        print(f"\n✓ Loaded {len(self.data)} datasets total")
        
    def compute_erds_features(self, epochs, freqs=None):
        """
        Compute ERD/ERS - Event-Related Desynchronization/Synchronization
        This shows how brain rhythms change during movement
        """
        if freqs is None:
            freqs = np.arange(4, 40, 1)  # 4-40 Hz in 1 Hz steps
        
        # Time-frequency decomposition using Morlet wavelets
        power = mne.time_frequency.tfr_morlet(
            epochs, freqs=freqs, n_cycles=freqs/2,
            use_fft=True, return_itc=False, average=True,
            n_jobs=1, verbose=False
        )
        
        # Baseline correction - convert to percent change
        baseline_power = power.copy().crop(tmin=-1.0, tmax=0.0).data.mean(axis=-1, keepdims=True)
        erds = ((power.data - baseline_power) / baseline_power) * 100
        
        return power, erds
    
    def plot_overview_comparison(self):
        """Main comparison plot - this is the big one"""
        print("\nCreating overview comparison...")
        
        fig = plt.figure(figsize=(22, 14))
        fig.suptitle(f'{self.subject_id} - Imagined vs Actual Movement', 
                    fontsize=20, fontweight='bold', y=0.98)
        
        gs = GridSpec(3, 4, figure=fig, hspace=0.4, wspace=0.4)
        
        # 1. Power spectrum comparison
        ax_psd = fig.add_subplot(gs[0, :2])
        self._plot_average_psd_comparison(ax_psd)
        
        # 2. Channel amplitudes
        ax_amp = fig.add_subplot(gs[0, 2:])
        self._plot_amplitude_comparison(ax_amp)
        
        # 3. Time-frequency maps
        for idx, task_type in enumerate(['fist', 'fists_feet']):
            ax_tf_real = fig.add_subplot(gs[1, idx*2])
            ax_tf_imag = fig.add_subplot(gs[1, idx*2+1])
            self._plot_timefreq_comparison(ax_tf_real, ax_tf_imag, task_type)
        
        # 4. Stats table
        ax_stats = fig.add_subplot(gs[2, :])
        self._plot_statistical_comparison(ax_stats)
        
        output_path = self.output_dir / '01_overview_comparison.png'
        plt.savefig(output_path, bbox_inches='tight', dpi=300, facecolor='white')
        print(f"  ✓ Saved: {output_path}")
        plt.close()
        
    def _plot_average_psd_comparison(self, ax):
        """Plot power spectral density - frequency content of signals"""
        freqs = np.arange(1, 40, 0.5)
        psd_real_all = []
        psd_imag_all = []
        psd_baseline_open = []
        psd_baseline_closed = []
        
        # Compute PSDs for all conditions
        for key, data in self.data.items():
            if 'baseline_eyes_open' in key:
                psd = data['raw'].compute_psd(
                    method='welch', fmin=1, fmax=40, n_fft=512,
                    n_overlap=256, verbose=False
                )
                psd_data, freqs = psd.get_data(return_freqs=True)
                psd_baseline_open = psd_data.mean(axis=0)
            elif 'baseline_eyes_closed' in key:
                psd = data['raw'].compute_psd(
                    method='welch', fmin=1, fmax=40, n_fft=512,
                    n_overlap=256, verbose=False
                )
                psd_data, freqs = psd.get_data(return_freqs=True)
                psd_baseline_closed = psd_data.mean(axis=0)
            elif 'real' in key and 'epochs' in data:
                psd = data['epochs'].compute_psd(
                    method='welch', fmin=1, fmax=40, n_fft=512,
                    n_overlap=256, verbose=False
                )
                psd_data, freqs = psd.get_data(return_freqs=True)
                psd_real_all.append(psd_data.mean(axis=(0, 1)))
            elif 'imagined' in key and 'epochs' in data:
                psd = data['epochs'].compute_psd(
                    method='welch', fmin=1, fmax=40, n_fft=512,
                    n_overlap=256, verbose=False
                )
                psd_data, freqs = psd.get_data(return_freqs=True)
                psd_imag_all.append(psd_data.mean(axis=(0, 1)))
        
        # Average and convert to dB
        psd_real_mean = np.mean(psd_real_all, axis=0)
        psd_imag_mean = np.mean(psd_imag_all, axis=0)
        psd_real_sem = stats.sem(psd_real_all, axis=0)
        psd_imag_sem = stats.sem(psd_imag_all, axis=0)
        
        psd_real_db = 10 * np.log10(psd_real_mean)
        psd_imag_db = 10 * np.log10(psd_imag_mean)
        if len(psd_baseline_open) > 0:
            psd_baseline_open_db = 10 * np.log10(psd_baseline_open)
        if len(psd_baseline_closed) > 0:
            psd_baseline_closed_db = 10 * np.log10(psd_baseline_closed)
        
        # Plot baselines first (in background)
        if len(psd_baseline_open) > 0:
            ax.plot(freqs, psd_baseline_open_db, color='#7F8C8D', linewidth=2, 
                   linestyle='--', label='Baseline (Eyes Open)', alpha=0.7)
        if len(psd_baseline_closed) > 0:
            ax.plot(freqs, psd_baseline_closed_db, color='#34495E', linewidth=2, 
                   linestyle='--', label='Baseline (Eyes Closed)', alpha=0.7)
        
        # Plot movement conditions with confidence bands
        ax.plot(freqs, psd_real_db, color=COLORS['actual'], linewidth=2, label='Actual')
        ax.fill_between(freqs, 
                       10*np.log10(psd_real_mean - psd_real_sem),
                       10*np.log10(psd_real_mean + psd_real_sem),
                       alpha=0.3, color=COLORS['actual'])
        
        ax.plot(freqs, psd_imag_db, color=COLORS['imagined'], linewidth=2, label='Imagined')
        ax.fill_between(freqs,
                       10*np.log10(psd_imag_mean - psd_imag_sem),
                       10*np.log10(psd_imag_mean + psd_imag_sem),
                       alpha=0.3, color=COLORS['imagined'])
        
        # Highlight important frequency bands
        ax.axvspan(8, 13, alpha=0.1, color='gray', label='Mu (8-13 Hz)')
        ax.axvspan(13, 30, alpha=0.1, color='lightgray', label='Beta (13-30 Hz)')
        
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Power (dB)')
        ax.set_title('Power Spectrum: Actual vs Imagined', fontweight='bold')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(1, 40)
        
    def _plot_amplitude_comparison(self, ax):
        """Compare signal amplitudes across channels"""
        channels = []
        amplitudes_real = []
        amplitudes_imag = []
        amplitudes_baseline_open = []
        amplitudes_baseline_closed = []
        
        # Find available channels
        first_key = None
        for key, data in self.data.items():
            if 'epochs' in data:
                first_key = key
                break
        
        if not first_key:
            ax.text(0.5, 0.5, 'No epoch data', ha='center', va='center')
            ax.axis('off')
            return
            
        available_channels = self.data[first_key]['epochs'].ch_names
        
        # Calculate RMS amplitude for each channel
        for ch in available_channels:
            ch_amps_real = []
            ch_amps_imag = []
            ch_baseline_open = None
            ch_baseline_closed = None
            
            # Baselines
            for key, data in self.data.items():
                if 'baseline_eyes_open' in key and ch in data['raw'].ch_names:
                    ch_idx = data['raw'].ch_names.index(ch)
                    raw_data = data['raw'].get_data()[ch_idx, :]
                    ch_baseline_open = np.sqrt(np.mean(raw_data**2)) * 1e6  # µV
                elif 'baseline_eyes_closed' in key and ch in data['raw'].ch_names:
                    ch_idx = data['raw'].ch_names.index(ch)
                    raw_data = data['raw'].get_data()[ch_idx, :]
                    ch_baseline_closed = np.sqrt(np.mean(raw_data**2)) * 1e6
            
            # Tasks
            for key, data in self.data.items():
                if 'epochs' in data and ch in data['epochs'].ch_names:
                    ch_idx = data['epochs'].ch_names.index(ch)
                    epochs_data = data['epochs'].get_data()[:, ch_idx, :]
                    
                    # RMS during movement (0-3s)
                    sfreq = data['epochs'].info['sfreq']
                    movement_data = epochs_data[:, int(1*sfreq):int(3*sfreq)]
                    rms_amp = np.sqrt(np.mean(movement_data**2))
                    
                    if 'real' in key:
                        ch_amps_real.append(rms_amp * 1e6)
                    elif 'imagined' in key:
                        ch_amps_imag.append(rms_amp * 1e6)
            
            channels.append(ch)
            amplitudes_real.append(np.mean(ch_amps_real) if ch_amps_real else 0)
            amplitudes_imag.append(np.mean(ch_amps_imag) if ch_amps_imag else 0)
            amplitudes_baseline_open.append(ch_baseline_open if ch_baseline_open is not None else 0)
            amplitudes_baseline_closed.append(ch_baseline_closed if ch_baseline_closed is not None else 0)
        
        # Bar plot with 4 conditions
        x = np.arange(len(channels))
        width = 0.2
        
        bars1 = ax.bar(x - 1.5*width, amplitudes_baseline_open, width, 
                      label='Baseline (Open)', color='#7F8C8D', alpha=0.7)
        bars2 = ax.bar(x - 0.5*width, amplitudes_baseline_closed, width, 
                      label='Baseline (Closed)', color='#34495E', alpha=0.7)
        bars3 = ax.bar(x + 0.5*width, amplitudes_real, width, 
                      label='Actual', color=COLORS['actual'])
        bars4 = ax.bar(x + 1.5*width, amplitudes_imag, width, 
                      label='Imagined', color=COLORS['imagined'])
        
        ax.set_xlabel('Channel')
        ax.set_ylabel('RMS Amplitude (µV)')
        ax.set_title('Amplitude by Channel', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(channels)
        ax.legend(loc='upper right', ncol=2)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add labels (only for movement conditions to avoid clutter)
        for bars in [bars3, bars4]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1f}', ha='center', va='bottom', fontsize=8)
    
    # ... (rest of the methods would follow similar pattern)
    
    def run_full_analysis(self):
        """Run everything"""
        print("\n" + "="*70)
        print("STARTING ANALYSIS")
        print("="*70)
        
        self.load_and_preprocess_data()
        self.plot_overview_comparison()
        # Add other plot calls...
        
        print("\n" + "="*70)
        print("DONE!")
        print(f"Check: {self.output_dir}")
        print("="*70)


def main():
    # Run for subject S001
    analyzer = ImaginedVsActualAnalyzer(subject_id='S001')
    analyzer.run_full_analysis()


if __name__ == "__main__":
    main()
