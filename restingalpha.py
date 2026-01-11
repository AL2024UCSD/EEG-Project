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

# PROCESS
    # 1. USe Welch's method to reduce noise
    # 2. integrate over alpha band frequency
    # 3. compute power relative to other frequency bands
        # RPL = alpha power / total power
    # 4. average over motor cortex channels

# Notes:
     # remember: resting state, so from -1 to 0 seconds
     # consider doing power with relative ERD/ERS and not absolute


def resting_alpha_power(subject_id, base_path):
    """Computing resting Alpha Power for one subject"""

    checker = False

    subject_path = Path(base_path) / subject_id
    filename = subject_path / f"{subject_id}R02.edf"    # R02 corresponds to eyes-closed baseline

    channels = ['C3', 'C4', 'Cz', 'FC1', 'FC2', 'CP1', 'CP2']
    
    raw = mne.io.read_raw_edf(filename, preload=True, verbose=False)
    raw.rename_channels(lambda x: x.strip('.'))
    
    # Pick motor channels if available
    available_channels = [ch for ch in channels if ch in raw.ch_names]

    raw.pick_channels(available_channels)

    #Bandpass filter, 0.5 to 40 Hz
    raw.filter(0.5, 40, fir_design='firwin', verbose=False)
    alpha_psd = raw.compute_psd(
                method='welch', 
                fmin=8,     #alpha band start freq
                fmax=13,    # alpha band end freq
                n_fft=2048,  # window size; 2048 is pretty large -> better frequency distinction
                n_overlap= 1024,    # convention is n_fft/2
                verbose=False
            )
    alpha_psd_data, freqs = alpha_psd.get_data(return_freqs=True)   # alpha_psd_data is 2D, (n_channels, n_freqs), freqs is 1D array
    resting_state_alpha_power = alpha_psd_data.mean(axis=1).mean(axis = 0)  # Average over frequencies then channels, result is single val

    total_psd = raw.compute_psd(       #get total PSD by not restricting to alpha band freqs
        method = 'welch',
        fmin = 0.5,
        fmax = 40,
        n_fft = 2048,
        n_overlap = 1024,
        verbose = False
    )
    total_psd_data, freqs = total_psd.get_data(return_freqs = True)
    resting_state_total_power = total_psd_data.mean(axis=1).mean(axis=0)

    # Calculating Relative Power Level (RPL)
    rpl = resting_state_alpha_power/resting_state_total_power

    if rpl is not None:
        checker = True

    # Considerations:
        # can average across just frequencies
        # can average across just channels
        # can average across both (for a single number) (current)

    # Additional, general considerations:
        # 1. resting alpha power relative to other bands
        # 2. resting alpha power with eyes open, and/or average across the two

    if checker:
        return {
            'subject_id': subject_id,
            'alpha_power': resting_state_alpha_power,
            'total_power': resting_state_total_power,
            'rpl': rpl,
            'success': True,
            'error': None
        }
    else:
       return {
            'subject_id': subject_id,
            'alpha_power': None,
            'total_power': None,
            'rpl': None,
            'success': False,
            'error': True
        } 



def all_subjects_analysis(base_path = 'eeg-motor-movementimagery-dataset-1.0.0/files'):
    """Extract feature for all subjects"""

    base_path = Path(base_path)
    
    # Get all subject directories
    subject_dirs = sorted([d for d in base_path.iterdir() 
                              if d.is_dir() and d.name.startswith('S') and d.name[1:].isdigit()])
    
    
    resting_alpha_powers = []
    
    for subject in subject_dirs:
        resting_alpha = resting_alpha_power(subject.name, base_path)   # subject.name ex: 'S001'
        resting_alpha_powers.append(resting_alpha)
    
    resting_alpha_powers = np.array(resting_alpha_powers)
    
    for i in resting_alpha_powers:
        print('\n')
        print(f"subject: {i['subject_id']}")
        print(f"alpha power: {i['alpha_power']}")
        print(f"total_power: {i['total_power']}")
        print(f"rpl: {i['rpl']}")

if __name__ == "__main__":
    all_subjects_analysis()
