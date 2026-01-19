import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import mne
from scipy import signal, stats, optimize
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


# Second feature: SMR baseline strength
def baseline_smr_strength(subject_id, base_path):
    """Computing SMR Baseline Strength for one subject"""
    checker = False

    subject_path = Path(base_path) / subject_id
    filename = subject_path / f"{subject_id}R01.edf"    # R01 corresponds to eyes-open baseline

    channels = ['C3', 'C4']
    
    # Laplacian Filtering - for reducing noise and volume, and isolating local SMR
    c3_surrounding = ['FC3', 'C1', 'CP3', 'C5']
    c4_surrounding = ['FC4', 'CP4', 'C2', 'C6']
    
    raw = mne.io.read_raw_edf(filename, preload=True, verbose=False)
    raw.rename_channels(lambda x: x.strip('.'))
    
    # Pick motor channels if available
    available_channels = [ch for ch in channels if ch in raw.ch_names]
    available_c3 = [ch for ch in c3_surrounding if ch in raw.ch_names]
    available_c4 = [ch for ch in c4_surrounding if ch in raw.ch_names]

    all_channels_needed = available_channels + available_c3 + available_c4

    raw.pick_channels(all_channels_needed)

    #Bandpass filter, 2 to 35 Hz, as used in Blankertz et al. for their feature calculation
    raw.filter(2, 35, fir_design='firwin', verbose=False)


    data = raw.get_data()   # 2D numpy array, (n_channels x n_timepoints)
    ch_names = raw.ch_names #list of channel names as strings

    # Laplacians:
    c3_idx = ch_names.index('C3')
    c3_surrounding_idx = [ch_names.index(ch) for ch in available_c3]
    c3_laplacian = data[c3_idx, :] - np.mean(data[c3_surrounding_idx, :], axis = 0)

    c4_idx = ch_names.index('C4')
    c4_surrounding_idx = [ch_names.index(ch) for ch in available_c4]
    c4_laplacian = data[c4_idx, :] - np.mean(data[c4_surrounding_idx, :], axis = 0)


    # SMR = sensorimotor rhyth, includes alpha (mu) and beta bands, range is 8-30 Hz
    # can't use MNE compute_psd anymore since working with np arrays and not raw
    
    freqs, smr_psd_c3 = signal.welch(
        c3_laplacian,
        fs = raw.info['sfreq'],
        window = 'hann',    # default, standard
        nperseg = 2048,
        noverlap = 1024
    )

    freqs, smr_psd_c4 = signal.welch(
        c4_laplacian,
        fs = raw.info['sfreq'],
        window = 'hann',    # default, standard
        nperseg = 2048,
        noverlap = 1024
    )

    smr_psd_data_c3 = 10 * np.log10(smr_psd_c3)  # convert psd to dB
    smr_psd_data_c4 = 10 * np.log10(smr_psd_c4)

    #freqs filtering for SMR range 8-30 Hz
    freqs_mask = (freqs >= 2) & (freqs <= 35)
    freqs = freqs[freqs_mask]
    psd_c3_subset = smr_psd_data_c3[freqs_mask] # keeps only dB readings at positions of valid frequency
    psd_c4_subset = smr_psd_data_c4[freqs_mask]


    # Fitting 1/freqs gaussian model for C3

    bounds = (
    [0.5, -150, 0, 0, 0, 8, 13, 1, 1],        # Lower bounds
    [2.0,  150, 50, 20, 20, 13, 30, 10, 10]   # Upper bounds
    )

    k1_guess = np.clip(psd_c3_subset[-1], -20, 20)    # approximate expected high-freq value
    initial_param_guesses = [
        1.0,    # lambda starting with 1/f
        k1_guess,  
        10.0,   # conventional default guess for k2
        5.0,    # assume similar heights for k3 and k4 initially
        5.0,
        10.0,   # expected center for alpha
        23.0,   # expected center for beta
        3.0,    # typical default conventional guesses for widths
        3.0
    ]

    optimal_params_c3, c3_cov_matrix = optimize.curve_fit(    # look for optimal parameters for 1/freqs
        freq_curve,
        freqs,
        psd_c3_subset,
        p0 = initial_param_guesses,
        bounds = bounds,
        maxfev = 5000   # max function evaluations
    )

    lambda_fit, k1, k2, k3, k4, mu1, mu2, sig1, sig2 = optimal_params_c3
    g_total_c3 = freq_curve(freqs, *optimal_params_c3)  # full fitted curve in dB
    g1_c3 = k1 + k2 / (freqs ** lambda_fit)     # noise componenet in dB
    g2_c3 = g_total_c3 - g1_c3      # g2 = total - noise in dB
    smr_strength_c3 = np.max(g2_c3)     # maximum peak height above noise

    # Fitting 1/freqs gaussian model for C4
    k1_guess = np.clip(psd_c4_subset[-1], -20, 20)    # approximate expected high-freq value
    initial_param_guesses = [
        1.0,    # lambda starting with 1/f
        k1_guess,  
        10.0,   # conventional default guess for k2
        5.0,    # assume similar heights for k3 and k4 initially
        5.0,
        10.0,   # expected center for alpha
        23.0,   # expected center for beta
        3.0,    # typical default conventional guesses for widths
        3.0
    ]

    optimal_params_c4, c4_cov_matrix = optimize.curve_fit(    # look for optimal parameters for 1/freqs
        freq_curve,
        freqs,
        psd_c4_subset,
        p0 = initial_param_guesses,
        bounds = bounds,
        maxfev = 5000   # max function evaluations
    )

    lambda_fit, k1, k2, k3, k4, mu1, mu2, sig1, sig2 = optimal_params_c4
    g_total_c4 = freq_curve(freqs, *optimal_params_c4)  # full fitted curve in dB
    g1_c4 = k1 + k2 / (freqs ** lambda_fit)     # noise componenet in dB
    g2_c4 = g_total_c4 - g1_c4      # g2 = total - noise in dB
    smr_strength_c4 = np.max(g2_c4)     # maximum peak height above noise

    smr_strength = (smr_strength_c4 + smr_strength_c3) / 2


    if smr_strength is not None:
        checker = True

    if checker:
        return smr_strength
    else:
        return None
    
def freq_curve(freqs, lambda_val, k1, k2, k3, k4, mu1, mu2, sig1, sig2):
    # Parameters:
        # freqs: array of frequencies

        # k1: baseline offset (shufts whole curve up/down)
        # k2: scale factor (how steep the 1/f decline is)
        # lambda_val: exponent (controls slope)
        
        # mu1, mu2: peak positions
        # sig1, sig2: peak widths
        # k3, k4: peak amplitudes
    
    # Notes: PSD modeled as two components:
        # g1(freqs) = Noise floor: k1 + k2/f^lambda_val
        # g2(freqs) = Two gaussian peaks: k3 * gaussian1 + k4 * gaussian2 (alpha (mu) and beta peaks)
        # total output: g1(freqs) + g2(freqs)
    

    g1 = k1 + k2/(freqs**lambda_val)

    gaussian1 = stats.norm.pdf(freqs, loc = mu1, scale = sig1)
    gaussian2 = stats.norm.pdf(freqs, loc = mu2, scale = sig2)
    g2 = k3 * gaussian1 + k4*gaussian2

    return g1 + g2


# Feature 3: ERD/ERS Magnitude
def erd_ers_magnitude(subject_id, base_path):
    checker = False

    subject_path = Path(base_path) / subject_id

    channels = ['C3', 'C4', 'Cz']

    magnitudes = {}

    for i in ['R03', 'R04']:
        filename = subject_path / f"{subject_id}{i}.edf"    
        # R03 corresponds to real left/right fist, R04 corresponds to imagined left/right fist
    

        raw = mne.io.read_raw_edf(filename, preload=True, verbose=False)
        raw.rename_channels(lambda x: x.strip('.'))
    
        # Pick motor channels if available
        available_channels = [ch for ch in channels if ch in raw.ch_names]

        raw.pick_channels(available_channels)

        # Bandpass filter
        raw.filter(0.5, 40, fir_design='firwin', verbose=False)

        data = raw.get_data()   # 2D numpy array, (n_channels x n_timepoints)
        ch_names = raw.ch_names #list of channel names as strings

        events, event_dict = mne.events_from_annotations(raw, verbose=False)
        t1_id = event_dict['T1']    # event dict and events code T1 and T2 as ints
        t2_id = event_dict['T2']
        movements = events[(events[:, 2] == t1_id) | (events[:, 2] == t2_id)]

        # Create epochs
        epochs = mne.Epochs(raw, 
                            movements,
                            tmin=-1.0, 
                            tmax=4.0, 
                            baseline=None,
                            preload=True, 
                            verbose=False)
        #crop to appropriate time windows
        epochs_baseline = epochs.copy().crop(tmin = -1.0, tmax = 0.0)
        epochs_movement = epochs.copy().crop(tmin = 0.5, tmax = 4.0)

        # compute psd
        psd_baseline = epochs_baseline.compute_psd(
                    method='welch', fmin=8, fmax=30, n_fft=128,
                    n_overlap=64, verbose=False
                )
        psd_movement = epochs_movement.compute_psd(
                    method='welch', fmin=8, fmax=30, n_fft=128,
                    n_overlap=64, verbose=False
                )
        
        psd_baseline_data, freqs = psd_baseline.get_data(return_freqs=True)
        psd_movement_data, freqs = psd_movement.get_data(return_freqs = True)

        # boolean masks
        mu_freqs = (freqs >= 8) & (freqs <= 13)
        beta_freqs = (freqs >= 13) & (freqs <= 30)

        # axis = -1 is averaging across frequencies, one value per channel
        baseline_mu_power = psd_baseline_data[:, :, mu_freqs].mean(axis = -1)   # baseline power for mu frequencies
        movement_mu_power = psd_movement_data[:, :, mu_freqs].mean(axis = -1)   # movement power for mu frequencies
        baseline_beta_power = psd_baseline_data[:, :, beta_freqs].mean(axis = -1)   # baseline power for beta frequencies
        movement_beta_power = psd_movement_data[:, :, beta_freqs].mean(axis = -1)   # movement power for beta frequencies

        mu_percent_change = (movement_mu_power - baseline_mu_power) / baseline_mu_power * 100
        beta_percent_change = (movement_beta_power - baseline_beta_power) / baseline_beta_power * 100

        mu_erd_magnitude = np.min(mu_percent_change, axis = 0)
        beta_erd_magnitude = np.min(beta_percent_change, axis = 0)
        mu_ers_magnitude = np.max(mu_percent_change, axis = 0)
        beta_ers_magnitude = np.max(beta_percent_change, axis = 0)

        if i == 'R03':
            magnitudes['real'] = {
                "mu_erd": mu_erd_magnitude,   # array, one for each channel
                "beta_erd": beta_erd_magnitude,
                "mu_ers": mu_ers_magnitude,
                "beta_ers": beta_ers_magnitude
            }
        else:
            magnitudes['imagined'] = {
                "mu_erd": mu_erd_magnitude,
                "beta_erd": beta_erd_magnitude,
                "mu_ers": mu_ers_magnitude,
                "beta_ers": beta_ers_magnitude
            }
    return magnitudes


def all_subjects_analysis(base_path = 'eeg-motor-movementimagery-dataset-1.0.0/files'):
    """Extract feature for all subjects"""

    base_path = Path(base_path)
    
    # Get all subject directories
    subject_dirs = sorted([d for d in base_path.iterdir() 
                              if d.is_dir() and d.name.startswith('S') and d.name[1:].isdigit()])
    
    # FEATURE 1: Resting Alpha Power
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


    # FEATURE 2: SMR Baseline Strength
    smr_arr = []
    for subject in subject_dirs:
            smr = baseline_smr_strength(subject.name, base_path)
            smr_arr.append(smr)
            print('\n')
            print(f"subject: {subject.name}")
            print(f"SMR Strength: {smr}")

    # FEATURE 3: ERD/ERS Magnitude
    erd_ers_arr = []
    for subject in subject_dirs:
        erd_ers = erd_ers_magnitude(subject.name, base_path)
        erd_ers_arr.append(erd_ers)
        print('\n')
        print(f"subject: {subject.name}")
        print(f"erd_ers: {erd_ers}")


            


# def test_single_subject():
#     base_path = 'eeg-motor-movementimagery-dataset-1.0.0/files'
#     subject_id = 'S001'
    
#     print(f"Testing {subject_id}...")
    
#     try:
#         erd_ers = erd_ers_magnitude(subject_id, base_path)
#         print(f"ERD/ERS: {erd_ers}")
#         print("Success!")
#     except Exception as e:
#         print(f"Error: {e}")
#         import traceback
#         traceback.print_exc()

if __name__ == "__main__":
    # test_single_subject()
    all_subjects_analysis()
