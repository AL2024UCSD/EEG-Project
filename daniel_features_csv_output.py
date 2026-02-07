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

    subject_path = Path(base_path) / subject_id
    filename = subject_path / f"{subject_id}R02.edf"    # R02 corresponds to eyes-closed baseline

    channels = ['C3', 'C4', 'Cz', 'FC1', 'FC2', 'CP1', 'CP2']
    
    raw = mne.io.read_raw_edf(filename, preload=True, verbose=False)
    raw.rename_channels(lambda x: x.strip('.'))
    
    # Pick motor channels if available
    available_channels = [ch for ch in channels if ch in raw.ch_names]

    raw.pick_channels(available_channels)

    #Bandpass filter, 2 to 35 Hz
    raw.filter(2, 35, fir_design='firwin', verbose=False)
    alpha_psd = raw.compute_psd(
                method='welch', 
                fmin=8,     #alpha band start freq
                fmax=13,    # alpha band end freq
                n_fft=2048,  # window size; 2048 is pretty large -> better frequency distinction
                n_overlap= 1024,    # convention is n_fft/2
                verbose=False
            )
    alpha_psd_data, freqs = alpha_psd.get_data(return_freqs=True)   # alpha_psd_data is 2D, (n_channels, n_freqs), freqs is 1D array
    resting_state_alpha_power = np.trapezoid(alpha_psd_data, freqs, axis = 1).mean()  #Integrate over frequencies, avg across channels

    total_psd = raw.compute_psd(       #get total PSD by not restricting to alpha band freqs
        method = 'welch',
        fmin = 2,
        fmax = 35,
        n_fft = 2048,
        n_overlap = 1024,
        verbose = False
    )
    total_psd_data, freqs = total_psd.get_data(return_freqs = True)
    resting_state_total_power = np.trapezoid(total_psd_data, freqs, axis = 1).mean()

    # Calculating Relative Power Level (RPL)
    rpl = resting_state_alpha_power/resting_state_total_power

    # Considerations:
        # can average across just frequencies
        # can average across just channels
        # can average across both (for a single number) (current)

    # Additional, general considerations:
        # 1. resting alpha power relative to other bands
        # 2. resting alpha power with eyes open, and/or average across the two

    return {
        "rpl_alpha": rpl, 
        "resting alpha power": resting_state_alpha_power, 
        "resting total power": resting_state_total_power
    }

    # if checker:
    #     return {
    #         'subject_id': subject_id,
    #         'alpha_power': resting_state_alpha_power,
    #         'total_power': resting_state_total_power,
    #         'rpl': rpl,
    #         'success': True,
    #         'error': None
    #     }
    # else:
    #    return {
    #         'subject_id': subject_id,
    #         'alpha_power': None,
    #         'total_power': None,
    #         'rpl': None,
    #         'success': False,
    #         'error': True
    #     }


def resting_beta_power(subject_id, base_path):
    """Computing resting Beta Power for one subject"""

    subject_path = Path(base_path) / subject_id
    filename = subject_path / f"{subject_id}R02.edf"    # R02 corresponds to eyes-closed baseline

    channels = ['C3', 'C4', 'Cz', 'FC1', 'FC2', 'CP1', 'CP2']
    
    raw = mne.io.read_raw_edf(filename, preload=True, verbose=False)
    raw.rename_channels(lambda x: x.strip('.'))
    
    # Pick motor channels if available
    available_channels = [ch for ch in channels if ch in raw.ch_names]

    raw.pick_channels(available_channels)

    #Bandpass filter, 2 to 35 Hz
    raw.filter(2, 35, fir_design='firwin', verbose=False)
    beta_psd = raw.compute_psd(
                method='welch', 
                fmin=13,     #beta band start freq
                fmax=30,    # beta band end freq
                n_fft=2048,  # window size; 2048 is pretty large -> better frequency distinction
                n_overlap= 1024,    # convention is n_fft/2
                verbose=False
            )
    beta_psd_data, freqs = beta_psd.get_data(return_freqs=True)   # betapsd_data is 2D, (n_channels, n_freqs), freqs is 1D array
    resting_state_beta_power = np.trapezoid(beta_psd_data, freqs, axis = 1).mean()  #Integrate over frequencies, avg across channels

    total_psd = raw.compute_psd(       #get total PSD by not restricting to alpha band freqs
        method = 'welch',
        fmin = 2,
        fmax = 35,
        n_fft = 2048,
        n_overlap = 1024,
        verbose = False
    )
    total_psd_data, freqs = total_psd.get_data(return_freqs = True)
    resting_state_total_power = np.trapezoid(total_psd_data, freqs, axis = 1).mean()

    # Calculating Relative Power Level (RPL)
    rpl = resting_state_beta_power/resting_state_total_power

    # Considerations:
        # can average across just frequencies
        # can average across just channels
        # can average across both (for a single number) (current)

    # Additional, general considerations:
        # 1. resting alpha power relative to other bands
        # 2. resting alpha power with eyes open, and/or average across the two

    return {
        "rpl_beta": rpl, 
        "resting beta power": resting_state_beta_power, 
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


    # SMR = sensorimotor rhythm, includes alpha (mu) and beta bands, range is 8-30 Hz
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

    #freqs filtering for SMR range 2-35 Hz
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
        return {"smr strength": smr_strength}
    else:
        return {"smr strength": None}
    
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


# Feature 3: ERD/ERS Magnitude      (!!!CUT BECAUSE NEEDS MI)
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
        raw.filter(2, 35, fir_design='firwin', verbose=False)

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
            magnitudes["mu_erd_real"] = mu_erd_magnitude   # array, one for each channel
            magnitudes["beta_erd_real"] = beta_erd_magnitude
            magnitudes["mu_ers_real"] = mu_ers_magnitude
            magnitudes["beta_ers_real"] = beta_ers_magnitude
        else:
            magnitudes["mu_erd_imagined"] = mu_erd_magnitude   # array, one for each channel
            magnitudes["beta_erd_imagined"] = beta_erd_magnitude
            magnitudes["mu_ers_imagined"] = mu_ers_magnitude
            magnitudes["beta_ers_imagined"] = beta_ers_magnitude
    
    return magnitudes

# # Feature 4: Lateralization Analysis  (!!!CUT BECAUSE NEEDS MI)
# def lateralization_analysis(subject_id, base_path):
#     subject_path = Path(base_path) / subject_id
#     filename = subject_path / f"{subject_id}R02.edf"    # R02 corresponds to eyes-closed baseline

#     channels = ['C3', 'C4', 'Cz', 'FC1', 'FC2', 'CP1', 'CP2']
    
#     raw = mne.io.read_raw_edf(filename, preload=True, verbose=False)
#     raw.rename_channels(lambda x: x.strip('.'))
    
#     # Pick motor channels if available
#     available_channels = [ch for ch in channels if ch in raw.ch_names]

#     raw.pick_channels(available_channels)

#     #Bandpass filter, 2 to 35 Hz
    # raw.filter(2, 35, fir_design='firwin', verbose=False)


# Feature 5: PSE (baseline runs) (based on Andrew's code, andrew_notebook.ipynb)
def compute_pse(subject_id, base_path):
    subject_path = Path(base_path) / subject_id
    filename = subject_path / f"{subject_id}R01.edf"    # R01 corresponds to eyes-open baseline (no occipital alpha interference)

    channels = ['C3', 'C4', 'Cz', 'FC1', 'FC2', 'CP1', 'CP2']
    
    raw = mne.io.read_raw_edf(filename, preload=True, verbose=False)
    raw.rename_channels(lambda x: x.strip('.'))
    
    # Pick motor channels if available
    available_channels = [ch for ch in channels if ch in raw.ch_names]

    raw.pick_channels(available_channels)

    #Bandpass filter, 2 to 35 Hz
    raw.filter(2, 35, fir_design='firwin', verbose=False)

    psd = raw.compute_psd(
        method = 'welch',
        fmin = 8,   # start of mu
        fmax = 30,  # end of beta (end of SMR range)
        n_fft = 512,    # 3.2 second window
        n_overlap = 256,   # 50% overlap
        verbose = False
    )

    # extract PSD
    psd_data, freqs = psd.get_data(return_freqs = True)
    
    pse_dict = {
        f"pse_{ch_name}": spectral_entropy(psd_data[i, :]) for i, ch_name in enumerate(available_channels)
    }

    pse_dict['pse_avg'] = np.mean(list(pse_dict.values()))

    return pse_dict


def spectral_entropy(psd, eps=1e-12):
    """
    PSE Helper
    """
    
    # Add small epsilon to avoid log(0)
    psd = psd + eps
    
    # Normalize PSD to make it a probability distribution
    psd = psd / psd.sum()
    
    # Compute Shannon entropy
    H = -np.sum(psd * np.log2(psd))
    
    # Normalize by maximum possible entropy (log2 of number of bins)
    H_normalized = H / np.log2(len(psd))
    
    return H_normalized


def lempel_ziv_complexity(subject_id, base_path):
    subject_path = Path(base_path) / subject_id
    filename = subject_path / f"{subject_id}R01.edf"    # R01 corresponds to eyes-open baseline

    channels = ['C3', 'C4', 'Cz', 'FC1', 'FC2', 'CP1', 'CP2']
    
    raw = mne.io.read_raw_edf(filename, preload=True, verbose=False)
    raw.rename_channels(lambda x: x.strip('.'))
    
    # Pick motor channels if available
    available_channels = [ch for ch in channels if ch in raw.ch_names]

    raw.pick_channels(available_channels)

    #Bandpass filter, 2 to 35 Hz
    raw.filter(2, 35, fir_design='firwin', verbose=False)
    data = raw.get_data()

    lzc_dict = {}

    for i, ch_name in enumerate(available_channels):
        channel_signal = data[i, :]     # time series for channel, 1D

        binary_signal = binarize_signal(channel_signal)  # above/below median

        lzc_value = lempel_ziv_complexity_calculation(binary_signal)
    
        lzc_dict[f"lzc_{ch_name}"] = lzc_value

    # Compute average LZC across all channels
    lzc_dict["lzc_avg"] = np.mean(list(lzc_dict.values()))
    
    return lzc_dict

def binarize_signal(x: np.ndarray) -> np.ndarray:
    """Convert signal to binary: 1 if above median, 0 if below."""
    return (x > np.median(x)).astype(int)


def lempel_ziv_complexity_calculation(binary_sequence: np.ndarray) -> float:
    """Compute normalized Lempel-Ziv complexity (LZ76 algorithm)."""
    # Convert binary array to string
    s = ''.join(binary_sequence.astype(str))
    n = len(s)
    
    # LZ76 algorithm
    i, k, l = 0, 1, 1
    c = 1
    
    while True:
        if s[i + k - 1] == s[l + k - 1]:
            k += 1
            if l + k > n:
                c += 1
                break
        else:
            if k > 1:
                i += 1
                k -= 1
            else:
                c += 1
                l += 1
                if l > n:
                    break
                i = 0
                k = 1
    
    # Normalize by maximum
    return c * np.log2(n) / n


def compute_theta_alpha_ratio(subject_id, base_path):
    subject_path = Path(base_path) / subject_id
    filename = subject_path / f"{subject_id}R01.edf"    # R01 corresponds to eyes-open baseline (no occipital alpha interference)

    channels = ['C3', 'C4', 'Cz', 'FC1', 'FC2', 'CP1', 'CP2']
    
    raw = mne.io.read_raw_edf(filename, preload=True, verbose=False)
    raw.rename_channels(lambda x: x.strip('.'))
    
    # Pick motor channels if available
    available_channels = [ch for ch in channels if ch in raw.ch_names]

    raw.pick_channels(available_channels)

    #Bandpass filter, 2 to 35 Hz
    raw.filter(2, 35, fir_design='firwin', verbose=False)

    # Compute theta power (4-8 Hz)
    theta_psd = raw.compute_psd(
        method='welch',
        fmin=4,
        fmax=8,
        n_fft=512,
        n_overlap=256,
        verbose=False
    )
    theta_psd_data, theta_freqs = theta_psd.get_data(return_freqs=True)
    theta_power = np.trapezoid(theta_psd_data, theta_freqs, axis=1).mean()
    
    # Compute alpha power (8-13 Hz)
    alpha_psd = raw.compute_psd(
        method='welch',
        fmin=8,
        fmax=13,
        n_fft=512,
        n_overlap=256,
        verbose=False
    )
    alpha_psd_data, alpha_freqs = alpha_psd.get_data(return_freqs=True)
    alpha_power = np.trapezoid(alpha_psd_data, alpha_freqs, axis=1).mean()
    
    # Compute theta/alpha ratio
    theta_alpha_ratio = theta_power / (alpha_power + 1e-10)  # Small epsilon to avoid division by zero
    
    return {
        "theta_power": theta_power,
        "alpha_power": alpha_power,
        "theta_alpha_ratio": theta_alpha_ratio
    }

def all_subjects_analysis(
    base_path='eeg-motor-movementimagery-dataset-1.0.0/files',
    out_csv='eeg_features.csv'):
    """Extract features for all subjects -> DataFrame -> CSV."""
    base_path = Path(base_path)

    subject_dirs = sorted([
        d for d in base_path.iterdir()
        if d.is_dir() and d.name.startswith('S') and d.name[1:].isdigit()
    ])

    # expected channel order for ERD/ERS arrays
    erd_channels = ["C3", "C4", "Cz"]

    rows = []

    for subject in tqdm(subject_dirs, desc="Computing features"):
        sid = subject.name
        row = {"subject_id": sid}

        # Feature 1: Resting Alpha Power
        # returns dict: {"rpl": ..., "resting alpha power": ..., "resting total power": ...}
        try:
            ra = resting_alpha_power(sid, base_path)
            row["rpl_alpha"] = float(ra.get("rpl_alpha")) if ra.get("rpl_alpha") is not None else None
            row["resting_alpha_power"] = float(ra.get("resting alpha power")) if ra.get("resting alpha power") is not None else None
            row["resting_total_power"] = float(ra.get("resting total power")) if ra.get("resting total power") is not None else None
        except Exception as e:
            row["rpl_alpha"] = None
            row["resting_alpha_power"] = None
            row["resting_total_power"] = None
            row["resting_alpha_error"] = str(e)
        
        # Resting Beta Power
        try:
            ra = resting_beta_power(sid, base_path)
            row["rpl_beta"] = float(ra.get("rpl_beta")) if ra.get("rpl_beta") is not None else None
            row["resting_beta_power"] = float(ra.get("resting beta power")) if ra.get("resting beta power") is not None else None
        except Exception as e:
            row["rpl_beta"] = None
            row["resting_beta_power"] = None
            row["resting_beta_error"] = str(e)
    
        # Feature 2: SMR Baseline Strength
        # returns dict: {"smr strength": ...}
        try:
            smr = baseline_smr_strength(sid, base_path)
            val = smr.get("smr strength")
            row["smr_strength"] = float(val) if val is not None else None
        except Exception as e:
            row["smr_strength"] = None
            row["smr_error"] = str(e)

        # Feature 3: ERD/ERS Magnitude (arrays per channel)
        # returns dict:
        # {
        #   "mu_erd_real": array([..C3.., ..C4.., ..Cz..]),
        #   ...
        # }
        # expand to columns:
        # mu_erd_real_C3, mu_erd_real_C4, mu_erd_real_Cz, ...
        try:
            ee = erd_ers_magnitude(sid, base_path)

            for key, vec in ee.items():
                vec = np.asarray(vec).reshape(-1)
                for i, ch in enumerate(erd_channels):
                    col = f"{key}_{ch}"
                    row[col] = float(vec[i]) if i < len(vec) else None

        except Exception as e:
            # create the expected schema with Nones if it fails
            expected_keys = [
                "mu_erd_real", "beta_erd_real", "mu_ers_real", "beta_ers_real",
                "mu_erd_imagined", "beta_erd_imagined", "mu_ers_imagined", "beta_ers_imagined",
            ]
            for k in expected_keys:
                for ch in erd_channels:
                    row[f"{k}_{ch}"] = None
            row["erd_ers_error"] = str(e)
        
        # PSE
        try:
            pse = compute_pse(sid, base_path)
            row.update(pse) 
        except Exception as e:
            row["pse_error"] = str(e)

        # LZC
        try:
            lzc = lempel_ziv_complexity(sid, base_path)
            row.update(lzc)
        except Exception as e:
            row["lzc_error"] = str(e)
        
        # Theta/Alpha Ratio
        try:
            tar = compute_theta_alpha_ratio(sid, base_path)
            row["theta_power"] = float(tar.get("theta_power")) if tar.get("theta_power") is not None else None
            row["alpha_power"] = float(tar.get("alpha_power")) if tar.get("alpha_power") is not None else None
            row["theta_alpha_ratio"] = float(tar.get("theta_alpha_ratio")) if tar.get("theta_alpha_ratio") is not None else None
        except Exception as e:
            row["theta_power"] = None
            row["alpha_power"] = None
            row["theta_alpha_ratio"] = None
            row["tar_error"] = str(e)

        rows.append(row)

    df = pd.DataFrame(rows).set_index("subject_id")
    df.to_csv(out_csv)

    print(f"\nSaved CSV to: {out_csv}")
    print(df.head())
    return df


# def all_subjects_analysis(base_path = 'eeg-motor-movementimagery-dataset-1.0.0/files'):
#     """Extract feature for all subjects"""

#     base_path = Path(base_path)

#     df = {}
    
#     # Get all subject directories
#     subject_dirs = sorted([d for d in base_path.iterdir() 
#                               if d.is_dir() and d.name.startswith('S') and d.name[1:].isdigit()])
    
#     # FEATURE 1: Resting Alpha Power
#     resting_alpha_powers = []
    
#     for subject in subject_dirs:
#         resting_alpha = resting_alpha_power(subject.name, base_path)   # subject.name ex: 'S001'
#         resting_alpha_powers.append(resting_alpha)
    
#     resting_alpha_powers = np.array(resting_alpha_powers)
    
#     for i in resting_alpha_powers:
#         print('\n')
#         print(f"subject: {i['subject_id']}")
#         print(f"alpha power: {i['alpha_power']}")
#         print(f"total_power: {i['total_power']}")
#         print(f"rpl: {i['rpl']}")


#     # FEATURE 2: SMR Baseline Strength
#     smr_arr = []
#     for subject in subject_dirs:
#             smr = baseline_smr_strength(subject.name, base_path)
#             smr_arr.append(smr)
#             print('\n')
#             print(f"subject: {subject.name}")
#             print(f"SMR Strength: {smr}")

#     # FEATURE 3: ERD/ERS Magnitude
#     erd_ers_arr = []
#     for subject in subject_dirs:
#         erd_ers = erd_ers_magnitude(subject.name, base_path)
#         erd_ers_arr.append(erd_ers)
#         print('\n')
#         print(f"subject: {subject.name}")
#         print(f"erd_ers: {erd_ers}")


            


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
