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


LAPLACIAN_SURROUNDING = {   # 4 closest electrodes to key electrode (includes above, left, right, and below)
    'C3':  ['FC3', 'C1', 'CP3', 'C5'],
    'C4':  ['FC4', 'C2', 'CP4', 'C6'],
    'Cz':  ['FCz', 'C1', 'C2', 'CPz'],
    'FC1': ['F1', 'FCz', 'FC3', 'C1'],
    'FC2': ['F2', 'FCz', 'FC4', 'C2'],
    'CP1': ['C1', 'CPz', 'CP3', 'P1'],
    'CP2': ['C2', 'CPz', 'CP4', 'P2'],
}

def apply_laplacian_filtering(raw_cleaned, channels):
    """ Applies Laplacian filtering to channels """

    raw = raw_cleaned.copy()
    
    centers = [ch for ch in channels if ch in raw.ch_names]

    all_needed = set(centers)

    for ch in centers:
        surrounds = [s for s in LAPLACIAN_SURROUNDING[ch] if s in raw.ch_names]
        all_needed.update(surrounds)
    
    raw.pick_channels(list(all_needed))

    data = raw.get_data()   # (n_channels, n_timepoints)
    ch_names = raw.ch_names

    laplacian_data = {}
    for ch in centers:
        surrounds = [s for s in LAPLACIAN_SURROUNDING[ch] if s in ch_names]

        if len(surrounds) == 0:
            laplacian_data[ch] = data[ch_names.index(ch), :]    # if no surrounding electrodes available, fall back
        else:
            center_idx = ch_names.index(ch)
            surround_idx = [ch_names.index(s) for s in surrounds]
            laplacian_data[ch] = data[center_idx, :] - np.mean(data[surround_idx, :], axis = 0)
    
    return laplacian_data, raw.info['sfreq']



def resting_alpha_power(raw_cleaned):
    """Computing resting Alpha Power for one subject"""

    raw = raw_cleaned.copy()   # copy to not mutate the original
    
    channels = ['C3', 'C4', 'Cz', 'FC1', 'FC2', 'CP1', 'CP2']
    laplacian_data, sfreq = apply_laplacian_filtering(raw, channels)

    alpha_powers = {}
    total_powers = []

    for ch, ch_signal in laplacian_data.items():
        freqs, psd = signal.welch(ch_signal,
                                  fs = sfreq,
                                  window = 'hann',  # default, standard
                                  nperseg = 2048,   # window size, large value 2048 gives better frequency distribution
                                  noverlap = 1024)  # convention is window size / 2
        
        alpha_mask = (freqs >= 8) & (freqs <= 13)
        alpha_powers[ch] = np.trapezoid(psd[alpha_mask], freqs[alpha_mask])   # integrate over frequencies

        total_mask = (freqs >= 1) & (freqs <= 40)
        total_powers.append(np.trapezoid(psd[total_mask], freqs[total_mask]))
    
    resting_state_alpha_power = np.mean(list(alpha_powers.values()))
    resting_state_total_power = np.mean(total_powers)
    rpl = resting_state_alpha_power / resting_state_total_power # relative power level
    

    # Considerations:
        # can average across just frequencies
        # can average across just channels
        # can average across both (for a single number) (current)

    # Additional, general considerations:
        # 1. resting alpha power relative to other bands
        # 2. resting alpha power with eyes open, and/or average across the two

    return {
        "rpl_alpha": rpl,   # relative power level
        "resting_alpha_power": resting_state_alpha_power, 
        "resting_total_power": resting_state_total_power,
        "alpha_power_c3": alpha_powers['C3'],
        "alpha_power_c4": alpha_powers['C4'],
        "alpha_asymmetry": np.log(alpha_powers['C4']) - np.log(alpha_powers['C3'])
    }


def alpha_power_variability(raw_cleaned):
    """ compute variability of alpha power across sliding windows at C3 and C4 electrodes for one subject """
    
    raw = raw_cleaned.copy()
    channels = ['C3', 'C4']
    laplacian_data, sfreq = apply_laplacian_filtering(raw, channels)

    # 2 second windows with 1 second overlaps means there is always clean overlap
    # convert time durations into array indices
    window_sec = 2.0
    overlap_sec = 1.0
    window_samples = int(window_sec * sfreq)
    step_samples = int((window_sec - overlap_sec) * sfreq)

    result = {}
    for ch, ch_signal in laplacian_data.items():
        alpha_powers = []
        start = 0
        while start + window_samples <= len(ch_signal):
            segment = ch_signal[start:start + window_samples]
            freqs, psd = signal.welch(segment,
                                      fs = sfreq,
                                      window = 'hann',
                                      nperseg = len(segment),
                                      noverlap = 0)
            alpha_mask= (freqs >= 8) & (freqs <= 13)
            alpha_powers.append(np.trapezoid(psd[alpha_mask], freqs[alpha_mask]))
            start += step_samples
        
        alpha_powers = np.array(alpha_powers)
        result[f"alpha_var_{ch}"] = np.std(alpha_powers)    # raw std dev of alpha power across windows
        result[f"alpha_cv_{ch}"] = np.std(alpha_powers) / (np.mean(alpha_powers) + 1e-10)   # coefficient of variation (normalized)
    
    return result


def interhemispheric_coherence(raw_cleaned):
    """ Compute magnitude squared coherence between C3 and C4 in mu and beta bands for one subject"""

    raw = raw_cleaned.copy()
    channels = ['C3', 'C4']
    laplacian_data, sfreq = apply_laplacian_filtering(raw, channels)

    c3_signal = laplacian_data['C3']
    c4_signal = laplacian_data['C4']

    freqs, coh = signal.coherence(c3_signal, 
                                  c4_signal,
                                  fs = sfreq, 
                                  window = 'hann',
                                  nperseg = 2048,
                                  noverlap = 1024)
    mu_mask = (freqs >= 8) & (freqs <= 13)
    beta_mask = (freqs >= 13) & (freqs <= 30)
    low_beta_mask = (freqs >= 13) & (freqs <= 20)
    upper_beta_mask = (freqs >= 20) & (freqs <= 30)

    return {
        "coherence_mu": np.mean(coh[mu_mask]),
        "coherence_beta": np.mean(coh[beta_mask]),
        "coherence_low_beta": np.mean(coh[low_beta_mask]),
        "coherence_upper_beta": np.mean(coh[upper_beta_mask])
    }


def resting_lower_beta_power(raw_cleaned):
    """Computing resting lower Beta Power for one subject (13-20 Hz)"""

    raw = raw_cleaned.copy()   # copy to not mutate the original
    
    channels = ['C3', 'C4', 'Cz', 'FC1', 'FC2', 'CP1', 'CP2']
    laplacian_data, sfreq = apply_laplacian_filtering(raw, channels)

    beta_powers = []
    total_powers = []

    for ch, ch_signal in laplacian_data.items():
        freqs, psd = signal.welch(ch_signal,
                                  fs = sfreq,
                                  window = 'hann',  # default, standard
                                  nperseg = 2048,   # window size, large value 2048 gives better frequency distribution
                                  noverlap = 1024)  # convention is window size / 2
        
        beta_mask = (freqs >= 13) & (freqs <= 20)
        beta_powers.append(np.trapezoid(psd[beta_mask], freqs[beta_mask]))   # integrate over frequencies

        total_mask = (freqs >= 1) & (freqs <= 40)
        total_powers.append(np.trapezoid(psd[total_mask], freqs[total_mask]))
    
    resting_state_beta_power = np.mean(beta_powers)
    resting_state_total_power = np.mean(total_powers)
    rpl = resting_state_beta_power / resting_state_total_power # relative power level

    return {
        "rpl_lower_beta": rpl, 
        "resting_lower_beta_power": resting_state_beta_power, 
    }


def resting_upper_beta_power(raw_cleaned):
    """Computing resting upper Beta Power for one subject (20-30 Hz)"""

    raw = raw_cleaned.copy()   # copy to not mutate the original
    
    channels = ['C3', 'C4', 'Cz', 'FC1', 'FC2', 'CP1', 'CP2']
    laplacian_data, sfreq = apply_laplacian_filtering(raw, channels)

    beta_powers = []
    total_powers = []

    for ch, ch_signal in laplacian_data.items():
        freqs, psd = signal.welch(ch_signal,
                                  fs = sfreq,
                                  window = 'hann',  # default, standard
                                  nperseg = 2048,   # window size, large value 2048 gives better frequency distribution
                                  noverlap = 1024)  # convention is window size / 2
        
        beta_mask = (freqs >= 20) & (freqs <= 30)
        beta_powers.append(np.trapezoid(psd[beta_mask], freqs[beta_mask]))   # integrate over frequencies

        total_mask = (freqs >= 1) & (freqs <= 40)
        total_powers.append(np.trapezoid(psd[total_mask], freqs[total_mask]))
    
    resting_state_beta_power = np.mean(beta_powers)
    resting_state_total_power = np.mean(total_powers)
    rpl = resting_state_beta_power / resting_state_total_power # relative power level

    return {
        "rpl_upper_beta": rpl, 
        "resting_upper_beta_power": resting_state_beta_power, 
    }


# Second feature: SMR baseline strength
def baseline_smr_strength(raw_cleaned):
    """Computing SMR Baseline Strength for one subject"""
    raw = raw_cleaned.copy()

    channels = ['C3', 'C4']
    
    # Laplacian Filtering - for reducing noise and volume, and isolating local SMR
    c3_surrounding = ['FC3', 'C1', 'CP3', 'C5']
    c4_surrounding = ['FC4', 'CP4', 'C2', 'C6']
    
    # Pick motor channels if available
    available_channels = [ch for ch in channels if ch in raw.ch_names]
    available_c3 = [ch for ch in c3_surrounding if ch in raw.ch_names]
    available_c4 = [ch for ch in c4_surrounding if ch in raw.ch_names]

    all_channels_needed = available_channels + available_c3 + available_c4

    raw.pick_channels(all_channels_needed)


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
    [2.0,  150, 50, 100, 100, 13, 30, 10, 10]   # Upper bounds
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

    # re unpack with distinct names for returning
    lambda_c3, k1_c3, k2_c3, k3_c3, k4_c3, mu1_c3, mu2_c3, sig1_c3, sig2_c3 = optimal_params_c3
    lambda_c4, k1_c4, k2_c4, k3_c4, k4_c4, mu1_c4, mu2_c4, sig1_c4, sig2_c4 = optimal_params_c4

    # if IAF right near boundary, no clear peak
    if (mu1_c3 >= 8.01) and (mu1_c3 <= 12.99):
        iaf_c3 = mu1_c3
    else:
        iaf_c3 = np.nan
    
    if (mu1_c4 >= 8.01) and (mu1_c4 <= 12.99):
        iaf_c4 = mu1_c4
    else:
        iaf_c4 = np.nan

    return {"smr_strength": smr_strength,
            "IAF_c3": iaf_c3,     # individual alpha frequency
            "alpha_peak_amp_c3": k3_c3,     # alpha peak height
            "beta_peak_amp_c3": k4_c3,      # beta peak height
            "beta_center_freq_c3": mu2_c3,
            "aperiodic_exp_c3": lambda_c3,      # 1/f slope
            "IAF_c4": iaf_c4,
            "alpha_peak_amp_c4": k3_c4,
            "beta_peak_amp_c4": k4_c4,
            "beta_center_freq_c4": mu2_c4,
            "aperiodic_exp_c4": lambda_c4
            }

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


# Feature 5: PSE (baseline runs) (based on Andrew's code, andrew_notebook.ipynb)
def compute_pse(raw_cleaned):
    """ Computing Power Spectral Entropy (PSE) for one subject """
    
    raw = raw_cleaned.copy()
    channels = ['C3', 'C4', 'Cz', 'FC1', 'FC2', 'CP1', 'CP2']
    laplacian_data, sfreq = apply_laplacian_filtering(raw, channels)

    pse_dict = {}
    for ch, ch_signal in laplacian_data.items():
        freqs, psd = signal.welch(ch_signal,
                                  fs = sfreq,
                                  window = 'hann',
                                  nperseg = 512,
                                  noverlap= 256)
        smr_mask = (freqs >= 8) & (freqs <= 30)
        pse_dict[f"pse_{ch}"] = spectral_entropy(psd[smr_mask])
    
    pse_dict['pse_avg'] = np.mean(list(pse_dict.values()))
    return pse_dict


def spectral_entropy(psd, eps=1e-12):
    """ PSE Helper """
    
    # Add small epsilon to avoid log(0)
    psd = psd + eps
    
    # Normalize PSD to make it a probability distribution
    psd = psd / psd.sum()
    
    # Compute Shannon entropy
    H = -np.sum(psd * np.log2(psd))
    
    # Normalize by maximum possible entropy (log2 of number of bins)
    H_normalized = H / np.log2(len(psd))
    
    return H_normalized


def lempel_ziv_complexity(raw_cleaned):
    """ Computing Lempel-Ziv Complexity (LZC) for one subject"""
    
    raw = raw_cleaned.copy()
    channels = ['C3', 'C4', 'Cz', 'FC1', 'FC2', 'CP1', 'CP2']

    available = [ch for ch in channels if ch in raw.ch_names]
    raw.pick_channels(available)

    data = raw.get_data()
    ch_names = raw.ch_names

    lzc_dict = {}
    for ch in available:
        ch_signal = data[ch_names.index(ch), :]
        binary_signal = binarize_signal(ch_signal)
        lzc_dict[f"lzc_{ch}"] = lempel_ziv_complexity_calculation(binary_signal)
    
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


def compute_theta_alpha_ratio(raw_cleaned):
    """ compute theta/alpha ratio (TAR) for one subject """
    raw = raw_cleaned.copy()
    channels = ['C3', 'C4', 'Cz', 'FC1', 'FC2', 'CP1', 'CP2']

    laplacian_data, sfreq = apply_laplacian_filtering(raw, channels)

    theta_powers  = []
    alpha_powers = []

    for ch, ch_signal in laplacian_data.items():
        freqs, psd = signal.welch(ch_signal,
                                  fs = sfreq,
                                  window = 'hann',
                                  nperseg = 512,
                                  noverlap = 256)
        
        theta_mask = (freqs >= 4) & (freqs <= 8)
        theta_powers.append(np.trapezoid(psd[theta_mask], freqs[theta_mask]))

        alpha_mask = (freqs >= 8) & (freqs <= 13)
        alpha_powers.append(np.trapezoid(psd[alpha_mask], freqs[alpha_mask]))
    
    theta_power = np.mean(theta_powers)
    alpha_power = np.mean(alpha_powers)
    tar = theta_power / (alpha_power + 1e-10)   # prevent from ballooning

    return {
        "theta_power": theta_power,
        "alpha_power": alpha_power,
        "tar": tar
    }

    

def preprocess(subject_id, base_path, run_id = 'R01'):
    """Loads raw data, applies ICA artifact removal, returns cleaned Raw object"""
    
    subject_path = Path(base_path)/subject_id
    filename = subject_path / f"{subject_id}{run_id}.edf"

    # loading raw data
    raw = mne.io.read_raw_edf(filename, preload = True, verbose = False)
    raw.rename_channels(lambda x: x.strip('.'))

    # 10-10 montage: maps from channel names to 3D coordinates
    montage = mne.channels.make_standard_montage('standard_1005')
    raw.set_montage(montage, on_missing = 'ignore')

    # re referencing with common average reference
    raw.set_eeg_reference('average', verbose = False)

    # Bandpass filter: using 1-40 instead of 2-35 like in some papers to improve ICA
    raw.filter(1, 40, fir_design = 'firwin', verbose = False)

    # ICA
    ica = mne.preprocessing.ICA(
        n_components = 20,   # conventional starting point to capture artifacts
        method = 'picard',  # to help with convergence
        max_iter = 'auto'
    )
    ica.fit(raw, verbose = False)

    # we don't have EOG; using frontal channels as proxy (Fp1, Fp2, AF7 and AF8)
    eog_indices = []

    # Vertical EOG (blinks) use Fp1 and Fp2
    if 'Fp1' in raw.ch_names:
        blink_idx, blink_scores = ica.find_bads_eog(
            raw, ch_name = 'Fp1', verbose = False
        )
        eog_indices.extend(blink_idx)

    # Horizontal EOG use AF7/8
    for ch in ['AF7', 'AF8']:
        if ch in raw.ch_names:
            horiz_idx, horiz_scores = ica.find_bads_eog(
                raw, ch_name = ch, verbose = False
            )
            eog_indices.extend(horiz_idx)
            break   # only need one horizontal proxy

    
    # remove duplicates
    ica.exclude = list(set(eog_indices))

    # Apply ICA
    ica.apply(raw, verbose = False)

    return raw



def all_subjects_analysis(
    base_path='eeg-motor-movementimagery-dataset-1.0.0/files',
    out_csv='eeg_features.csv'):
    """Extract features for all subjects -> DataFrame -> CSV."""
    
    base_path = Path(base_path)
    subject_dirs = sorted([
        d for d in base_path.iterdir()
        if d.is_dir() and d.name.startswith('S') and d.name[1:].isdigit()
    ])

    rows = []
    for subject in tqdm(subject_dirs, desc="Computing features"):
        sid = subject.name
        row = {"subject_id": sid}

        try:
            raw_r01 = preprocess(sid, base_path, run_id = 'R01')    # eyes open baseline
            raw_r02 = preprocess(sid, base_path, run_id = 'R02')    # eyes closed baseline

            # pass data to feature functions

            ra = resting_alpha_power(raw_r02)   # use eyes closed
            row.update(ra)

            rlb = resting_lower_beta_power(raw_r02)    # use eyes closed
            row.update(rlb)

            rub = resting_upper_beta_power(raw_r02) # use eyes closed
            row.update(rub)

            smr = baseline_smr_strength(raw_r01)    # use eyes open
            row.update(smr)

            apv = alpha_power_variability(raw_r01)    # use eyes open
            row.update(apv)

            ihc = interhemispheric_coherence(raw_r01)  # use eyes open
            row.update(ihc)

            pse = compute_pse(raw_r01)    # use eyes open
            row.update(pse)

            lzc = lempel_ziv_complexity(raw_r01)    # use eyes open
            row.update(lzc)

            tar = compute_theta_alpha_ratio(raw_r01)    # use eyes open
            row.update(tar)

        except Exception as e:
            row['preprocessing_error'] = str(e)
        
        rows.append(row)
    
    df = pd.DataFrame(rows).set_index("subject_id")
    df.to_csv(out_csv)
    return df

    

if __name__ == "__main__":
    all_subjects_analysis()