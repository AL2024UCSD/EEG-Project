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


def resting_alpha_power(raw_cleaned):
    """Computing resting Alpha Power for one subject"""

    raw = raw_cleaned.copy()   # copy so picking doesn't mutate the original
    
    channels = ['C3', 'C4', 'Cz', 'FC1', 'FC2', 'CP1', 'CP2']
    available_channels = [ch for ch in channels if ch in raw.ch_names]
    raw.pick_channels(available_channels)
    
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
        fmin = 1,
        fmax = 40,
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


def resting_beta_power(raw_cleaned):
    """Computing resting Beta Power for one subject"""

    raw = raw_cleaned.copy()
    channels = ['C3', 'C4', 'Cz', 'FC1', 'FC2', 'CP1', 'CP2']
    available_channels = [ch for ch in channels if ch in raw.ch_names]
    raw.pick_channels(available_channels)


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
        fmin = 1,
        fmax = 40,
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
    return {"smr strength": smr_strength}
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
    
    raw = raw_cleaned.copy()
    channels = ['C3', 'C4', 'Cz', 'FC1', 'FC2', 'CP1', 'CP2']
    available_channels = [ch for ch in channels if ch in raw.ch_names]
    raw.pick_channels(available_channels)

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


def lempel_ziv_complexity(raw_cleaned):
     
    raw = raw_cleaned.copy()
    channels = ['C3', 'C4', 'Cz', 'FC1', 'FC2', 'CP1', 'CP2']
    available_channels = [ch for ch in channels if ch in raw.ch_names]
    raw.pick_channels(available_channels)

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


def compute_theta_alpha_ratio(raw_cleaned):
    
    raw = raw_cleaned.copy()
    channels = ['C3', 'C4', 'Cz', 'FC1', 'FC2', 'CP1', 'CP2']
    available_channels = [ch for ch in channels if ch in raw.ch_names]
    raw.pick_channels(available_channels)

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

def preprocess(subject_id, base_path, run_id = 'R01'):
    """Loads raw data, applies ICA artifact removal, returns cleaned Raw object"""
    
    subject_path = Path(base_path)/subject_id
    filename = subject_path / f"{subject_id}{run_id}.edf"

    # loading raw data
    raw = mne.io.read_raw_edf(filename, preload = True, verbose = False)
    raw.rename_channels(lambda x: x.strip('.'))

    # 10-10 montage: maps from channel names to 3D coordinates
    montage = mne.channels.make_standard_montage('standard_1010')
    raw.set_montage(montage, on_missing = 'ignore')

    # re referencing with common average reference
    raw.set_eeg_reference('average', verbose = False)

    # Bandpass filter: using 1-40 instead of 2-35 like in some papers to improve ICA
    raw.filter(1, 40, fir_design = 'firwin', verbose = False)

    # ICA
    ica = mne.preprocessing.ICA(
        n_components = 20,   # conventional starting point to capture artifacts
        method = 'fastica',
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

            rb = resting_beta_power(raw_r02)    # use eyes closed
            row.update(rb)

            smr = baseline_smr_strength(raw_r01)    # use eyes open
            row.update(smr)

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