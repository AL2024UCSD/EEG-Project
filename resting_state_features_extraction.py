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
import mne
from typing import Dict


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
    """
    Apply surface Laplacian filtering to a set of EEG channels.

    For each requested channel, subtracts the mean of its surrounding electrodes
    (as defined in LAPLACIAN_SURROUNDING) from the center electrode signal. This
    reduces volume conduction and improves spatial specificity. If no surrounding
    electrodes are available for a channel, the raw signal is returned as-is.

    Parameters:
    raw_cleaned : mne.io.Raw
        Preprocessed MNE Raw object containing EEG data.
    channels : list of str
        List of center electrode names to apply Laplacian filtering to.

    Returns:
    laplacian_data : dict
        Dictionary mapping channel name to its Laplacian-filtered signal
        as a 1D numpy array (n_timepoints,).
    sfreq : float
        Sampling frequency of the data in Hz.
    """

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
    """
    Compute resting-state alpha power features for one subject.

    Applies Laplacian filtering to motor channels, then computes Welch PSD
    for each channel. Extracts absolute alpha (8-13 Hz) and theta (4-8 Hz)
    power, total broadband power (1-40 Hz), and derives relative alpha power
    (RPL) and theta/alpha ratio (TAR). Also returns per-hemisphere alpha power
    at C3/C4 and their log-ratio asymmetry score.

    Parameters:
    raw_cleaned : mne.io.Raw
        Preprocessed MNE Raw object. Intended for use with eyes-closed baseline (R02).

    Returns:
    dict with keys:
        rpl_alpha : float
            Alpha power as a fraction of total broadband power (1-40 Hz).
        tar : float
            Theta-to-alpha power ratio.
        resting_alpha_power : float
            Mean absolute alpha power across channels (µV²/Hz).
        resting_total_power : float
            Mean total broadband power across channels (µV²/Hz).
        alpha_power_c3 : float
            Absolute alpha power at C3.
        alpha_power_c4 : float
            Absolute alpha power at C4.
        alpha_asymmetry : float
            Log(alpha_C4) - log(alpha_C3). Positive values indicate right-hemisphere dominance.
    """

    raw = raw_cleaned.copy()   # copy to not mutate the original
    
    channels = ['C3', 'C4', 'Cz', 'FC1', 'FC2', 'CP1', 'CP2']
    laplacian_data, sfreq = apply_laplacian_filtering(raw, channels)

    alpha_powers = {}
    theta_powers = {}
    total_powers = []

    for ch, ch_signal in laplacian_data.items():
        freqs, psd = signal.welch(ch_signal,
                                  fs = sfreq,
                                  window = 'hann',  # default, standard
                                  nperseg = 2048,   # window size, large value 2048 gives better frequency distribution
                                  noverlap = 1024)  # convention is window size / 2
        
        alpha_mask = (freqs >= 8) & (freqs <= 13)
        alpha_powers[ch] = np.trapezoid(psd[alpha_mask], freqs[alpha_mask])   # integrate over frequencies

        theta_mask = (freqs >= 4) & (freqs <= 8)
        theta_powers[ch] = np.trapezoid(psd[theta_mask], freqs[theta_mask])   # integrate over frequencies

        total_mask = (freqs >= 1) & (freqs <= 40)
        total_powers.append(np.trapezoid(psd[total_mask], freqs[total_mask]))
    
    resting_state_alpha_power = np.mean(list(alpha_powers.values()))
    resting_state_theta_power = np.mean(list(theta_powers.values()))
    resting_state_total_power = np.mean(total_powers)
    tar = resting_state_theta_power / resting_state_alpha_power
    rpl = resting_state_alpha_power / resting_state_total_power # relative power level

    return {
        "rpl_alpha": rpl,   # relative power level
        "tar": tar,         # theta/alpha ratio
        "resting_alpha_power": resting_state_alpha_power, 
        "resting_total_power": resting_state_total_power,
        "alpha_power_c3": alpha_powers['C3'],
        "alpha_power_c4": alpha_powers['C4'],
        "alpha_asymmetry": np.log(alpha_powers['C4']) - np.log(alpha_powers['C3'])
    }


def alpha_power_variability(raw_cleaned):
    """
    Compute variability of alpha power over time at C3 and C4 for one subject.

    Applies Laplacian filtering to C3 and C4, then slides a 2-second window
    (1-second overlap) over each signal, computing alpha band power (8-13 Hz)
    in each window via Welch PSD. Returns both the raw standard deviation and
    the coefficient of variation (CV) of alpha power across windows, separately
    for each electrode.

    Parameters:
    raw_cleaned : mne.io.Raw
        Preprocessed MNE Raw object. Intended for use with eyes-open baseline (R01).

    Returns:
    dict with keys:
        alpha_var_C3 : float
            Standard deviation of alpha power across windows at C3.
        alpha_cv_C3 : float
            Coefficient of variation (std / mean) of alpha power at C3.
        alpha_var_C4 : float
            Standard deviation of alpha power across windows at C4.
        alpha_cv_C4 : float
            Coefficient of variation of alpha power at C4.
    """
    
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
    """
    Compute magnitude-squared coherence between C3 and C4 for one subject.

    Applies Laplacian filtering to C3 and C4, then computes coherence across
    the mu (8-13 Hz), full beta (13-30 Hz), low beta (13-20 Hz), and upper
    beta (20-30 Hz) bands using Welch's method.

    Parameters:
    raw_cleaned : mne.io.Raw
        Preprocessed MNE Raw object. Intended for use with eyes-open baseline (R01).

    Returns:
    dict with keys:
        coherence_mu : float
            Mean coherence in the mu/alpha band (8-13 Hz).
        coherence_beta : float
            Mean coherence across the full beta band (13-30 Hz).
        coherence_low_beta : float
            Mean coherence in the low beta band (13-20 Hz).
        coherence_upper_beta : float
            Mean coherence in the upper beta band (20-30 Hz).
    """

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
    """
    Compute resting-state lower beta power features for one subject (13-20 Hz).

    Applies Laplacian filtering to motor channels, computes Welch PSD for each,
    and returns mean absolute lower beta power and its fraction of total broadband
    power (1-40 Hz).

    Parameters:
    raw_cleaned : mne.io.Raw
        Preprocessed MNE Raw object. Intended for use with eyes-closed baseline (R02).

    Returns:
    dict with keys:
        rpl_lower_beta : float
            Lower beta power as a fraction of total broadband power (1-40 Hz).
        resting_lower_beta_power : float
            Mean absolute lower beta power across channels (µV²/Hz).
    """

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
    """
    Compute resting-state upper beta power features for one subject (20-30 Hz).

    Applies Laplacian filtering to motor channels, computes Welch PSD for each,
    and returns mean absolute upper beta power and its fraction of total broadband
    power (1-40 Hz).

    Parameters:
    raw_cleaned : mne.io.Raw
        Preprocessed MNE Raw object. Intended for use with eyes-closed baseline (R02).

    Returns:
    dict with keys:
        rpl_upper_beta : float
            Upper beta power as a fraction of total broadband power (1-40 Hz).
        resting_upper_beta_power : float
            Mean absolute upper beta power across channels (µV²/Hz).
    """

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


# SMR baseline strength

def fit_with_multistart(freqs, psd_subset, bounds):
    """
    Fit freq_curve to PSD data using multiple lambda starting points.

    Tries 6 evenly-spaced lambda initializations (0.25 to 2.0) and returns
    the parameter set that achieves the lowest sum-of-squared residuals.
    This improves robustness over single-start fitting, which is sensitive
    to the choice of initial parameters.

    Parameters:
    freqs : np.ndarray
        Array of frequency values (Hz), shape (n_freqs,).
    psd_subset : np.ndarray
        PSD values in dB at each frequency in freqs, shape (n_freqs,).
    bounds : tuple of (list, list)
        Lower and upper bounds for each parameter in freq_curve, passed
        directly to scipy.optimize.curve_fit.

    Returns:
    best_params : np.ndarray
        Optimal parameters for freq_curve with the lowest residual across
        all starting points, shape (9,).

    Raises:
    RuntimeError
        If all 6 starting points fail to converge.
    """
    best_params = None
    best_residual = np.inf
    
    lambda_inits = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
    
    for lambda_init in lambda_inits:
        try:
            k1_guess = np.clip(psd_subset[-1], -20, 20)
            p0 = [lambda_init, k1_guess, 10.0, 5.0, 5.0, 10.0, 23.0, 3.0, 3.0]
            
            params, _ = optimize.curve_fit(
                freq_curve,
                freqs,
                psd_subset,
                p0=p0,
                bounds=bounds,
                maxfev=5000
            )
            
            residual = np.sum((freq_curve(freqs, *params) - psd_subset) ** 2)
            if residual < best_residual:
                best_residual = residual
                best_params = params
                
        except Exception:
            continue
    
    if best_params is None:
        raise RuntimeError("All starting points failed to converge")
    
    return best_params

def baseline_smr_strength(raw_cleaned):
    """
    Compute SMR baseline strength and spectral peak features for one subject.

    Applies Laplacian filtering to C3 and C4, computes Welch PSD (converted to
    dB), and fits a parametric model (1/f noise floor + two Gaussians for alpha
    and beta peaks) using multistart curve fitting. SMR strength is defined as
    the maximum peak height above the fitted noise floor, averaged across C3 and
    C4. Also extracts individual alpha frequency (IAF), alpha/beta peak amplitudes,
    beta center frequency, and aperiodic exponent per electrode.

    IAF is set to NaN if the fitted alpha peak center is at the band boundary
    (indicating no clear peak) or if the peak amplitude is below threshold.

    Parameters:
    raw_cleaned : mne.io.Raw
        Preprocessed MNE Raw object. Intended for use with eyes-open baseline (R01).

    Returns:
    dict with keys:
        smr_strength : float
            Mean peak height above noise floor across C3 and C4 (dB).
        IAF_c3 : float or NaN
            Individual alpha frequency at C3 (Hz). NaN if no clear peak detected.
        alpha_peak_amp_c3 : float
            Alpha Gaussian amplitude at C3 (dB above noise).
        beta_peak_amp_c3 : float
            Beta Gaussian amplitude at C3 (dB above noise).
        beta_center_freq_c3 : float
            Center frequency of fitted beta peak at C3 (Hz).
        aperiodic_exp_c3 : float
            Aperiodic (1/f) exponent at C3.
        IAF_c4, alpha_peak_amp_c4, beta_peak_amp_c4,
        beta_center_freq_c4, aperiodic_exp_c4 : float
            Same quantities for C4.
    """
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
    [0.1, -150, 0, 0, 0, 8, 13, 1, 1],        # Lower bounds
    [2.0,  150, 50, 100, 100, 13, 30, 10, 10]   # Upper bounds
    )

    optimal_params_c3 = fit_with_multistart(freqs, psd_c3_subset, bounds)

    lambda_fit, k1, k2, k3, k4, mu1, mu2, sig1, sig2 = optimal_params_c3
    g_total_c3 = freq_curve(freqs, *optimal_params_c3)  # full fitted curve in dB
    g1_c3 = k1 + k2 / (freqs ** lambda_fit)     # noise component in dB
    g2_c3 = g_total_c3 - g1_c3      # g2 = total - noise in dB
    smr_strength_c3 = np.max(g2_c3)     # maximum peak height above noise

    # Fitting 1/freqs gaussian model for C4
    optimal_params_c4 = fit_with_multistart(freqs, psd_c4_subset, bounds)

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

    min_alpha_peak_amp = 1.0

    if (mu1_c3 >= 8.01) and (mu1_c3 <= 12.99) and (k3_c3 >= min_alpha_peak_amp):
        iaf_c3 = mu1_c3
    else:
        iaf_c3 = np.nan
    
    if (mu1_c4 >= 8.01) and (mu1_c4 <= 12.99) and (k3_c4 >= min_alpha_peak_amp):
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
    """
    Parametric model of EEG power spectral density in dB.

    Models the PSD as the sum of an aperiodic 1/f noise floor (g1) and two
    Gaussian peaks representing the alpha/mu and beta rhythms (g2):

        g1(f) = k1 + k2 / f^lambda_val
        g2(f) = k3 * exp(-0.5 * ((f - mu1) / sig1)^2)
              + k4 * exp(-0.5 * ((f - mu2) / sig2)^2)
        output = g1(f) + g2(f)

    Parameters:
    freqs : np.ndarray
        Array of frequency values (Hz).
    lambda_val : float
        Aperiodic exponent controlling the steepness of the 1/f slope.
    k1 : float
        Baseline offset; shifts the entire noise floor up or down.
    k2 : float
        Scale factor for the 1/f component.
    k3 : float
        Amplitude of the first Gaussian (alpha/mu peak).
    k4 : float
        Amplitude of the second Gaussian (beta peak).
    mu1 : float
        Center frequency of the alpha/mu Gaussian (Hz).
    mu2 : float
        Center frequency of the beta Gaussian (Hz).
    sig1 : float
        Width (standard deviation) of the alpha/mu Gaussian (Hz).
    sig2 : float
        Width (standard deviation) of the beta Gaussian (Hz).

    Returns:
    np.ndarray
        Modeled PSD values at each frequency in freqs (dB).
    """
    

    g1 = k1 + k2/(freqs**lambda_val)

    gaussian1 = np.exp(-0.5 * ((freqs - mu1) / sig1) ** 2)
    gaussian2 = np.exp(-0.5 * ((freqs - mu2) / sig2) ** 2)
    g2 = k3 * gaussian1 + k4*gaussian2

    return g1 + g2


# Feature 5: PSE (baseline runs)
def compute_pse(raw_cleaned):
    """
    Compute Power Spectral Entropy (PSE) for one subject.

    Applies Laplacian filtering to motor channels, computes Welch PSD for each,
    and calculates normalized Shannon entropy over the SMR band (8-30 Hz).
    Lower PSE indicates a more peaked, organised spectrum; higher PSE indicates
    a flatter, more random distribution of power across frequencies.

    Parameters:
    raw_cleaned : mne.io.Raw
        Preprocessed MNE Raw object. Intended for use with eyes-open baseline (R01).

    Returns:
    dict with keys:
        pse_C3, pse_C4, pse_Cz, pse_FC1, pse_FC2, pse_CP1, pse_CP2 : float
            Normalized spectral entropy (0-1) in the SMR band for each channel.
        pse_avg : float
            Mean PSE across all channels.
    """
    
    raw = raw_cleaned.copy()
    channels = ['C3', 'C4', 'Cz', 'FC1', 'FC2', 'CP1', 'CP2']
    laplacian_data, sfreq = apply_laplacian_filtering(raw, channels)
    
    pse_dict = {}
    for ch, ch_signal in laplacian_data.items():
        freqs, psd = signal.welch(ch_signal,
                                  fs=sfreq,
                                  window='hann',
                                  nperseg=512,
                                  noverlap=256)
        smr_mask = (freqs >= 8) & (freqs <= 30)
        pse_dict[f"pse_{ch}"] = spectral_entropy(psd[smr_mask])
    
    pse_dict['pse_avg'] = np.mean(list(pse_dict.values()))
    return pse_dict


def spectral_entropy(psd, eps=1e-12):
    """
    Compute normalized Shannon spectral entropy of a PSD array.

    Treats the PSD as a probability distribution over frequency bins (after
    normalization) and computes Shannon entropy, normalized by the maximum
    possible entropy for that number of bins (log2 of bin count). The result
    is bounded in [0, 1], where 1 is a perfectly flat spectrum and 0 is all
    power concentrated in a single bin.

    Parameters:
    psd : np.ndarray
        Power spectral density values for a frequency band, shape (n_freqs,).
    eps : float, optional
        Small constant added to PSD values to avoid log(0). Default is 1e-12.

    Returns:
    float
        Normalized spectral entropy in [0, 1].
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
    """
    Compute Lempel-Ziv Complexity (LZC) for one subject.

    For each motor channel, binarizes the signal relative to its median and
    applies the LZ76 algorithm to measure signal complexity. Higher LZC
    indicates a more random, less structured signal; lower LZC indicates
    more regularity or periodicity. Does not apply Laplacian filtering —
    uses raw channel signals directly.

    Parameters:
    raw_cleaned : mne.io.Raw
        Preprocessed MNE Raw object. Intended for use with eyes-open baseline (R01).

    Returns:
    dict with keys:
        lzc_C3, lzc_C4, lzc_Cz, lzc_FC1, lzc_FC2, lzc_CP1, lzc_CP2 : float
            Normalized LZC for each channel.
        lzc_avg : float
            Mean LZC across all channels.
    """
    
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
    """
    Convert a continuous signal to a binary sequence using the median as threshold.

    Parameters:
    x : np.ndarray
        1D array of signal values.

    Returns:
    np.ndarray
        Binary array of the same shape as x: 1 where x > median(x), 0 elsewhere.
    """
    return (x > np.median(x)).astype(int)


def lempel_ziv_complexity_calculation(binary_sequence: np.ndarray) -> float:
    """
    Compute normalized Lempel-Ziv complexity using the LZ76 algorithm.

    Converts the binary sequence to a string and counts the number of distinct
    substrings encountered during a left-to-right scan (LZ76 complexity measure).
    Normalizes by n / log2(n) so that the result is comparable across sequences
    of different lengths.

    Parameters:
    binary_sequence : np.ndarray
        1D binary array (values 0 or 1), typically produced by binarize_signal.

    Returns:
    float
        Normalized LZC value. Higher values indicate greater complexity.
    """
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

    

def preprocess(subject_id, base_path, run_id = 'R01'):
    """
    Load a raw EEG file and apply standard preprocessing for one subject and run.

    Steps:
        1. Load the .edf file and strip trailing dots from channel names.
        2. Apply the standard 10-05 montage.
        3. Re-reference to common average.
        4. Bandpass filter 1-40 Hz (firwin).
        5. Fit ICA (Picard, 20 components) and auto-identify ocular artifact
           components using frontal channels as EOG proxies (Fp1 for blinks,
           AF7/AF8 for horizontal eye movements).
        6. Remove identified artifact components and return the cleaned Raw object.

    Parameters:
    subject_id : str
        Subject folder name, e.g. 'S001'.
    base_path : str or Path
        Root directory containing per-subject folders.
    run_id : str, optional
        Run identifier appended to the subject ID to form the filename,
        e.g. 'R01' (eyes open) or 'R02' (eyes closed). Default is 'R01'.

    Returns:
    mne.io.Raw
        ICA-cleaned Raw object with data preloaded.
    """
    
    subject_path = Path(base_path)/subject_id
    filename = subject_path / f"{subject_id}{run_id}.edf"

    # loading raw data
    raw = mne.io.read_raw_edf(filename, preload = True, verbose = False)
    raw.rename_channels(lambda x: x.strip('.'))

    # 10-05 montage: maps from channel names to 3D coordinates
    montage = mne.channels.make_standard_montage('standard_1005')
    raw.set_montage(montage, on_missing = 'ignore')

    # re referencing to common average 
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
    out_csv='resting_state_features.csv'):
    """
    Extract resting-state EEG features for all subjects and save to CSV.

    Iterates over all subject directories under base_path (folders matching
    pattern S[digits]), preprocesses eyes-open (R01) and eyes-closed (R02)
    baseline runs for each subject, and computes all feature sets. Results
    are collected into a DataFrame indexed by subject_id and written to CSV.
    If preprocessing or feature extraction fails for a subject, the error
    message is stored in a 'preprocessing_error' column and processing
    continues with the next subject.

    Feature assignment by condition:
        - Eyes closed (R02): resting_alpha_power, resting_lower_beta_power,
          resting_upper_beta_power
        - Eyes open (R01): baseline_smr_strength, alpha_power_variability,
          interhemispheric_coherence, compute_pse, lempel_ziv_complexity

    Parameters:
    base_path : str or Path, optional
        Root directory containing per-subject folders.
        Default is 'eeg-motor-movementimagery-dataset-1.0.0/files'.
    out_csv : str, optional
        Output CSV filename. Default is 'eeg_features.csv'.

    Returns:
    pd.DataFrame
        DataFrame indexed by subject_id with one column per extracted feature.
    """

    Path("outputs").mkdir(exist_ok=True)
    logging.basicConfig(
        filename="outputs/feature_extraction.log",
        level=logging.INFO,
        format="%(asctime)s - %(message)s"
    )
    logging.info("Feature extraction started")
    
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

            # tar = compute_theta_alpha_ratio(raw_r01)    # use eyes open
            # row.update(tar)

        except Exception as e:
            row['preprocessing_error'] = str(e)
            logging.error(f"{sid}: {e}")
        else:
            logging.info(f"{sid}: features extracted successfully")
        
        rows.append(row)
    
    df = pd.DataFrame(rows).set_index("subject_id")
    df.to_csv(out_csv)
    return df

    

if __name__ == "__main__":
    all_subjects_analysis()