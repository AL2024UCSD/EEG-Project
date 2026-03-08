"""
Unit tests for resting_state_features_extraction.py

Tests cover pure computation functions (no EEG file or MNE Raw object required)
and MNE-based feature functions using a synthetic Raw object.

Run with:
    pytest test_features.py -v
"""

import numpy as np
import pytest
import mne
from unittest.mock import patch
from resting_state_features_extraction import (
    spectral_entropy,
    binarize_signal,
    lempel_ziv_complexity_calculation,
    freq_curve,
    fit_with_multistart,
    apply_laplacian_filtering,
    resting_alpha_power,
    alpha_power_variability,
    interhemispheric_coherence,
    resting_lower_beta_power,
    resting_upper_beta_power,
    compute_pse,
    lempel_ziv_complexity,
)


# Helper functions

def make_synthetic_raw(sfreq=160.0, duration=60.0, seed=42):
    """
    Create a minimal synthetic MNE Raw object with motor channels.
    Uses pink-ish noise to roughly mimic EEG spectral shape.
    """
    rng = np.random.default_rng(seed)

    ch_names = ['C3', 'C4', 'Cz', 'FC1', 'FC2', 'CP1', 'CP2',
                'FC3', 'C1', 'CP3', 'C5',
                'FC4', 'C2', 'CP4', 'C6',
                'FCz', 'CPz', 'F1', 'F2', 'P1', 'P2']

    n_times = int(sfreq * duration)
    n_ch = len(ch_names)

    # Generate white noise, then apply mild 1/f shaping via cumsum
    data = rng.standard_normal((n_ch, n_times)) * 1e-6
    data = np.cumsum(data, axis=1)
    data -= data.mean(axis=1, keepdims=True)

    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg')
    raw = mne.io.RawArray(data, info, verbose=False)
    return raw


# spectral entropy test

class TestSpectralEntropy:

    def test_flat_spectrum_returns_one(self):
        """A uniform PSD should have maximum entropy (1.0)."""
        psd = np.ones(64)
        result = spectral_entropy(psd)
        assert np.isclose(result, 1.0, atol=1e-6)

    def test_impulse_spectrum_returns_near_zero(self):
        """A PSD concentrated in one bin should have near-zero entropy."""
        psd = np.zeros(64)
        psd[0] = 1.0
        result = spectral_entropy(psd)
        assert result < 0.01

    def test_output_bounded(self):
        """Result should always be in [0, 1]."""
        rng = np.random.default_rng(0)
        for _ in range(20):
            psd = np.abs(rng.standard_normal(50)) + 1e-6
            result = spectral_entropy(psd)
            assert 0.0 <= result <= 1.0

    def test_eps_prevents_log_zero(self):
        """Zero values in PSD should not cause NaN or error."""
        psd = np.zeros(32)
        psd[5] = 1.0
        result = spectral_entropy(psd, eps=1e-12)
        assert np.isfinite(result)


# binarize signal test

class TestBinarizeSignal:

    def test_output_is_binary(self):
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = binarize_signal(x)
        assert set(result).issubset({0, 1})

    def test_above_median_is_one(self):
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = binarize_signal(x)
        # median is 3.0; values above should be 1
        assert result[3] == 1
        assert result[4] == 1

    def test_below_median_is_zero(self):
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = binarize_signal(x)
        assert result[0] == 0
        assert result[1] == 0

    def test_output_shape_matches_input(self):
        x = np.random.randn(1000)
        assert binarize_signal(x).shape == x.shape


# LZC test

class TestLempelZivComplexity:

    def test_constant_signal_low_complexity(self):
        """A constant binary sequence should have very low complexity."""
        seq = np.zeros(500, dtype=int)
        lzc = lempel_ziv_complexity_calculation(seq)
        assert lzc < 0.1

    def test_random_signal_higher_complexity(self):
        """A random binary sequence should have higher complexity than constant."""
        rng = np.random.default_rng(0)
        constant = np.zeros(500, dtype=int)
        random_seq = rng.integers(0, 2, size=500)
        assert lempel_ziv_complexity_calculation(random_seq) > \
               lempel_ziv_complexity_calculation(constant)

    def test_output_is_positive(self):
        rng = np.random.default_rng(1)
        seq = rng.integers(0, 2, size=300)
        assert lempel_ziv_complexity_calculation(seq) > 0

    def test_output_is_finite(self):
        rng = np.random.default_rng(2)
        seq = rng.integers(0, 2, size=1000)
        assert np.isfinite(lempel_ziv_complexity_calculation(seq))


# freq curve test

class TestFreqCurve:

    def test_output_shape(self):
        freqs = np.linspace(2, 35, 100)
        result = freq_curve(freqs, 1.0, 0.0, 10.0, 5.0, 3.0, 10.0, 23.0, 2.0, 2.0)
        assert result.shape == freqs.shape

    def test_output_is_finite(self):
        freqs = np.linspace(2, 35, 100)
        result = freq_curve(freqs, 1.0, 0.0, 10.0, 5.0, 3.0, 10.0, 23.0, 2.0, 2.0)
        assert np.all(np.isfinite(result))

    def test_zero_peaks_matches_noise_floor(self):
        """With zero peak amplitudes, output should equal the noise floor only."""
        freqs = np.linspace(2, 35, 100)
        result = freq_curve(freqs, 1.0, 5.0, 10.0, 0.0, 0.0, 10.0, 23.0, 2.0, 2.0)
        expected = 5.0 + 10.0 / freqs ** 1.0
        np.testing.assert_allclose(result, expected, rtol=1e-6)


# fit_with_multistart test

class TestFitWithMultistart:

    def _make_clean_psd(self):
        """Generate a synthetic PSD from known parameters for fitting."""
        freqs = np.linspace(2, 35, 200)
        true_params = [1.0, 5.0, 10.0, 8.0, 4.0, 10.0, 23.0, 1.5, 2.0]
        psd = freq_curve(freqs, *true_params)
        return freqs, psd, true_params

    def test_returns_nine_params(self):
        freqs, psd, _ = self._make_clean_psd()
        bounds = (
            [0.1, -150, 0, 0, 0, 8, 13, 1, 1],
            [2.0,  150, 50, 100, 100, 13, 30, 10, 10]
        )
        params = fit_with_multistart(freqs, psd, bounds)
        assert len(params) == 9

    def test_fit_is_finite(self):
        freqs, psd, _ = self._make_clean_psd()
        bounds = (
            [0.1, -150, 0, 0, 0, 8, 13, 1, 1],
            [2.0,  150, 50, 100, 100, 13, 30, 10, 10]
        )
        params = fit_with_multistart(freqs, psd, bounds)
        assert np.all(np.isfinite(params))

    def test_raises_on_impossible_fit(self):
        """NaN input should cause all starts to fail and raise RuntimeError."""
        freqs = np.linspace(2, 35, 100)
        psd = np.full(100, np.nan)
        bounds = (
            [0.1, -150, 0, 0, 0, 8, 13, 1, 1],
            [2.0,  150, 50, 100, 100, 13, 30, 10, 10]
        )
        with pytest.raises(RuntimeError, match="All starting points failed"):
            fit_with_multistart(freqs, psd, bounds)


# laplacian filtering tests

class TestApplyLaplacianFiltering:

    def test_output_keys_match_requested_channels(self):
        raw = make_synthetic_raw()
        channels = ['C3', 'C4']
        laplacian_data, sfreq = apply_laplacian_filtering(raw, channels)
        assert set(laplacian_data.keys()) == {'C3', 'C4'}

    def test_output_length_matches_input(self):
        raw = make_synthetic_raw(duration=10.0)
        channels = ['C3', 'C4']
        laplacian_data, sfreq = apply_laplacian_filtering(raw, channels)
        expected_samples = int(160.0 * 10.0)
        for ch, sig in laplacian_data.items():
            assert len(sig) == expected_samples

    def test_sfreq_correct(self):
        raw = make_synthetic_raw(sfreq=256.0)
        laplacian_data, sfreq = apply_laplacian_filtering(raw, ['C3'])
        assert sfreq == 256.0

    def test_laplacian_differs_from_raw(self):
        """Laplacian-filtered signal should differ from the original channel signal."""
        raw = make_synthetic_raw()
        raw_data = raw.get_data()
        c3_idx = raw.ch_names.index('C3')
        original = raw_data[c3_idx, :]
        laplacian_data, _ = apply_laplacian_filtering(raw, ['C3'])
        assert not np.allclose(laplacian_data['C3'], original)


# output tests

class TestFeatureFunctionOutputs:
    """
    Verify each feature function returns a dict with the
    expected keys and finite values. Does not validate numerical accuracy.
    """

    @pytest.fixture(scope="class")
    def raw(self):
        return make_synthetic_raw(duration=60.0)

    def _check_dict(self, result, expected_keys):
        assert isinstance(result, dict)
        for key in expected_keys:
            assert key in result, f"Missing key: {key}"
            assert np.isfinite(result[key]) or np.isnan(result[key]), \
                f"Non-finite value for {key}: {result[key]}"

    def test_resting_alpha_power(self, raw):
        result = resting_alpha_power(raw)
        self._check_dict(result, [
            'rpl_alpha', 'tar', 'resting_alpha_power',
            'resting_total_power', 'alpha_power_c3', 'alpha_power_c4', 'alpha_asymmetry'
        ])

    def test_rpl_alpha_is_proportion(self, raw):
        result = resting_alpha_power(raw)
        assert 0.0 < result['rpl_alpha'] < 1.0

    def test_alpha_power_variability(self, raw):
        result = alpha_power_variability(raw)
        self._check_dict(result, ['alpha_var_C3', 'alpha_cv_C3', 'alpha_var_C4', 'alpha_cv_C4'])

    def test_alpha_var_nonnegative(self, raw):
        result = alpha_power_variability(raw)
        assert result['alpha_var_C3'] >= 0.0
        assert result['alpha_var_C4'] >= 0.0

    def test_interhemispheric_coherence(self, raw):
        result = interhemispheric_coherence(raw)
        self._check_dict(result, [
            'coherence_mu', 'coherence_beta',
            'coherence_low_beta', 'coherence_upper_beta'
        ])

    def test_coherence_bounded(self, raw):
        """Coherence values must be in [0, 1]."""
        result = interhemispheric_coherence(raw)
        for key in ['coherence_mu', 'coherence_beta', 'coherence_low_beta', 'coherence_upper_beta']:
            assert 0.0 <= result[key] <= 1.0, f"{key} out of bounds: {result[key]}"

    def test_resting_lower_beta_power(self, raw):
        result = resting_lower_beta_power(raw)
        self._check_dict(result, ['rpl_lower_beta', 'resting_lower_beta_power'])

    def test_resting_upper_beta_power(self, raw):
        result = resting_upper_beta_power(raw)
        self._check_dict(result, ['rpl_upper_beta', 'resting_upper_beta_power'])

    def test_compute_pse(self, raw):
        result = compute_pse(raw)
        self._check_dict(result, [
            'pse_C3', 'pse_C4', 'pse_Cz',
            'pse_FC1', 'pse_FC2', 'pse_CP1', 'pse_CP2', 'pse_avg'
        ])

    def test_pse_bounded(self, raw):
        result = compute_pse(raw)
        for key, val in result.items():
            assert 0.0 <= val <= 1.0, f"{key} out of bounds: {val}"

    def test_lempel_ziv_complexity(self, raw):
        result = lempel_ziv_complexity(raw)
        self._check_dict(result, [
            'lzc_C3', 'lzc_C4', 'lzc_Cz',
            'lzc_FC1', 'lzc_FC2', 'lzc_CP1', 'lzc_CP2', 'lzc_avg'
        ])

    def test_lzc_positive(self, raw):
        result = lempel_ziv_complexity(raw)
        for key, val in result.items():
            assert val > 0.0, f"{key} should be positive, got {val}"
