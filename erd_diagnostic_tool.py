"""
ERD Diagnostic Tool - Why is ERD not showing suppression???????

Author: Shaheer
Date: November 2024
"""

import numpy as np
import mne
from pathlib import Path
import matplotlib.pyplot as plt
from scipy import signal
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def investigate_erd_issue(subject_id='S002', base_path='raw_data'):
    """
    Deep dive into why ERD is not showing suppression.
    """
    logger.info("="*70)
    logger.info(f"ERD DIAGNOSTIC FOR {subject_id}")
    logger.info("="*70)
    
    # Load one motor imagery run
    run_id = 4  # First motor imagery run
    filepath = Path(base_path) / subject_id / f"{subject_id}R{run_id:02d}.edf"
    
    if not filepath.exists():
        logger.error(f"File not found: {filepath}")
        return
    
    # Load raw data
    raw = mne.io.read_raw_edf(filepath, preload=True, verbose=False)
    raw.rename_channels(lambda x: x.strip('.'))
    
    # Pick C3 for analysis (left motor cortex)
    if 'C3' not in raw.ch_names:
        logger.error("C3 not found in channels")
        return
    
    raw.pick_channels(['C3'])
    
    # Get sampling frequency
    sfreq = raw.info['sfreq']
    logger.info(f"Sampling frequency: {sfreq} Hz")
    
    # Get events
    events, event_dict = mne.events_from_annotations(raw, verbose=False)
    logger.info(f"Events found: {event_dict}")
    
    # Create figure for diagnostic plots
    fig, axes = plt.subplots(4, 2, figsize=(15, 12))
    fig.suptitle(f'ERD Diagnostic: {subject_id} - Why No Suppression?', fontsize=14, fontweight='bold')
    
    # ============ PLOT 1: Raw Signal ============
    ax = axes[0, 0]
    
    # Get 30 seconds of raw data
    raw_data = raw.get_data()[0, :int(30*sfreq)]
    time = np.arange(len(raw_data)) / sfreq
    
    ax.plot(time, raw_data * 1e6, 'b-', alpha=0.7)
    ax.set_title('Raw EEG Signal (C3)', fontweight='bold')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude (μV)')
    ax.grid(True, alpha=0.3)
    
    # ============ PLOT 2: Power Spectrum ============
    ax = axes[0, 1]
    
    # Compute PSD
    psd = raw.compute_psd(method='welch', fmin=1, fmax=40, verbose=False)
    psd_data, freqs = psd.get_data(return_freqs=True)
    
    ax.semilogy(freqs, psd_data[0], 'g-', linewidth=2)
    ax.axvspan(8, 13, alpha=0.2, color='yellow', label='Mu band')
    ax.axvspan(13, 30, alpha=0.2, color='orange', label='Beta band')
    ax.set_title('Power Spectral Density', fontweight='bold')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Power (V²/Hz)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # ============ PLOT 3: Event-locked Analysis ============
    # Apply minimal filtering
    raw.filter(1, 40, verbose=False)
    
    # Create epochs
    if 'T2' in event_dict:  # Right fist (should show C3 suppression)
        epochs = mne.Epochs(raw, events, event_id={'T2': event_dict['T2']},
                           tmin=-2, tmax=4, baseline=None, preload=True, verbose=False)
        
        if len(epochs) > 0:
            # Plot ERP
            ax = axes[1, 0]
            evoked = epochs.average()
            times = evoked.times
            data = evoked.data[0] * 1e6
            
            ax.plot(times, data, 'r-', linewidth=2)
            ax.axvline(0, color='k', linestyle='--', alpha=0.5)
            ax.axhspan(-10, 10, alpha=0.2, color='gray')
            ax.set_title('Event-Related Potential (Right Fist)', fontweight='bold')
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Amplitude (μV)')
            ax.grid(True, alpha=0.3)
            
            # ============ PLOT 4: Time-Frequency WITHOUT baseline correction ============
            ax = axes[1, 1]
            
            # Compute TFR
            freqs_tfr = np.arange(5, 30, 1)
            tfr = mne.time_frequency.tfr_morlet(
                epochs, freqs=freqs_tfr, n_cycles=freqs_tfr/2,
                use_fft=True, return_itc=False, average=True, verbose=False
            )
            
            # Plot raw power (no baseline correction)
            im = ax.imshow(tfr.data[0], aspect='auto', origin='lower',
                         extent=[times[0], times[-1], freqs_tfr[0], freqs_tfr[-1]],
                         cmap='viridis')
            ax.axvline(0, color='white', linestyle='--', alpha=0.5)
            ax.set_title('Raw Power (No Baseline Correction)', fontweight='bold')
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Frequency (Hz)')
            plt.colorbar(im, ax=ax, label='Power')
            
            # ============ PLOT 5: Time-Frequency WITH baseline correction ============
            ax = axes[2, 0]
            
            # Apply baseline correction
            tfr.apply_baseline(baseline=(-1.5, -0.5), mode='percent', verbose=False)
            
            # Plot ERD/ERS
            im = ax.imshow(tfr.data[0] * 100, aspect='auto', origin='lower',
                         extent=[times[0], times[-1], freqs_tfr[0], freqs_tfr[-1]],
                         cmap='RdBu_r', vmin=-50, vmax=50)
            ax.axvline(0, color='black', linestyle='--', alpha=0.5)
            ax.axhline(10, color='white', linestyle='--', alpha=0.3)  # Mu band
            ax.set_title('ERD/ERS (% change from baseline)', fontweight='bold')
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Frequency (Hz)')
            plt.colorbar(im, ax=ax, label='ERD/ERS (%)')
            
            # ============ PLOT 6: Mu Band Time Course ============
            ax = axes[2, 1]
            
            # Extract mu band (8-13 Hz)
            mu_band = (freqs_tfr >= 8) & (freqs_tfr <= 13)
            mu_power = tfr.data[0, mu_band, :].mean(axis=0) * 100
            
            ax.plot(times, mu_power, 'b-', linewidth=2)
            ax.axhline(0, color='k', linestyle='-', alpha=0.3)
            ax.axhline(-20, color='g', linestyle='--', alpha=0.5, label='Expected ERD')
            ax.axvline(0, color='k', linestyle='--', alpha=0.5)
            ax.fill_between([0, 3], -50, 50, alpha=0.1, color='red', label='Motor Imagery')
            ax.set_title('Mu Power Time Course (8-13 Hz)', fontweight='bold')
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Power Change (%)')
            ax.set_ylim(-50, 50)
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Calculate actual ERD values
            baseline_period = (times >= -1.5) & (times <= -0.5)
            task_period = (times >= 0.5) & (times <= 2.5)
            
            baseline_power = mu_power[baseline_period].mean()
            task_power = mu_power[task_period].mean()
            
            ax.text(0.5, 0.95, f'Baseline: {baseline_power:.1f}%\nTask: {task_power:.1f}%\nERD: {task_power:.1f}%',
                   transform=ax.transAxes, fontsize=10,
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            # ============ PLOT 7: Different Baseline Windows ============
            ax = axes[3, 0]
            
            baseline_windows = [
                (-2.0, -1.0, 'Long baseline'),
                (-1.0, 0.0, 'Standard baseline'),
                (-0.5, 0.0, 'Short baseline'),
                (-0.2, 0.0, 'Minimal baseline')
            ]
            
            erd_values = []
            labels = []
            
            for start, end, label in baseline_windows:
                # Recompute TFR with different baseline
                tfr_test = mne.time_frequency.tfr_morlet(
                    epochs, freqs=freqs_tfr, n_cycles=freqs_tfr/2,
                    use_fft=True, return_itc=False, average=True, verbose=False
                )
                tfr_test.apply_baseline(baseline=(start, end), mode='percent', verbose=False)
                
                # Calculate ERD
                mu_band = (freqs_tfr >= 8) & (freqs_tfr <= 13)
                mu_power_test = tfr_test.data[0, mu_band, :].mean(axis=0) * 100
                task_period = (tfr_test.times >= 0.5) & (tfr_test.times <= 2.5)
                erd = mu_power_test[task_period].mean()
                
                erd_values.append(erd)
                labels.append(f'{label}\n({start:.1f} to {end:.1f}s)')
            
            bars = ax.bar(range(len(erd_values)), erd_values, color=['red', 'orange', 'yellow', 'green'])
            ax.axhline(0, color='k', linestyle='-', alpha=0.3)
            ax.axhline(-20, color='g', linestyle='--', alpha=0.5)
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=45, ha='right')
            ax.set_ylabel('ERD (%)')
            ax.set_title('ERD with Different Baseline Windows', fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
            
            # Add value labels
            for bar, val in zip(bars, erd_values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                       f'{val:.1f}%', ha='center', va='bottom' if val < 0 else 'top')
            
            # ============ PLOT 8: Diagnostic Summary ============
            ax = axes[3, 1]
            ax.axis('off')
            
            # Analyze the issue
            issues = []
            
            if all(v > -5 for v in erd_values):
                issues.append("• NO ERD detected in any baseline configuration")
                issues.append("• Possible causes:")
                issues.append("  - Subject not performing motor imagery")
                issues.append("  - Wrong electrode placement")
                issues.append("  - EMG contamination masking ERD")
                issues.append("  - Excessive alpha during baseline")
            elif any(v < -10 for v in erd_values):
                issues.append("• ERD detected with specific baseline")
                issues.append("• Use shorter baseline (-0.5 to 0s)")
            
            # Add frequency analysis
            if psd_data[0, (freqs >= 8) & (freqs <= 13)].mean() < 1e-12:
                issues.append("• Weak mu rhythm - may be naturally low")
            
            diagnostic_text = "DIAGNOSTIC SUMMARY:\n\n" + "\n".join(issues)
            
            ax.text(0.05, 0.95, diagnostic_text, transform=ax.transAxes,
                   fontsize=10, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    plt.tight_layout()
    
    # Save figure
    output_path = f'erd_diagnostic_{subject_id}.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    logger.info(f"\nDiagnostic plot saved: {output_path}")
    
    # Return diagnostic results
    return {
        'subject_id': subject_id,
        'erd_values': erd_values if 'erd_values' in locals() else [],
        'has_mu_rhythm': psd_data[0, (freqs >= 8) & (freqs <= 13)].mean() > 1e-12,
        'recommendation': 'Use minimal baseline (-0.2 to 0s) and ensure subject performs kinesthetic imagery'
    }


def test_synthetic_erd():
    """
    Test ERD calculation on synthetic data to verify the method works.
    """
    logger.info("\n" + "="*70)
    logger.info("TESTING ERD CALCULATION ON SYNTHETIC DATA")
    logger.info("="*70 + "\n")
    
    # Create synthetic signal with known ERD
    sfreq = 160
    duration = 10
    times = np.arange(0, duration, 1/sfreq)
    
    # Create mu rhythm (10 Hz) with ERD
    baseline_amplitude = 20  # μV
    erd_amplitude = 10  # 50% reduction
    
    sig = np.zeros_like(times)
    
    # Baseline period (0-3s)
    baseline_mask = times < 3
    sig[baseline_mask] = baseline_amplitude * np.sin(2 * np.pi * 10 * times[baseline_mask])
    
    # ERD period (3-7s) - reduced amplitude
    erd_mask = (times >= 3) & (times < 7)
    sig[erd_mask] = erd_amplitude * np.sin(2 * np.pi * 10 * times[erd_mask])
    
    # Recovery (7-10s)
    recovery_mask = times >= 7
    sig[recovery_mask] = baseline_amplitude * np.sin(2 * np.pi * 10 * times[recovery_mask])
    
    # Add noise
    sig += np.random.randn(len(sig)) * 2
    
    # Calculate spectrogram
    f, t_spec, Sxx = signal.spectrogram(sig, sfreq, nperseg=sfreq, noverlap=sfreq//2)
    
    # Calculate ERD
    mu_band = (f >= 8) & (f <= 12)
    mu_power = Sxx[mu_band, :].mean(axis=0)
    
    baseline_power = mu_power[t_spec < 3].mean()
    erd_power = mu_power[(t_spec >= 3) & (t_spec < 7)].mean()
    
    erd_percent = ((erd_power - baseline_power) / baseline_power) * 100
    
    logger.info(f"Synthetic data test:")
    logger.info(f"  Expected ERD: -50%")
    logger.info(f"  Calculated ERD: {erd_percent:.1f}%")
    
    if erd_percent < -40:
        logger.info("  ✓ ERD calculation method is CORRECT")
    else:
        logger.error("  ✗ ERD calculation method has issues")
    
    # Plot synthetic results
    fig, axes = plt.subplots(2, 1, figsize=(10, 6))
    
    # Signal
    axes[0].plot(times, sig)
    axes[0].axvspan(3, 7, alpha=0.2, color='red')
    axes[0].set_title('Synthetic Signal with Known ERD')
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('Amplitude (μV)')
    
    # Spectrogram
    axes[1].imshow(10*np.log10(Sxx), aspect='auto', origin='lower',
                  extent=[t_spec[0], t_spec[-1], f[0], f[-1]])
    axes[1].axvline(3, color='white', linestyle='--')
    axes[1].axvline(7, color='white', linestyle='--')
    axes[1].set_title(f'Spectrogram (ERD = {erd_percent:.1f}%)')
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Frequency (Hz)')
    
    plt.tight_layout()
    plt.savefig('synthetic_erd_test.png')
    plt.show()
    
    return erd_percent


if __name__ == "__main__":
    # First test on synthetic data
    logger.info("Step 1: Testing ERD calculation on synthetic data...")
    synthetic_erd = test_synthetic_erd()
    
    # Then diagnose real subject
    logger.info("\nStep 2: Diagnosing real subject data...")
    result = investigate_erd_issue('S001')  # Use one of our 'good' subjects
    
    logger.info("\n" + "="*70)
    logger.info("DIAGNOSIS COMPLETE")
    logger.info("="*70)
    
    if result:
        logger.info(f"\nSubject {result['subject_id']}:")
        logger.info(f"  ERD values: {result['erd_values']}")
        logger.info(f"  Has mu rhythm: {result['has_mu_rhythm']}")
        logger.info(f"  Recommendation: {result['recommendation']}")
