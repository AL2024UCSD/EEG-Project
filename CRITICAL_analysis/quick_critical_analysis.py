"""
Quick Critical Analysis Script
==============================

Run this on your EXISTING processed data to generate the critical plots
without reprocessing all 109 subjects.

This script:
1. Loads your existing multi_subject results
2. Computes BCI accuracy for each subject
3. Generates the CRITICAL hypothesis-testing plots
4. Shows the truth about your data

Usage:
    python quick_critical_analysis.py --data_dir <path_to_processed_data>
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import mne
from scipy import stats
from scipy.stats import pearsonr
import pandas as pd
import logging
import argparse
from typing import Dict, List
from matplotlib.gridspec import GridSpec
from tqdm import tqdm

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


class QuickAnalyzer:
    """Quick analyzer for existing processed data."""
    
    def __init__(self, data_dir: str, output_dir: str = 'CRITICAL_analysis', n_subjects: int = 20):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.n_subjects = n_subjects
        
        # Will be populated by load_existing_data()
        self.subject_ids = []
        self.bci_accuracies = {}
        self.similarity_metrics = {}
        self.band_powers = {}
        self.psd_data = {}
    
    def compute_quick_bci_accuracy(self, epochs_imagined):
        """Quick BCI accuracy estimation."""
        from mne.decoding import CSP
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        from sklearn.model_selection import StratifiedKFold, cross_val_score
        from sklearn.pipeline import Pipeline
        
        try:
            epochs_train = epochs_imagined.copy().crop(1.0, 2.0)
            events = epochs_train.events
            unique_events = np.unique(events[:, -1])
            
            if len(unique_events) < 2:
                return 0.50
            
            labels = (events[:, -1] == unique_events[1]).astype(int)
            
            csp = CSP(n_components=4, reg=None, log=True)
            lda = LinearDiscriminantAnalysis()
            clf = Pipeline([('CSP', csp), ('LDA', lda)])
            
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            scores = cross_val_score(clf, epochs_train.get_data(), labels, cv=cv)
            
            return scores.mean()
        except:
            return 0.50
    
    def load_from_processed_files(self, base_data_path: str):
        """
        Load from PhysioNet EEG Motor Movement/Imagery Dataset.
        
        Structure: 
            base_path/
                S001/
                    S001R01.edf (baseline eyes open)
                    S001R02.edf (baseline eyes closed)
                    S001R03.edf (real left/right fist)
                    S001R04.edf (imagined left/right fist)
                    ...
                S002/
                    S002R01.edf
                    ...
        """
        base_path = Path(base_data_path)
        
        if not base_path.exists():
            logger.error(f"Data path does not exist: {base_path}")
            return
        
        # Find all subject directories
        all_items = list(base_path.iterdir())
        subject_dirs = sorted([d for d in all_items if d.is_dir() and d.name.startswith('S')])
        
        if not subject_dirs:
            logger.error(f"No subject directories found in {base_path}")
            logger.error(f"Looking for directories starting with 'S' (e.g., S001, S002, ...)")
            logger.error(f"Found items: {[item.name for item in all_items[:10]]}")
            return
        
        logger.info(f"Found {len(subject_dirs)} subject directories")
        logger.info(f"Processing first {self.n_subjects} subjects for quick analysis...")
        
        for subject_dir in tqdm(subject_dirs[:self.n_subjects], desc="Processing subjects"):
            subject_id = subject_dir.name
            
            try:
                # PhysioNet naming convention: S001R04.edf, S001R08.edf, etc.
                # Runs 4, 8, 12 = Motor IMAGERY (left/right fist)
                # Runs 3, 7, 11 = Motor EXECUTION (left/right fist - actual movement)
                imag_files = [subject_dir / f"{subject_id}R{run:02d}.edf" for run in [4, 8, 12]]
                real_files = [subject_dir / f"{subject_id}R{run:02d}.edf" for run in [3, 7, 11]]
                
                # Check if imagery files exist
                missing_imag = [f.name for f in imag_files if not f.exists()]
                if missing_imag:
                    logger.debug(f"  {subject_id}: Missing imagery files: {missing_imag}")
                    continue
                
                # Load and concatenate imagery runs
                raw_imag_list = []
                for f in imag_files:
                    try:
                        raw = mne.io.read_raw_edf(f, preload=True, verbose=False)
                        raw.rename_channels(lambda x: x.strip('.'))
                        
                        # Check available channels
                        available_motor_channels = [ch for ch in ['C3', 'C4', 'Cz', 'FC1', 'FC2', 'CP1', 'CP2'] 
                                                   if ch in raw.ch_names]
                        
                        if not available_motor_channels:
                            logger.warning(f"  {subject_id}: No motor channels found in {f.name}")
                            continue
                        
                        raw.pick_channels(available_motor_channels)
                        raw.filter(7, 30, fir_design='firwin', verbose=False)
                        raw_imag_list.append(raw)
                    except Exception as e:
                        logger.warning(f"  {subject_id}: Failed to load {f.name}: {e}")
                        continue
                
                if not raw_imag_list:
                    logger.debug(f"  {subject_id}: No valid imagery data loaded")
                    continue
                
                raw_imag = mne.io.concatenate_raws(raw_imag_list)
                
                # Create epochs from imagery data
                events, event_dict = mne.events_from_annotations(raw_imag, verbose=False)
                
                if len(events) == 0:
                    logger.debug(f"  {subject_id}: No events found in imagery data")
                    continue
                
                epochs_imag = mne.Epochs(raw_imag, events, event_id=event_dict,
                                        tmin=-1.0, tmax=4.0, baseline=(-1, 0),
                                        preload=True, verbose=False)
                
                if len(epochs_imag) < 10:
                    logger.debug(f"  {subject_id}: Too few epochs ({len(epochs_imag)}), skipping")
                    continue
                
                # Compute BCI accuracy
                accuracy = self.compute_quick_bci_accuracy(epochs_imag)
                self.bci_accuracies[subject_id] = accuracy
                
                # Compute PSD for similarity
                psd = epochs_imag.compute_psd(fmin=7, fmax=30, method='welch', verbose=False)
                psd_data = psd.get_data().mean(axis=0).flatten()
                self.psd_data[subject_id] = {'imagined': psd_data}
                
                # Load real movement for comparison
                missing_real = [f.name for f in real_files if not f.exists()]
                if not missing_real:
                    try:
                        raw_real_list = []
                        for f in real_files:
                            raw = mne.io.read_raw_edf(f, preload=True, verbose=False)
                            raw.rename_channels(lambda x: x.strip('.'))
                            available_motor_channels = [ch for ch in ['C3', 'C4', 'Cz', 'FC1', 'FC2', 'CP1', 'CP2'] 
                                                       if ch in raw.ch_names]
                            if available_motor_channels:
                                raw.pick_channels(available_motor_channels)
                                raw.filter(7, 30, fir_design='firwin', verbose=False)
                                raw_real_list.append(raw)
                        
                        if raw_real_list:
                            raw_real = mne.io.concatenate_raws(raw_real_list)
                            events_real, event_dict_real = mne.events_from_annotations(raw_real, verbose=False)
                            
                            if len(events_real) > 0:
                                epochs_real = mne.Epochs(raw_real, events_real, event_id=event_dict_real,
                                                        tmin=-1.0, tmax=4.0, baseline=(-1, 0),
                                                        preload=True, verbose=False)
                                
                                if len(epochs_real) >= 10:
                                    psd_real = epochs_real.compute_psd(fmin=7, fmax=30, method='welch', verbose=False)
                                    psd_real_data = psd_real.get_data().mean(axis=0).flatten()
                                    self.psd_data[subject_id]['real'] = psd_real_data
                                    
                                    # Compute similarity (ensure same length)
                                    min_len = min(len(psd_real_data), len(psd_data))
                                    corr, _ = pearsonr(psd_real_data[:min_len], psd_data[:min_len])
                                    self.similarity_metrics[subject_id] = {'psd_correlation': corr}
                    except Exception as e:
                        logger.debug(f"  {subject_id}: Could not load real movement data: {e}")
                
                self.subject_ids.append(subject_id)
                logger.info(f"  ✓ {subject_id}: BCI={accuracy:.1%}, n_epochs={len(epochs_imag)}")
                
            except Exception as e:
                logger.warning(f"  ✗ {subject_id}: {e}")
                continue
        
        logger.info(f"\n{'='*70}")
        logger.info(f"Successfully processed: {len(self.subject_ids)} subjects")
        logger.info(f"  With similarity data: {len(self.similarity_metrics)} subjects")
        logger.info(f"{'='*70}\n")
    
    def plot_critical_hypothesis_test(self):
        """THE MOST IMPORTANT PLOT: Does similarity predict BCI performance?"""
        
        # Extract data
        subjects_with_data = [s for s in self.subject_ids if s in self.similarity_metrics]
        
        if len(subjects_with_data) < 5:
            logger.error(f"Not enough subjects with similarity data: {len(subjects_with_data)}")
            logger.error("Need at least 5 subjects for meaningful analysis")
            logger.error(f"Total subjects processed: {len(self.subject_ids)}")
            logger.error(f"Subjects with similarity: {len(subjects_with_data)}")
            return
        
        accuracies = [self.bci_accuracies[s] for s in subjects_with_data]
        similarities = [self.similarity_metrics[s]['psd_correlation'] for s in subjects_with_data]
        
        logger.info(f"\nGenerating critical plot with {len(subjects_with_data)} subjects...")
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        
        # Plot 1: Similarity vs Accuracy Scatter
        ax = axes[0, 0]
        scatter = ax.scatter(similarities, accuracies, c=accuracies, cmap='RdYlGn',
                            vmin=0.5, vmax=0.9, s=100, alpha=0.7, edgecolors='black')
        plt.colorbar(scatter, ax=ax, label='BCI Accuracy')
        
        # Regression line
        slope, intercept, r_value, p_value, stderr = stats.linregress(similarities, accuracies)
        line_x = np.array([min(similarities), max(similarities)])
        line_y = slope * line_x + intercept
        ax.plot(line_x, line_y, 'r--', linewidth=3, alpha=0.8, label=f'r={r_value:.3f}')
        
        # Stats box
        sig_stars = '***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else 'ns'
        ax.text(0.05, 0.95, 
                f'Pearson r = {r_value:.3f}\nr² = {r_value**2:.3f}\np = {p_value:.4f} {sig_stars}\nn = {len(subjects_with_data)}',
                transform=ax.transAxes, fontsize=11, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.9, edgecolor='black', linewidth=2))
        
        ax.axhline(0.70, color='green', linestyle='--', linewidth=2, alpha=0.7, label='70% threshold')
        ax.axhline(0.60, color='orange', linestyle='--', linewidth=2, alpha=0.7, label='60% threshold')
        ax.set_xlabel('Real/Imagined PSD Correlation', fontsize=13, fontweight='bold')
        ax.set_ylabel('BCI Classification Accuracy', fontsize=13, fontweight='bold')
        ax.set_title('HYPOTHESIS TEST: Does Similarity Predict BCI Performance?',
                    fontsize=14, fontweight='bold', color='darkred')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Accuracy Distribution
        ax = axes[0, 1]
        ax.hist(accuracies, bins=20, color='steelblue', alpha=0.7, edgecolor='black')
        ax.axvline(0.70, color='green', linestyle='--', linewidth=2, label='70% (BCI Compatible)')
        ax.axvline(0.60, color='orange', linestyle='--', linewidth=2, label='60% (Threshold)')
        ax.axvline(np.mean(accuracies), color='red', linestyle='-', linewidth=3, label=f'Mean={np.mean(accuracies):.1%}')
        ax.set_xlabel('BCI Accuracy', fontsize=13, fontweight='bold')
        ax.set_ylabel('Number of Subjects', fontsize=13, fontweight='bold')
        ax.set_title(f'BCI Performance Distribution (n={len(accuracies)})',
                    fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Plot 3: Stratified Analysis
        ax = axes[1, 0]
        
        # Categorize subjects
        good_bci = [s for s in subjects_with_data if self.bci_accuracies[s] > 0.70]
        poor_bci = [s for s in subjects_with_data if self.bci_accuracies[s] < 0.60]
        
        good_similarities = [self.similarity_metrics[s]['psd_correlation'] for s in good_bci]
        poor_similarities = [self.similarity_metrics[s]['psd_correlation'] for s in poor_bci]
        
        bp = ax.boxplot([good_similarities, poor_similarities], labels=['Good BCI\n(>70%)', 'Poor BCI\n(<60%)'],
                        patch_artist=True, widths=0.6)
        bp['boxes'][0].set_facecolor('lightgreen')
        bp['boxes'][1].set_facecolor('lightcoral')
        
        # Add significance test
        if len(good_similarities) > 0 and len(poor_similarities) > 0:
            t_stat, t_p = stats.ttest_ind(good_similarities, poor_similarities)
            ax.text(0.5, 0.95, f't-test: p={t_p:.4f} {"***" if t_p<0.001 else "**" if t_p<0.01 else "*" if t_p<0.05 else "ns"}',
                   transform=ax.transAxes, ha='center', fontsize=11,
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        ax.set_ylabel('PSD Correlation', fontsize=13, fontweight='bold')
        ax.set_title('Similarity: Good vs Poor BCI Users', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Plot 4: Interpretation Guide
        ax = axes[1, 1]
        ax.axis('off')
        
        # Determine verdict
        if abs(r_value) > 0.3 and p_value < 0.05:
            if r_value > 0:
                verdict = "✓ HYPOTHESIS SUPPORTED\nSimilarity POSITIVELY predicts BCI performance"
                color = 'green'
            else:
                verdict = "✗ HYPOTHESIS REJECTED\nSimilarity NEGATIVELY predicts BCI performance"
                color = 'red'
        else:
            verdict = "⚠ HYPOTHESIS NOT SUPPORTED\nNo significant correlation found"
            color = 'orange'
        
        interpretation = f"""
{verdict}

INTERPRETATION:
────────────────────────────────
r = {r_value:.3f}, p = {p_value:.4f}

Strong correlation: |r| > 0.5
Moderate correlation: |r| > 0.3
Weak correlation: |r| > 0.1

Current: {"Strong" if abs(r_value) > 0.5 else "Moderate" if abs(r_value) > 0.3 else "Weak" if abs(r_value) > 0.1 else "None"}

WHAT THIS MEANS:
────────────────────────────────
"""
        
        if r_value > 0.3:
            interpretation += """
Subjects with MORE SIMILAR real/imagined
patterns show BETTER BCI performance.
This supports your original hypothesis.
"""
        elif r_value < -0.3:
            interpretation += """
Subjects with MORE DISSIMILAR real/imagined
patterns show BETTER BCI performance.
Your hypothesis was BACKWARDS!

Strong motor imagery = DIFFERENT pattern
Weak motor imagery = SIMILAR pattern
"""
        else:
            interpretation += """
Similarity does NOT predict BCI performance.
You need to find a DIFFERENT biomarker.

Test for Beta power magnitude, lateralization
strength, or signal-to-noise ratio.
"""
        
        ax.text(0.5, 0.5, interpretation,
               transform=ax.transAxes, fontsize=10, ha='center', va='center',
               bbox=dict(boxstyle='round', facecolor=color, alpha=0.3, edgecolor='black', linewidth=2),
               family='monospace')
        
        plt.suptitle('CRITICAL ANALYSIS: Testing Your BCI Compatibility Hypothesis',
                    fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout()
        
        output_path = self.output_dir / '00_CRITICAL_hypothesis_test.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"\n{'='*70}")
        logger.info(f"CRITICAL RESULT: r={r_value:.3f}, p={p_value:.4f}")
        verdict_first_line = verdict.splitlines()[0]
        logger.info(f"Verdict: {verdict_first_line}")
        logger.info(f"{'='*70}\n")
        logger.info(f"✓ Saved: {output_path}")
    
    def generate_quick_report(self):
        """Generate quick summary report."""
        report_path = self.output_dir / 'QUICK_ANALYSIS_SUMMARY.txt'
        
        subjects_with_data = [s for s in self.subject_ids if s in self.similarity_metrics]
        accuracies = [self.bci_accuracies[s] for s in subjects_with_data]
        similarities = [self.similarity_metrics[s]['psd_correlation'] for s in subjects_with_data]
        
        r_value, p_value = pearsonr(similarities, accuracies)
        
        with open(report_path, 'w') as f:
            f.write("="*70 + "\n")
            f.write("QUICK CRITICAL ANALYSIS SUMMARY\n")
            f.write("="*70 + "\n\n")
            
            f.write("HYPOTHESIS TEST RESULT\n")
            f.write("-"*50 + "\n")
            f.write(f"Research Question: Does real/imagined similarity predict BCI compatibility?\n\n")
            f.write(f"Sample size: {len(subjects_with_data)} subjects\n")
            f.write(f"Correlation: r = {r_value:.3f}\n")
            f.write(f"Significance: p = {p_value:.4f} {'***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else '(not significant)'}\n")
            f.write(f"R-squared: {r_value**2:.3f} ({r_value**2*100:.1f}% variance explained)\n\n")
            
            if abs(r_value) > 0.3 and p_value < 0.05:
                if r_value > 0:
                    f.write("VERDICT: ✓ HYPOTHESIS SUPPORTED\n")
                    f.write("Similarity POSITIVELY predicts BCI performance.\n")
                else:
                    f.write("VERDICT: ✗ HYPOTHESIS REJECTED\n")
                    f.write("Similarity NEGATIVELY predicts BCI performance.\n")
                    f.write("(Your hypothesis was backwards!)\n")
            else:
                f.write("VERDICT: ⚠ HYPOTHESIS NOT SUPPORTED\n")
                f.write("No significant correlation between similarity and BCI performance.\n")
            
            f.write("\n\nBCI PERFORMANCE STATISTICS\n")
            f.write("-"*50 + "\n")
            f.write(f"Mean accuracy: {np.mean(accuracies):.1%} ± {np.std(accuracies):.1%}\n")
            f.write(f"Median accuracy: {np.median(accuracies):.1%}\n")
            f.write(f"Range: {np.min(accuracies):.1%} - {np.max(accuracies):.1%}\n")
            f.write(f"\nBCI-compatible (>70%): {sum(a > 0.70 for a in accuracies)} ({sum(a > 0.70 for a in accuracies)/len(accuracies)*100:.1f}%)\n")
            f.write(f"BCI-incompatible (<60%): {sum(a < 0.60 for a in accuracies)} ({sum(a < 0.60 for a in accuracies)/len(accuracies)*100:.1f}%)\n")
            
            f.write("\n\nRECOMMENDATIONS\n")
            f.write("-"*50 + "\n")
            if abs(r_value) < 0.3:
                f.write("1. Your similarity metric may not be the right biomarker\n")
                f.write("2. Try other features: beta power, lateralization, SNR\n")
                f.write("3. Consider multivariate approach (multiple features)\n")
                f.write("4. Investigate individual good/poor performers for patterns\n")
            else:
                f.write("1. Similarity IS a useful biomarker\n")
                f.write("2. Build classifier using similarity features\n")
                f.write("3. Validate on held-out test set\n")
                f.write("4. Investigate WHY similarity predicts performance\n")
            
            f.write("\n" + "="*70 + "\n")
        
        logger.info(f"✓ Saved report: {report_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Quick critical analysis of PhysioNet EEG Motor Imagery data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python quick_critical_analysis.py --data_dir /path/to/eeg-motor-movementimagery-dataset-1.0.0

Expected data structure:
  data_dir/
    S001/
      S001R01.edf  (baseline eyes open)
      S001R02.edf  (baseline eyes closed)
      S001R03.edf  (real left/right fist movement)
      S001R04.edf  (imagined left/right fist movement)
      ...
      S001R14.edf
    S002/
      S002R01.edf
      ...
    
This script will:
1. Process first 20 subjects (for speed)
2. Compute BCI classification accuracy (CSP-LDA)
3. Test if real/imagined similarity predicts BCI performance
4. Generate critical hypothesis-testing plot
        """
    )
    parser.add_argument('--data_dir', type=str, required=True, 
                       help='Path to PhysioNet dataset directory containing S001/, S002/, etc.')
    parser.add_argument('--output_dir', type=str, default='CRITICAL_analysis', 
                       help='Output directory for results')
    parser.add_argument('--n_subjects', type=int, default=20,
                       help='Number of subjects to process (default: 20 for quick analysis)')
    args = parser.parse_args()
    
    # Validate data directory
    data_path = Path(args.data_dir)
    if not data_path.exists():
        logger.error(f"Data directory does not exist: {data_path}")
        logger.error("\nPlease provide the full path to your PhysioNet dataset")
        logger.error("Example: /Users/shaheerkhan/Documents/EEG-Project/eeg-motor-movementimagery-dataset-1.0.0/")
        return
    
    # Check for subject directories
    subject_dirs = [d for d in data_path.iterdir() if d.is_dir() and d.name.startswith('S')]
    if not subject_dirs:
        logger.error(f"No subject directories found in {data_path}")
        logger.error("\nExpected structure:")
        logger.error("  data_dir/S001/, data_dir/S002/, ..., data_dir/S109/")
        logger.error("\nCheck your path - you may need to go one level deeper or higher")
        return
    
    logger.info("="*70)
    logger.info("QUICK CRITICAL ANALYSIS: BCI COMPATIBILITY")
    logger.info("="*70)
    logger.info(f"\nData directory: {data_path}")
    logger.info(f"Found {len(subject_dirs)} subject directories")
    logger.info(f"Will process first {args.n_subjects} subjects\n")
    
    analyzer = QuickAnalyzer(args.data_dir, args.output_dir, args.n_subjects)
    analyzer.load_from_processed_files(args.data_dir)
    
    if len(analyzer.subject_ids) > 0:
        logger.info(f"\nGenerating visualizations and report...")
        analyzer.plot_critical_hypothesis_test()
        analyzer.generate_quick_report()
        
        logger.info("\n" + "="*70)
        logger.info("ANALYSIS COMPLETE!")
        logger.info("="*70)
        logger.info(f"\nResults saved to: {analyzer.output_dir}/")
        logger.info(f"  - 00_CRITICAL_hypothesis_test.png")
        logger.info(f"  - QUICK_ANALYSIS_SUMMARY.txt")
        logger.info("\nOpen the PNG file to see if your hypothesis is supported!")
        logger.info("="*70)
    else:
        logger.error("\n" + "="*70)
        logger.error("NO SUBJECTS SUCCESSFULLY PROCESSED!")
        logger.error("="*70)
        logger.error("\nPossible issues:")
        logger.error("1. Check that EDF files exist (S001R04.edf, S001R08.edf, etc.)")
        logger.error("2. Check that motor channels exist (C3, C4, Cz)")
        logger.error("3. Verify data is from PhysioNet Motor Movement/Imagery dataset")
        logger.error("="*70)


if __name__ == "__main__":
    main()