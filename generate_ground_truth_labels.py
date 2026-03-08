"""
Generate Ground Truth Labels
--------------------------------------
Run MetaBCI CSP+LDA decoder on PhysionetMI dataset to extract
per-subject accuracy scores for all 109 subjects.

Uses stratified 5-fold cross-validation within each subject.
Implements MetaBCI's CSP spatial filter with sklearn's LDA classifier.
"""

import json
import os
import warnings
from datetime import datetime
from pathlib import Path

import mne
import numpy as np
from mne.datasets import eegbci
from mne.io import concatenate_raws, read_raw_edf
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
# Use MetaBCI's CSP decoder
from metabci.brainda.algorithms.decomposition import CSP
from tqdm import tqdm

warnings.filterwarnings('ignore')
mne.set_log_level('ERROR')


class PhysionetMIGroundTruthGenerator:
    """Generate ground truth labels using MetaBCI CSP+LDA on PhysionetMI dataset.

    Uses MetaBCI's CSP spatial filter for motor imagery feature extraction
    and sklearn's LDA for classification.
    """

    def __init__(self, n_subjects=109, n_folds=5, random_state=42):
        """
        Initialize the ground truth generator.

        Parameters
        ----------
        n_subjects : int
            Number of subjects to process (default: 109)
        n_folds : int
            Number of folds for cross-validation (default: 5)
        random_state : int
            Random seed for reproducibility
        """
        self.n_subjects = n_subjects
        self.n_folds = n_folds
        self.random_state = random_state

        # PhysionetMI motor imagery runs
        # Task 1: Left vs Right Hand (Runs 4, 8, 12)
        # Task 2: Fists vs Feet (Runs 6, 10, 14)
        self.tasks = {
            'left_right': [4, 8, 12],
            'hands_feet': [6, 10, 14]
        }

        # EEG parameters
        self.tmin, self.tmax = 0.5, 4.0  # Time window for MI (optimized)
        self.fmin, self.fmax = 8.0, 13.0  # Frequency band (Mu band, optimized)

        # CSP parameters
        self.n_components = 8  # Number of CSP components (optimized)

    def load_subject_data(self, subject_id, runs):
        """
        Load and preprocess EEG data for a single subject.

        Parameters
        ----------
        subject_id : int
            Subject ID (1-109)
        runs : list
            List of run indices to load

        Returns
        -------
        X : ndarray, shape (n_trials, n_channels, n_times)
            EEG data
        y : ndarray, shape (n_trials,)
            Labels
        run_info : list
            Information about which run each trial belongs to
        """
        raw_fnames = []

        # Load motor imagery runs for this subject
        for run in runs:
            try:
                fnames = eegbci.load_data(subject_id, run, update_path=True)
                raw_fnames.extend(fnames)
            except Exception as e:
                print(f"  Warning: Could not load run {run} for subject {subject_id}: {e}")
                continue

        if not raw_fnames:
            raise ValueError(f"No valid data files found for subject {subject_id}")

        # Read and concatenate all runs
        raws = [read_raw_edf(fname, preload=True, verbose=False) for fname in raw_fnames]
        raw = concatenate_raws(raws)

        # Standardize channel names (PhysioNet EDFs have trailing dots like "C3.", "Cz.")
        eegbci.standardize(raw)

        # Apply bandpass filter
        raw.filter(self.fmin, self.fmax, fir_design='firwin', verbose=False)

        # Extract events
        events, event_id = mne.events_from_annotations(raw, verbose=False)

        # For PhysionetMI:
        # T1 = left fist (or both fists for runs with hands vs feet)
        # T2 = right fist (or both feet for runs with hands vs feet)
        # We'll use T1 vs T2 as our two-class problem

        # Map events to binary labels (0 vs 1)
        # T1 (left/hands) -> 0, T2 (right/feet) -> 1
        event_id_binary = {'T1': 0, 'T2': 1}

        # Find which event IDs correspond to T1 and T2
        t1_id = event_id.get('T1', None)
        t2_id = event_id.get('T2', None)

        if t1_id is None or t2_id is None:
            raise ValueError(f"Could not find T1/T2 events for subject {subject_id}")

        # Keep only T1 and T2 events
        mask = np.isin(events[:, 2], [t1_id, t2_id])
        events = events[mask]

        # Create binary labels
        labels = np.where(events[:, 2] == t1_id, 0, 1)

        # Pick only EEG channels (excluding EOG)
        picks = mne.pick_types(raw.info, meg=False, eeg=True, eog=False, stim=False)

        # Epoch the data
        epochs = mne.Epochs(
            raw, events,
            tmin=self.tmin, tmax=self.tmax,
            proj=True, picks=picks,
            baseline=None, preload=True,
            verbose=False
        )

        # Get data as numpy array
        X = epochs.get_data()  # (n_trials, n_channels, n_times)
        
        # FIX: Derive labels from epochs.events to handle dropped epochs
        # This prevents mismatch between X and y length
        y = np.where(epochs.events[:, 2] == t1_id, 0, 1)

        # Store run information (which run each trial came from)
        run_info = self._get_run_info(events, runs)

        return X, y, run_info

    def _get_run_info(self, events, runs):
        """Get information about which run each trial belongs to."""
        # Simplified: just return run indices
        # In a more sophisticated version, we'd track exact run boundaries
        n_trials = len(events)
        n_runs = len(runs)
        if n_runs > 0:
            trials_per_run = n_trials // n_runs
        else:
            trials_per_run = n_trials # Fallback

        run_info = []
        for i in range(n_trials):
            if n_runs > 0 and trials_per_run > 0:
                run_idx = min(i // trials_per_run, n_runs - 1)
                run_info.append(runs[run_idx])
            else:
                run_info.append(0)

        return run_info

    def build_decoder(self):
        """
        Build MetaBCI CSP + LDA decoder pipeline.

        Uses MetaBCI's CSP spatial filter for feature extraction and
        sklearn's LDA for classification.

        Returns
        -------
        pipeline : sklearn.pipeline.Pipeline
            MetaBCI CSP+LDA decoder pipeline
        """
        # MetaBCI CSP+LDA pipeline for motor imagery classification
        pipeline = Pipeline([
            ('CSP', CSP(n_components=self.n_components)),  # MetaBCI's CSP
            ('Scaler', StandardScaler()),  # Feature scaling
            ('LDA', LinearDiscriminantAnalysis(
                solver='lsqr',    # Suitable for high-dimensional data
                shrinkage=0.5     # Regularization (optimized)
            ))
        ])

        return pipeline

    def evaluate_subject(self, subject_id, runs):
        """
        Evaluate decoder performance for a single subject using cross-validation.

        Parameters
        ----------
        subject_id : int
            Subject ID (1-109)
        runs : list
            List of run indices

        Returns
        -------
        results : dict
            Dictionary containing performance metrics
        """
        try:
            # Load data
            X, y, run_info = self.load_subject_data(subject_id, runs)

            # Check if we have enough trials
            if len(X) < self.n_folds:
                print(f"  Warning: Subject {subject_id} has too few trials ({len(X)})")
                return None

            # Check class balance
            unique, counts = np.unique(y, return_counts=True)
            if len(unique) < 2:
                print(f"  Warning: Subject {subject_id} has only one class")
                return None

            # Initialize MetaBCI-style cross-validation
            cv = StratifiedKFold(n_splits=self.n_folds,
                                shuffle=True,
                                random_state=self.random_state)

            # Store results for each fold
            fold_accuracies = []
            fold_predictions = []
            fold_true_labels = []

            # Perform cross-validation (MetaBCI-style within-subject evaluation)
            for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X, y)):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]

                # Build and train MetaBCI-style decoder
                decoder = self.build_decoder()
                decoder.fit(X_train, y_train)

                # Predict on test set
                y_pred = decoder.predict(X_test)

                # Calculate accuracy
                accuracy = np.mean(y_pred == y_test)
                fold_accuracies.append(accuracy)
                fold_predictions.extend(y_pred.tolist())
                fold_true_labels.extend(y_test.tolist())

            # Calculate overall metrics
            overall_accuracy = np.mean(fold_accuracies)
            std_accuracy = np.std(fold_accuracies)

            # Calculate per-class accuracy
            fold_predictions = np.array(fold_predictions)
            fold_true_labels = np.array(fold_true_labels)

            class_accuracies = {}
            for class_id in unique:
                class_mask = fold_true_labels == class_id
                class_acc = np.mean(fold_predictions[class_mask] == fold_true_labels[class_mask])
                class_accuracies[int(class_id)] = float(class_acc)

            # Compile results
            results = {
                'subject_id': int(subject_id),
                'n_trials': int(len(X)),
                'n_channels': int(X.shape[1]),
                'class_distribution': {int(c): int(cnt) for c, cnt in zip(unique, counts)},
                'accuracy': float(overall_accuracy),
                'accuracy_std': float(std_accuracy),
                'fold_accuracies': [float(acc) for acc in fold_accuracies],
                'class_accuracies': class_accuracies,
                'cv_folds': self.n_folds,
                'runs_used': runs,
                'success': True
            }

            return results

        except Exception as e:
            print(f"  Error processing subject {subject_id}: {e}")
            return {
                'subject_id': int(subject_id),
                'success': False,
                'error': str(e)
            }

    def generate_all_labels(self, output_path='json-from-bci-classifier/ground_truth_labels.json'):
        """
        Generate ground truth labels for all subjects across both tasks.

        Parameters
        ----------
        output_path : str
            Path to save results JSON file

        Returns
        -------
        output : dict
            Full output dictionary with results for both tasks
        """
        print(f"Generating ground truth labels for {self.n_subjects} subjects...")
        print(f"Using {self.n_folds}-fold stratified cross-validation")
        print(f"CSP components: {self.n_components}")
        
        task_results = {}
        
        for task_name, runs in self.tasks.items():
            print("\n" + "=" * 60)
            print(f"TASK: {task_name} (Runs: {runs})")
            print("=" * 60)
            
            all_results = []
            
            # Process each subject
            for subject_id in tqdm(range(1, self.n_subjects + 1), desc=f"Processing {task_name}"):
                results = self.evaluate_subject(subject_id, runs)

                if results is not None:
                    # Rename metric for clarity if needed, but keeping standard
                    all_results.append(results)

                    if results['success']:
                        pass # Less verbose for individual lines to keep log clean
                    else:
                        print(f"  Subject {subject_id} ✗ Failed: {results.get('error', 'Unknown error')}")

            # Calculate summary statistics for this task
            successful_subjects = [r for r in all_results if r['success']]
            accuracies = [r['accuracy'] for r in successful_subjects]
            
            mean_acc = float(np.mean(accuracies)) if accuracies else 0.0
            std_acc = float(np.std(accuracies)) if accuracies else 0.0
            
            print(f"\n{task_name} Summary:")
            print(f"  Successful: {len(successful_subjects)}/{self.n_subjects}")
            print(f"  Mean Accuracy: {mean_acc:.3f} ± {std_acc:.3f}")
            
            summary = {
                'total_subjects': self.n_subjects,
                'successful_subjects': len(successful_subjects),
                'failed_subjects': self.n_subjects - len(successful_subjects),
                'mean_accuracy': mean_acc,
                'std_accuracy': std_acc,
                'min_accuracy': float(np.min(accuracies)) if accuracies else 0.0,
                'max_accuracy': float(np.max(accuracies)) if accuracies else 0.0,
                'median_accuracy': float(np.median(accuracies)) if accuracies else 0.0,
            }

            task_results[task_name] = {
                'runs': runs,
                'summary': summary,
                'subjects': all_results
            }

            # SAVE INTERMEDIATE RESULTS
            # Compile current output with available tasks
            current_output = {
                'metadata': {
                    'generated_at': datetime.now().isoformat(),
                    'n_subjects': self.n_subjects,
                    'n_folds': self.n_folds,
                    'tasks_defined': self.tasks,
                    'n_csp_components': self.n_components,
                    'frequency_band': [self.fmin, self.fmax],
                    'time_window': [self.tmin, self.tmax],
                    'decoder': 'MetaBCI CSP + sklearn LDA',
                    'cv_strategy': f'{self.n_folds}-fold stratified cross-validation',
                },
                'tasks': task_results
            }

            # Save to JSON
            output_path_obj = Path(output_path)
            output_path_obj.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path_obj, 'w') as f:
                json.dump(current_output, f, indent=2)
            
            print(f"  Saved intermediate results for '{task_name}' to {output_path_obj.absolute()}")

        # Compile final output
        output = {
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'n_subjects': self.n_subjects,
                'n_folds': self.n_folds,
                'tasks_defined': self.tasks,
                'n_csp_components': self.n_components,
                'frequency_band': [self.fmin, self.fmax],
                'time_window': [self.tmin, self.tmax],
                'decoder': 'MetaBCI CSP + sklearn LDA',
                'cv_strategy': f'{self.n_folds}-fold stratified cross-validation',
            },
            'tasks': task_results
        }

        # Save to JSON
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)

        print("\n" + "=" * 60)
        print("OVERALL SUMMARY")
        print("=" * 60)
        for task_name, data in task_results.items():
            summ = data['summary']
            print(f"{task_name}: {summ['mean_accuracy']:.3f} (n={summ['successful_subjects']})")
            
        print(f"\nResults saved to: {output_path.absolute()}")

        return output


def main():
    """Main execution function."""
    # Create generator
    generator = PhysionetMIGroundTruthGenerator(
        n_subjects=109,
        n_folds=5,
        random_state=42
    )

    # Generate labels
    results = generator.generate_all_labels(
        output_path='json-from-bci-classifier/ground_truth_labels.json'
    )

    return results


if __name__ == '__main__':
    main()
