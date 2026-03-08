# EEG-Project : Imagined vs Actual Movement Analysis (Quarter 1)

**Author:** Shaheer Khan , Daniel Mansperger, Andrew Li  
**Date:** October 2025

## What This Does
This pipeline processes EEG data from multiple subjects to compare imagined vs actual movement patterns, providing group-level statistics and visualizations.
Analyzes EEG brain signals to compare actual hand/foot movements versus imagined movements. The script processes data from the PhysioNet Motor Movement/Imagery Database and generates visualizations showing brain activity patterns.

**Main finding:** Imagined movements produce similar brain patterns to actual movements, just with smaller amplitude (~60-70%).

## Overview

The pipeline extends the single-subject analysis to process all 109 subjects provided in the Physionet Motor Imagery dataset, computing averaged results and generating comprehensive group-level insights about motor imagery vs actual movement.

## Features

### Data Processing
- **Parallel Processing**: Processes multiple subjects concurrently for efficiency
- **Error Handling**: handles subjects with missing or corrupted data
- **Feature Extraction**: Extracts comprehensive EEG features from each subject:
  - Power spectral density (PSD)
  - Channel amplitudes
  - Time-frequency representations
  - Band powers (theta, alpha, beta)
  - Lateralization indices
  - Temporal dynamics

### Group Analysis
- **Statistical Averaging**: Computes mean and SEM across all subjects
- **Significance Testing**: Paired t-tests between conditions
- **Effect Size Calculation**: Cohen's d for clinical relevance
- **Variability Assessment**: Inter-subject variability analysis

### Visualizations
1. **Group PSD Comparison**: Averaged power spectra with confidence intervals
2. **Channel Amplitude Analysis**: RMS amplitudes across motor channels
3. **Time-Frequency Maps**: Group-averaged ERD/ERS patterns
4. **Band Power Statistics**: Statistical comparison with effect sizes
5. **Lateralization Patterns**: Preserved hemispheric dominance
6. **Subject Variability**: Distribution of responses and success rates

### Direct Python Usage

```python
from multi_subject_imagined_vs_actual_new import MultiSubjectAnalyzer

# Create analyzer
analyzer = MultiSubjectAnalyzer(base_path='set-eeg-motor-imagery-raw-data', output_dir='results')

# Run full pipeline
analyzer.run_full_pipeline(max_workers=4)
```

## Output Structure

```
group_imagined_vs_actual/
├── 01_group_psd_comparison.png          # Power spectral density comparison
├── 02_group_amplitude_comparison.png    # Channel-wise amplitudes
├── 03_group_timefreq_analysis.png       # Time-frequency representations
├── 04_group_band_power_statistics.png   # Band power statistics
├── 05_group_lateralization_analysis.png # Lateralization patterns
├── 06_subject_variability_analysis.png  # Inter-subject variability
├── GROUP_ANALYSIS_SUMMARY.txt           # Comprehensive text report
├── individual_subject_data.json         # Raw extracted features
└── failed_subjects.json                 # List of failed subjects (if any)
```

## Technical Details

### Processing Steps
1. **Data Loading**: EDF files with event markers
2. **Preprocessing**: 0.5-40 Hz bandpass filter
3. **Epoching**: -1 to 4s around movement cues
4. **Feature Extraction**: Multiple domains (spectral, temporal, spatial)
5. **Aggregation**: Subject-wise then group-wise averaging
6. **Statistics**: Parametric tests with multiple comparison awareness

### Requirements
- Python 3.7+
- MNE-Python
- NumPy, SciPy, Pandas
- Matplotlib, Seaborn
- tqdm (progress bars)
- 8+ GB RAM recommended
- Multi-core CPU for parallel processing








# EEG Project: MI BCI Compatibility Classifier Project (Quarter 2)

**Author:** Shaheer Khan, Daniel Mansperger, Andrew Li  
**Date:** Jan–Mar 2025

## Overview

Our Quarter 2 project focuses on building a **low-sample MI-BCI literacy screening pipeline**. This repository contains the **data preprocessing, feature extraction, and exploratory visualizations** on PhysioNet’s EEG Motor Movement/Imagery dataset. We extract compact, interpretable EEG-derived features (primarily resting-state and other low-burden summaries) and generate decoder-derived “ground truth” labels used downstream for model training.

> **Model training + web demo** are maintained in the companion repo:  
> https://github.com/Shaheer2492/BCI-Classifer

### What this repo produces (outputs)
- Subject-level feature tables (CSV), including the final **resting-state feature table**
- Decoder-derived ground-truth labels (JSON/CSV)
- Visualizations used for the paper/poster

---

## Features Extracted
- Power Spectral Entropy (PSE)
- Lempel–Ziv Complexity (LZC)
- Theta–Alpha Power Ratio (TAR)
- Alpha and Beta power (including sub-bands)
- Alpha asymmetry
- Alpha power variance
- Alpha/Beta power peaks + Individual Alpha Frequency (IAF)
- Aperiodic exponent (spectral slope)
- SMR baseline strength
- Interhemispheric coherence (C3–C4)

> Note: features are often subdivided by **channel**, **condition** (rest / imagined / real), and **frequency band** where applicable.

---

## Relevant coding files
- `resting_state_features_extraction.py`  
  Primary script for preprocessing and **resting-state feature extraction**.
- `test_resting_state_features_extraction.py`  
  Quick validation / smoke-test script for the extraction pipeline.
- `resting_state_eda.ipynb`  
  EDA and plots focused on resting-state features.
- `more_feat_and_vis.ipynb`  
  Additional feature experiments + visualization notebook, code eventually used to help generate several features in the `resting_state_features_extraction.py`.
- `generate_ground_truth_labels.py`  
  Generates decoder-derived ground-truth labels used for downstream literacy prediction, taken from the BCI-Classifier repo.
- Quarter 1 legacy (kept for reference / reuse):
  - `multi_subject_imagined_vs_actual_new.py`
  - `imagined_vs_actual_analysis_new.py`

---

## Relevant folders/csv files
- `eeg-motor-movementimagery-dataset-1.0.0/`  
  Local copy of the PhysioNet EEGBCI dataset (EDF + event files).
- `resting_state_features.csv`  
  **Final** resting-state feature table
- `eeg_features_column_descriptions.pdf`  
  Column glossary / feature descriptions.
- `json-from-bci-classifier/`  
  Copies of results JSONs from the training repo for plotting/reporting (e.g., model evaluation + classifier metadata).
- `outputs/`  
  Logs generated during extraction runs.
- `Visualizations/` and `Visualizations_ERDERS%/`  
  Saved figures used in the report/poster.

---

## How to run code

### General workflow (recommended order)
1. **Generate ground-truth labels**
   ```bash
   python generate_ground_truth_labels.py
   ```
   Outputs: label JSON/CSV artifacts (see `json-from-bci-classifier/` if you keep copies here).

2. **Extract resting-state features**
   ```bash
   python resting_state_features_extraction.py
   ```
   Outputs: `resting_state_features.csv` (and logs under `outputs/`).

3. **EDA / visualizations (optional)**
   - Open and run:
     - `resting_state_eda.ipynb`
     - `more_feat_and_vis.ipynb`

### Notes for notebooks
- In VSCode/Jupyter, use **Run All**.
- If export cells are commented out to avoid verbose logs, uncomment and rerun those cells.

### Notes for `.py` scripts
Use:
```bash
python <filename>.py
```

---

## Output Structure
Typical outputs generated/used by the Quarter 2 pipeline:

```
outputs/
└── feature_extraction.log

json-from-bci-classifier/
├── ground_truth_labels.json
├── early_trial_features_merged copy.json
├── model_evaluation copy.json
└── rf_classifier_metadata.json

Visualizations/
└── *.png

Visualizations_ERDERS%/
└── *.png

resting_state_features.csv
```

---

## Technical Details
### Processing Steps (high level)
1. **Data loading:** EDF files + event markers from the PhysioNet EEGBCI dataset.
2. **Preprocessing:** re-referencing and bandpass filtering (parameters documented in the extraction script).
3. **Resting-state feature extraction:** compute spectral, rhythm-strength, and stability metrics from baseline recordings.
4. **Ground truth (decoder-derived literacy):** CSP–LDA decoding performance used as the target label for the screening model (trained in the companion repo).
5. **EDA/visualization:** quality checks, distribution plots, and report figures.

### Reproducibility tips
- Keep the dataset folder out of Docker build context (use `.dockerignore`) to avoid huge images.
- If channel names include punctuation (e.g., `Cz..`), you may need to normalize channel labels in code when selecting channels.

---

## Quick Start (Docker)
```bash
docker build -t eeg-project .
docker run -it --rm \
  -v "$(pwd)":/workspace \
  eeg-project bash
```

Inside Docker:
```bash
python generate_ground_truth_labels.py
python resting_state_features_extraction.py
```

(Optional) start Jupyter:
```bash
jupyter lab --ip 0.0.0.0 --port 8888 --no-browser --allow-root
```

---

## Local Installation
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Run:
```bash
python generate_ground_truth_labels.py
python resting_state_features_extraction.py
```

---

## Requirements
- Python **3.11+** recommended
- MNE-Python
- NumPy, SciPy, Pandas
- Matplotlib, Seaborn
- scikit-learn, statsmodels
- tqdm (progress bars)
- 8+ GB RAM recommended (more helps when running across many subjects)
- Multi-core CPU recommended for faster preprocessing/feature extraction
