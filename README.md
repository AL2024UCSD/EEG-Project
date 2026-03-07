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
Our project focuses on creating a **low-sample MI-BCI literacy screening model** that predicts, from minimal EEG recordings, whether a person is likely to be compatible with a motor imagery (MI) BCI. In this repo, we preprocess EEG from PhysioNet’s EEG Motor Movement/Imagery Dataset, extract interpretable EEG-derived features, and produce subject-level feature tables used downstream for model training. We also generate decoder-derived “ground truth” literacy labels using CSP–LDA (MetaBCI-style) decoders.

> **Model training + web demo** are maintained in the companion repo:
> - https://github.com/Shaheer2492/BCI-Classifer

### What this repo produces (outputs)
- Feature tables (CSV) per subject (resting + early-trial + engineered features)
- Various visualizations used to better understand select features, along with other visuals meant for the project poster and paper
- Ground-truth subject “compatibility scores” (decoder accuracy CSV)

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
- `imagined_vs_actual_analysis_new.py` (Quarter 1 legacy; some functions reused)
- `andrew_notebook_updated.ipynb`  
  Extracts PSE/LZC/TAR features and exports:
  - `eeg_features_andrew_new.csv`
  - `eeg_features_andrew_compact_new.csv`
- `daniel_final.py`  
  Extracts rhythm/SMR/coherence/aperiodic-style features and can export combined feature tables (filenames depend on current settings in the script).
- `EDA_daniel.ipynb` (feature EDA + plots)
- `test_load.py`  
  Generates decoder-derived labels and writes `physionetmi_subject_mean_accuracy.csv`

---

## Relevant folders/csv files
- `eeg-motor-movementimagery-dataset-1.0.0/`  
  Contains EEG recordings from 109 subjects (EDF + event files).
- `physionetmi_subject_mean_accuracy.csv`  
  Subject-level decoder accuracy (ground-truth “literacy score”).
- Feature tables (latest versions; filenames may change over time):
  - `eeg_features_v4.csv` (full merged/combined features per subject; newest version)
  - `eeg_features_andrew_new.csv` and `eeg_features_andrew_compact_new.csv`  
    (“compact” = one row per subject; non-compact may include trial-level breakdowns depending on notebook settings)

---

## How to run code

### General workflow (recommended order)
1. **Generate ground-truth labels**
   ```bash
   python test_load.py
   ```
   Output: `physionetmi_subject_mean_accuracy.csv`

2. **Extract features**
   - Run `daniel_final.py`:
     ```bash
     python daniel_final.py
     ```
   - Run `andrew_notebook_updated.ipynb` (Run All).  
     If you want to recreate the CSVs, ensure the export cells are **uncommented**.

3. **EDA / plots (optional)**
   - Open and run `EDA_daniel.ipynb`

### Notes for notebooks
- In VSCode/Jupyter, you can use **Run All**.
- If CSV-writing cells are commented out to avoid long logs, just uncomment them and rerun those cells.

### Notes for `.py` scripts
Use:
```bash
python <filename>.py
```

---

## Quick Start (Docker)

> **Tip:** If your dataset folder is inside the repo, add a `.dockerignore` so Docker doesn’t copy the entire dataset into the build context.

### Build
```bash
docker build -t eeg-project .
```

### Run (mount the repo into the container)
```bash
docker run -it --rm \
  -v "$(pwd)":/workspace \
  eeg-project bash
```

### Inside Docker: run scripts
```bash
python test_load.py
python daniel_final.py
```
Then open and run notebooks with Jupyter if desired:
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

Run scripts:
```bash
python test_load.py
python daniel_final.py
```

---

## Requirements
- Python **3.11+** recommended (older versions may work but are less tested)
- MNE-Python
- NumPy, SciPy, Pandas
- Matplotlib, Seaborn
- json
- scikit-learn, statsmodels
- tqdm (progress bars)
- 8+ GB RAM recommended (more helps when running across many subjects)
- Multi-core CPU recommended for faster preprocessing/feature extraction
