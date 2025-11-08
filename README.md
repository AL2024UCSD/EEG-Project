# EEG Motor Imagery Analysis

**Author:** Shaheer Khan , Daniel Mansperger, Andrew Li  
**Date:** October 2025

## What This Does

Analyzes EEG brain signals to compare actual hand/foot movements versus imagined movements. The script processes data from the PhysioNet Motor Movement/Imagery Database and generates visualizations showing brain activity patterns.

**Main finding:** Imagined movements produce similar brain patterns to actual movements, just with smaller amplitude (~60-70%).

## Requirements

### Python Packages
```bash
pip install numpy matplotlib seaborn mne scipy pandas
```

### Data
Download EEG data from PhysioNet: https://physionet.org/content/eegmmidb/1.0.0/

Place data in this structure:
```
raw_data/
  S001/
    S001R01.edf  # Eyes open baseline
    S001R02.edf  # Eyes closed baseline
    S001R03.edf  # Left/Right fist (actual)
    S001R04.edf  # Left/Right fist (imagined)
    ... (through S001R14.edf)
```

## Usage

```bash
python imagined_vs_actual_analysis.py
```

Runtime: ~30 seconds - 1 min

## Outputs

All files saved to `S001_imagined_vs_actual/`:

### 1. `01_overview_comparison.png`
- Power spectrum showing frequency content (with baseline comparison)
- Channel amplitudes across motor cortex
- Time-frequency maps (ERD/ERS)
- Statistical comparison table

### 2. `02_lateralization_analysis.png`
- Left vs right hand activity
- Shows contralateral motor control (left hand = right brain)
- Lateralization index plots

### 3. `03_complexity_analysis.png`
- Single limb vs dual limb movements
- Beta power comparison
- Inter-hemispheric coherence

### 4. `04_temporal_dynamics.png`
- Brain activity across three time windows:
  - Pre-movement (-1 to 0s)
  - Early movement (0 to 1s)
  - Sustained movement (1 to 3s)
- Theta, alpha, and beta band power

### 5. `ANALYSIS_SUMMARY.txt`
- Text report with key metrics
- Statistical results
- Percentage changes from baseline

## Quick Customization

**Analyze different subject:**
```python
analyzer = ImaginedVsActualAnalyzer(subject_id='S002')
```

**Change channels:**
```python
self.channels = ['C3', 'C4', 'Cz']  # Just primary motor cortex
```

**Adjust filters:**
```python
raw.filter(0.5, 40, ...)  # 0.5-40 Hz bandpass
```

## Troubleshooting

**"FileNotFoundError"** → Check data path is `raw_data/S001/`

**"No motor channels found"** → Update `self.channels` list to match your data

**Memory errors** → Close other programs, reduce number of epochs

**MNE import error** → Try `conda install -c conda-forge mne`

## References

- **Dataset:** Schalk et al. (2004), PhysioNet EEG Motor Movement/Imagery Database
- **Processing:** MNE-Python (Gramfort et al., 2013)
- **Theory:** Pfurtscheller & Lopes da Silva (1999), Event-related EEG synchronization














# EEG-Project : Imagined vs Actual Movement Analysis Pipeline (Coming Soon) 

This pipeline processes EEG data from multiple subjects to compare imagined vs actual movement patterns, providing group-level statistics and visualizations.

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
from multi_subject_imagined_vs_actual import MultiSubjectAnalyzer

# Create analyzer
analyzer = MultiSubjectAnalyzer(base_path='raw_data', output_dir='results')

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
