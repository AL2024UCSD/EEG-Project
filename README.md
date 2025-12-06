# EEG-Project : Imagined vs Actual Movement Analysis

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
