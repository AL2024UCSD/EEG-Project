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
**Author:** Shaheer Khan , Daniel Mansperger, Andrew Li  
**Date:** Jan-Mar 2025

## Overview
Our project focuses on the creation of a motor imagery brain-computer interface literacy classifier that can predict, given minimal EEG data and recordings, whether or not a person will be able to use a motor imagery BCI. Using multiple python and jupyter notebook files, along with reusing code and utilizing information garnered from our previous project (Imagined vs Actual Movement Analysis from Quarter 1), we preprocessed data from Physionet's EEG Motor Movement/Imagery Dataset, extracted various features we believed to be useful in predicting for MI BCI compatibility, and created a combined dataset with all of our features. We also used MetaBCI decoders to generate the ground truth labels (roughly how BCI-compatible they are) for each test subject. 

## Features Extracted
- Power Spectral Entropy (PSE)
- Lempel-Ziv Complexity (LZC)
- Theta-Alpha Power Ratio (TAR)
- Alpha and Beta power
- Alpha asymmetry
- Alpha Power Variance
- Alpha and Beta power peaks
- Aperiodic Exponent
- SMR baseline strength 
- individual alpha frequency (IAF)
- interhemispheric coherence

## Relevant coding files
- imagined_vs_actual_analysis_new.py (some functions reused in andrew_notebook.py)
- andrew_notebook.ipynb (used to initially extract Power Spectral Entropy (PSE), Lempel-Ziv Complexity (LZC), and Theta-Alpha Power Ratio (TAR), also used for some EDA and visualizations on those features)
- daniel_final.py (used to extract other features, modify the andrew_notebook features for easier implementation, and to create the combined features dataset (eeg_features.csv))
- EDA_daniel.ipynb (used for more EDA on features)
- test_load.py (extracts the MetaBCI labels, creates the physionetmi_subject_mean_accuracy.csv file)


## Relevant folders/csv files
- eeg-motor-movementimagery-dataset-1.0.0 (contains EEG recordings from 109 users, used for our feature extraction and model testing)
- eeg_features.csv (contains ALL of our features for each subject from the Physionet dataset)
- eeg_features_andrew and eeg_features_andrew_compact.csv (contains the features extracted from the andrew_notebook.ipynb file, the compact file removes the by-trial division found in the non-compact file)
- physionetmi_subject_mean_accuracy.csv (contains the "compatibility scores" for each subject from the Physionet dataset)

It should be noted that all features are divided into subfeatures via electrode, MI vs actual movement vs resting, frequency, and/or etc.

## How to run code

- Clone the repo, build the docker file, run the dockerfile (remember to have docker desktop open), alternatively run the requirements.txt file

- For the notebooks, all that is needed is to run all the cells in order (for example, if using VSCode, just hit the "Run All" button on the top center of the interface). For andrew_notebook.ipynb, if you wish to remake the csv files, just remove the single quotes from beginning and end of the 2 csv filebuilding sections. 

- for the py files, use: 
'''
python (python file name here).py
'''

### Quick Start (Docker)
```bash
docker build -t eeg-project .
docker run -it --rm -v $(pwd):/workspace eeg-project bash
from multi_subject_imagined_vs_actual_new import MultiSubjectAnalyzer
python test_load.py
```
(run rest of py files)

### Local Installation
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
from multi_subject_imagined_vs_actual_new import MultiSubjectAnalyzer
python test_load.py
```
(run rest of py files)

### Requirements
- Python 3.7+
- MNE-Python
- NumPy, SciPy, Pandas
- Matplotlib, Seaborn
- tqdm (progress bars)
- 8+ GB RAM recommended
- Multi-core CPU for parallel processing
