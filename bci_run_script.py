from alphabandpower import BCIFeatureExtractor  # replace with actual filename
import numpy as np

# Create an instance
extractor = BCIFeatureExtractor(subject_id='S001')

# Load and preprocess the data
extractor.load_and_preprocess_data()

# Print the time array from one of the raw objects
# Let's use the eyes_closed baseline as an example
if 'baseline_eyes_closed' in extractor.data:
    raw = extractor.data['baseline_eyes_closed']['raw']
    
    for i in raw.times:
        print(i)
    
    print(f"\nTotal duration: {raw.times[-1]:.2f} seconds")
    print(f"Sampling frequency: {raw.info['sfreq']} Hz")
    print(f"Number of samples: {len(raw.times)}")
    print(f"Number of channels: {len(raw.ch_names)}")
    print(f"Channel names: {raw.ch_names}")