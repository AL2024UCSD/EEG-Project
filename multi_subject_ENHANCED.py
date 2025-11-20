from motor_imagery_robust_analyzer import GroupMotorImageryAnalyzer
from motor_imagery_minimal_preprocessor import MinimalGroupAnalyzer
from erd_diagnostic_tool import investigate_erd_issue, test_synthetic_erd

# First verify the calculation works
synthetic_result = test_synthetic_erd()

# Then check real data
result = investigate_erd_issue('S001', '/Users/shaheerkhan/Documents/EEG-Project/eeg-motor-movementimagery-dataset-1.0.0/files')
#analyzer = MinimalGroupAnalyzer()
#analyzer.process_all(max_subjects=10)  # Test on 10 first
# Initialize analyzer with your data path
analyzer = GroupMotorImageryAnalyzer(base_path='/Users/shaheerkhan/Documents/EEG-Project/eeg-motor-movementimagery-dataset-1.0.0/files')

# Process all 109 subjects
#analyzer.process_all_subjects()

# Create visualizations
#analyzer.create_group_visualizations()

# Save results
analyzer.save_results()