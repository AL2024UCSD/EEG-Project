"""
Simple runner script for imagined vs actual movement analysis
"""

import logging
import sys
from pathlib import Path

# Add the current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from imagined_vs_actual_analysis import ImaginedVsActualAnalyzer

def main():
    """Run the analysis."""
    print("\n" + "="*70)
    print("IMAGINED VS ACTUAL MOVEMENT ANALYSIS")
    print("="*70)
    print("\nThis analysis will:")
    print("1. Load baseline EEG data (eyes open/closed)")
    print("2. Load EEG data for both imagined and actual movements")
    print("3. Compare brain activity patterns between all conditions")
    print("4. Generate high-resolution visualizations")
    print("5. Perform statistical comparisons")
    print("6. Create a comprehensive summary report")
    print("\nProcessing subject S001...")
    print("="*70 + "\n")
    
    try:
        # Create analyzer
        analyzer = ImaginedVsActualAnalyzer(subject_id='S001', base_path='eeg-motor-movementimagery-dataset-1.0.0')
        
        # Run full analysis
        analyzer.run_full_analysis()
        
        print("\n Analysis completed successfully!")
        print(f" Results saved in: S001_imagined_vs_actual/")
        print("\n Generated visualizations:")
        print("   - 01_overview_comparison.png (with baseline reference)")
        print("   - 02_lateralization_analysis.png") 
        print("   - 03_complexity_analysis.png")
        print("   - 04_temporal_dynamics.png")
        print("   - ANALYSIS_SUMMARY.txt")
        print("\n🔬 Analysis now includes:")
        print("   - Baseline (eyes open/closed) comparisons")
        print("   - Rest → Imagined → Actual movement continuum")
        print("   - Enhanced statistical analysis")
        
    except Exception as e:
        print(f"\n(X) Error during analysis: {e}")
        logging.error(f"Analysis failed: {e}", exc_info=True)
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
