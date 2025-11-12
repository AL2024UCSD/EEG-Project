#!/usr/bin/env python3
"""
Run multi-subject imagined vs actual movement analysis
"""

import sys
import argparse
from pathlib import Path
from multi_subject_imagined_vs_actual import MultiSubjectAnalyzer

def main():
    parser = argparse.ArgumentParser(description='Run multi-subject EEG analysis')
    parser.add_argument('--workers', type=int, default=4,
                       help='Number of parallel workers (default: 4)')
    parser.add_argument('--output', type=str, default='group_imagined_vs_actual',
                       help='Output directory name (default: group_imagined_vs_actual)')
    parser.add_argument('--data-path', type=str, default='raw_data',
                       help='Path to raw data directory (default: raw_data)')
    
    args = parser.parse_args()
    
    print(f"Starting multi-subject analysis with {args.workers} workers...")
    print(f"Data path: {args.data_path}")
    print(f"Output directory: {args.output}")
    
    # Create analyzer instance
    analyzer = MultiSubjectAnalyzer(base_path=args.data_path, output_dir=args.output)
    
    # Run the pipeline
    analyzer.run_full_pipeline(max_workers=args.workers)
    
    print("\nAnalysis complete!")
    print(f"Results saved to: {args.output}/")

if __name__ == "__main__":
    main()
