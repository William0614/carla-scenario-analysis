#!/usr/bin/env python3
"""
Main CLI interface for CARLA Scenario Similarity Analysis

Provides a clean command-line interface for running similarity analysis
with the consolidated framework.
"""

import os
import sys
import argparse
import json
from datetime import datetime

# Add the current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from carla_similarity import (
    FeatureExtractor,
    BasicGroundTruth,
    MultiDimensionalGroundTruth,
    SimilarityEvaluator
)


def main():
    parser = argparse.ArgumentParser(
        description='CARLA Scenario Similarity Analysis Framework',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract features from log files
  python main.py extract-features --log-dir log_files --output features.json
  
  # Generate basic ground truth
  python main.py basic-gt --features features.json --output basic_gt.json
  
  # Generate multi-dimensional ground truth  
  python main.py multi-gt --features features.json --output multi_gt.json
  
  # Evaluate all similarity metrics
  python main.py evaluate --features features.json --ground-truth multi_gt.json --output results.json
  
  # Full pipeline
  python main.py full-analysis --log-dir log_files --output-dir results/
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Extract features command
    extract_parser = subparsers.add_parser('extract-features', help='Extract 37-dimensional features from log files')
    extract_parser.add_argument('--log-dir', required=True, help='Directory containing CARLA log files')
    extract_parser.add_argument('--output', required=True, help='Output JSON file for features')
    
    # Basic ground truth command
    basic_gt_parser = subparsers.add_parser('basic-gt', help='Generate basic filename-based ground truth')
    basic_gt_parser.add_argument('--features', required=True, help='Features JSON file')
    basic_gt_parser.add_argument('--output', required=True, help='Output JSON file for ground truth')
    
    # Multi-dimensional ground truth command
    multi_gt_parser = subparsers.add_parser('multi-gt', help='Generate multi-dimensional ground truth')
    multi_gt_parser.add_argument('--features', required=True, help='Features JSON file')
    multi_gt_parser.add_argument('--output', required=True, help='Output JSON file for ground truth')
    multi_gt_parser.add_argument('--threshold', type=float, default=0.6, help='Similarity threshold (default: 0.6)')
    
    # Evaluate metrics command
    evaluate_parser = subparsers.add_parser('evaluate', help='Evaluate similarity metrics against ground truth')
    evaluate_parser.add_argument('--features', required=True, help='Features JSON file')
    evaluate_parser.add_argument('--ground-truth', required=True, help='Ground truth JSON file')
    evaluate_parser.add_argument('--output', required=True, help='Output JSON file for evaluation results')
    evaluate_parser.add_argument('--threshold', type=float, default=0.6, help='Similarity threshold (default: 0.6)')
    
    # Full analysis command
    full_parser = subparsers.add_parser('full-analysis', help='Run complete analysis pipeline')
    full_parser.add_argument('--log-dir', required=True, help='Directory containing CARLA log files')
    full_parser.add_argument('--output-dir', required=True, help='Output directory for all results')
    full_parser.add_argument('--threshold', type=float, default=0.6, help='Similarity threshold (default: 0.6)')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Set up CARLA environment
    if 'SCENARIO_RUNNER_ROOT' not in os.environ:
        print("Warning: SCENARIO_RUNNER_ROOT environment variable not set")
        print("Please set it to your CARLA data directory for proper functionality")
    
    try:
        if args.command == 'extract-features':
            extract_features(args)
        elif args.command == 'basic-gt':
            generate_basic_ground_truth(args)
        elif args.command == 'multi-gt':
            generate_multi_ground_truth(args)
        elif args.command == 'evaluate':
            evaluate_metrics(args)
        elif args.command == 'full-analysis':
            full_analysis(args)
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


def extract_features(args):
    """Extract 37-dimensional features from log files."""
    print(f"Extracting features from log files in {args.log_dir}...")
    
    extractor = FeatureExtractor()
    
    # Get all log files
    import glob
    log_files = glob.glob(os.path.join(args.log_dir, "*.log"))
    
    if not log_files:
        raise ValueError(f"No log files found in {args.log_dir}")
    
    features_dict = {}
    successful = 0
    
    for log_file in log_files:
        filename = os.path.basename(log_file)
        print(f"  Processing {filename}...")
        
        features = extractor.extract_features(f"log_files/{filename}")
        if features is not None:
            features_dict[filename] = features
            successful += 1
        else:
            print(f"    Failed to extract features from {filename}")
    
    # Save results
    with open(args.output, 'w') as f:
        json.dump(features_dict, f, indent=2, default=str)
    
    print(f"Feature extraction complete: {successful}/{len(log_files)} files processed")
    print(f"Features saved to {args.output}")


def generate_basic_ground_truth(args):
    """Generate basic filename-based ground truth."""
    print(f"Generating basic ground truth from {args.features}...")
    
    # Load features
    with open(args.features, 'r') as f:
        features_dict = json.load(f)
    
    scenarios = list(features_dict.keys())
    
    # Generate ground truth
    basic_gt = BasicGroundTruth()
    ground_truth = basic_gt.generate_ground_truth(scenarios)
    
    # Save results
    with open(args.output, 'w') as f:
        json.dump(ground_truth, f, indent=2, default=str)
    
    similar_pairs = sum(1 for gt in ground_truth.values() if isinstance(gt, dict) and gt.get('similar', False))
    total_pairs = len(scenarios) * (len(scenarios) - 1) // 2
    
    print(f"Basic ground truth generated: {similar_pairs}/{total_pairs} similar pairs")
    print(f"Ground truth saved to {args.output}")


def generate_multi_ground_truth(args):
    """Generate multi-dimensional ground truth."""
    print(f"Generating multi-dimensional ground truth from {args.features}...")
    
    # Load features
    with open(args.features, 'r') as f:
        features_dict = json.load(f)
    
    # Generate ground truth
    multi_gt = MultiDimensionalGroundTruth(similarity_threshold=args.threshold)
    ground_truth = multi_gt.generate_ground_truth(features_dict)
    
    # Save results
    with open(args.output, 'w') as f:
        json.dump(ground_truth, f, indent=2, default=str)
    
    if '_metadata' in ground_truth:
        metadata = ground_truth['_metadata']
        print(f"Multi-dimensional ground truth generated:")
        print(f"  Similar pairs: {metadata['similar_pairs']}/{metadata['total_comparisons']}")
        print(f"  Similarity rate: {metadata['similarity_rate']:.1%}")
    
    print(f"Ground truth saved to {args.output}")


def evaluate_metrics(args):
    """Evaluate similarity metrics against ground truth."""
    print(f"Evaluating similarity metrics...")
    
    # Load features and ground truth
    with open(args.features, 'r') as f:
        features_dict = json.load(f)
    
    with open(args.ground_truth, 'r') as f:
        ground_truth = json.load(f)
    
    # Remove metadata from ground truth if present
    if '_metadata' in ground_truth:
        del ground_truth['_metadata']
    
    # Run evaluation
    evaluator = SimilarityEvaluator(similarity_threshold=args.threshold)
    results = evaluator.evaluate_all_metrics(features_dict, ground_truth)
    
    # Save results
    evaluator.save_results(results, args.output)
    
    # Print summary
    evaluator.print_summary(results)


def full_analysis(args):
    """Run complete analysis pipeline."""
    print("Running full CARLA scenario similarity analysis...")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Step 1: Extract features
    features_file = os.path.join(args.output_dir, f"features_{timestamp}.json")
    print("\n1. Extracting features...")
    
    class MockArgs:
        def __init__(self, log_dir, output):
            self.log_dir = log_dir
            self.output = output
    
    extract_features(MockArgs(args.log_dir, features_file))
    
    # Step 2: Generate both ground truths
    basic_gt_file = os.path.join(args.output_dir, f"basic_ground_truth_{timestamp}.json")
    multi_gt_file = os.path.join(args.output_dir, f"multi_ground_truth_{timestamp}.json")
    
    print("\n2. Generating basic ground truth...")
    class MockGTArgs:
        def __init__(self, features, output, threshold=None):
            self.features = features
            self.output = output
            if threshold:
                self.threshold = threshold
    
    generate_basic_ground_truth(MockGTArgs(features_file, basic_gt_file))
    
    print("\n3. Generating multi-dimensional ground truth...")
    generate_multi_ground_truth(MockGTArgs(features_file, multi_gt_file, args.threshold))
    
    # Step 3: Evaluate against both ground truths
    basic_results_file = os.path.join(args.output_dir, f"evaluation_basic_gt_{timestamp}.json")
    multi_results_file = os.path.join(args.output_dir, f"evaluation_multi_gt_{timestamp}.json")
    
    print("\n4. Evaluating metrics against basic ground truth...")
    class MockEvalArgs:
        def __init__(self, features, ground_truth, output, threshold):
            self.features = features
            self.ground_truth = ground_truth
            self.output = output
            self.threshold = threshold
    
    evaluate_metrics(MockEvalArgs(features_file, basic_gt_file, basic_results_file, args.threshold))
    
    print("\n5. Evaluating metrics against multi-dimensional ground truth...")
    evaluate_metrics(MockEvalArgs(features_file, multi_gt_file, multi_results_file, args.threshold))
    
    # Generate summary report
    summary_file = os.path.join(args.output_dir, f"analysis_summary_{timestamp}.md")
    generate_summary_report(basic_results_file, multi_results_file, summary_file)
    
    print(f"\nFull analysis complete! Results saved in {args.output_dir}")
    print(f"Summary report: {summary_file}")


def generate_summary_report(basic_results_file, multi_results_file, output_file):
    """Generate a markdown summary report comparing results."""
    
    # Load results
    with open(basic_results_file, 'r') as f:
        basic_results = json.load(f)
    
    with open(multi_results_file, 'r') as f:
        multi_results = json.load(f)
    
    # Generate report
    with open(output_file, 'w') as f:
        f.write("# CARLA Scenario Similarity Analysis - Summary Report\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## Overview\n\n")
        f.write("This report compares the performance of different similarity metrics ")
        f.write("against both basic filename-based and multi-dimensional ground truth.\n\n")
        
        # Basic GT results
        f.write("## Results vs Basic Ground Truth\n\n")
        if 'summary' in basic_results and 'best_by_category' in basic_results['summary']:
            for category, best in basic_results['summary']['best_by_category'].items():
                f.write(f"- **{category.replace('_', ' ').title()}**: {best['metric']} ")
                f.write(f"(F1={best['f1_score']:.3f})\n")
        
        f.write(f"\n**Overall Best**: {basic_results['summary']['best_metric']} ")
        f.write(f"(F1={basic_results['summary']['best_f1_score']:.3f})\n\n")
        
        # Multi-dimensional GT results  
        f.write("## Results vs Multi-Dimensional Ground Truth\n\n")
        if 'summary' in multi_results and 'best_by_category' in multi_results['summary']:
            for category, best in multi_results['summary']['best_by_category'].items():
                f.write(f"- **{category.replace('_', ' ').title()}**: {best['metric']} ")
                f.write(f"(F1={best['f1_score']:.3f})\n")
        
        f.write(f"\n**Overall Best**: {multi_results['summary']['best_metric']} ")
        f.write(f"(F1={multi_results['summary']['best_f1_score']:.3f})\n\n")
        
        f.write("## Conclusion\n\n")
        f.write("The multi-dimensional ground truth provides a more sophisticated ")
        f.write("evaluation framework that considers behavioral, spatial, traffic, and ")
        f.write("contextual similarities beyond simple filename patterns.\n")
    
    print(f"Summary report generated: {output_file}")


if __name__ == "__main__":
    main()