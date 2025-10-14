#!/usr/bin/env python3
"""
CARLA Scenario Similarity Analysis Framework - Main CLI Interface

Command-line interface for the consolidated CARLA similarity analysis framework.
Provides easy access to feature extraction, ground truth generation, and evaluation.
"""

import argparse
import os
import sys
import json
from typing import Dict, Any

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from feature_extraction import FeatureExtractor
    from ground_truth import BasicGroundTruth, MultiDimensionalGroundTruth
    from evaluation import SimilarityEvaluator
    from similarity_metrics import DistanceBasedMetrics, SequenceBasedMetrics, SetBasedMetrics
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure CARLA is installed: pip install carla==0.9.15")
    sys.exit(1)


def extract_features_command(args):
    """Extract 37-dimensional features from log files."""
    print("🔍 Extracting 37-dimensional features from CARLA log files...")
    
    extractor = FeatureExtractor()
    features = extractor.extract_features_from_logs(args.log_directory)
    
    # Save features to file
    output_file = os.path.join(args.log_directory, "extracted_features.json")
    with open(output_file, 'w') as f:
        json.dump(features, f, indent=2, default=str)
    
    print(f"✅ Successfully extracted features from {len(features)} scenarios")
    print(f"📁 Features saved to: {output_file}")
    print(f"📊 Feature dimensions: {len(next(iter(features.values()))['features'])} per scenario")


def generate_basic_gt_command(args):
    """Generate basic ground truth (filename-based)."""
    print("🏷️ Generating basic ground truth (filename-based)...")
    
    # Get log files
    log_files = [f for f in os.listdir(args.log_directory) if f.endswith('.log')]
    
    basic_gt = BasicGroundTruth()
    pairs = basic_gt.generate_ground_truth(log_files)
    
    # Save ground truth
    output_file = os.path.join(args.log_directory, "basic_ground_truth.json")
    with open(output_file, 'w') as f:
        json.dump([(f1, f2, sim) for f1, f2, sim in pairs], f, indent=2)
    
    similar_count = sum(1 for _, _, sim in pairs if sim)
    print(f"✅ Generated {len(pairs)} ground truth pairs")
    print(f"📁 Ground truth saved to: {output_file}")
    print(f"📊 Similar pairs: {similar_count}, Different pairs: {len(pairs) - similar_count}")


def generate_multi_gt_command(args):
    """Generate multi-dimensional ground truth (behavioral)."""
    print("🧠 Generating multi-dimensional ground truth (behavioral)...")
    
    # Load features if they exist, otherwise extract them
    features_file = os.path.join(args.log_directory, "extracted_features.json")
    if os.path.exists(features_file):
        print("📂 Loading existing features...")
        with open(features_file, 'r') as f:
            features = json.load(f)
    else:
        print("🔍 Features not found, extracting...")
        extractor = FeatureExtractor()
        features = extractor.extract_features_from_logs(args.log_directory)
    
    multi_gt = MultiDimensionalGroundTruth()
    pairs = multi_gt.generate_ground_truth(features)
    
    # Save ground truth
    output_file = os.path.join(args.log_directory, "multi_dimensional_ground_truth.json")
    with open(output_file, 'w') as f:
        json.dump([(f1, f2, sim) for f1, f2, sim in pairs], f, indent=2)
    
    similar_count = sum(1 for _, _, sim in pairs if sim)
    print(f"✅ Generated {len(pairs)} ground truth pairs (multi-dimensional)")
    print(f"📁 Ground truth saved to: {output_file}")
    print(f"📊 Similar pairs: {similar_count}, Different pairs: {len(pairs) - similar_count}")


def evaluate_metrics_command(args):
    """Evaluate all similarity metrics."""
    print("🏆 Evaluating all similarity metrics...")
    
    # Load features and ground truth
    features_file = os.path.join(args.log_directory, "extracted_features.json")
    gt_file = os.path.join(args.log_directory, "multi_dimensional_ground_truth.json")
    
    if not os.path.exists(features_file):
        print("❌ Features file not found. Run extract-features first.")
        return
    
    if not os.path.exists(gt_file):
        print("❌ Ground truth file not found. Run multi-gt first.")
        return
    
    # Load data
    with open(features_file, 'r') as f:
        features = json.load(f)
    
    with open(gt_file, 'r') as f:
        pairs_data = json.load(f)
        pairs = [(f1, f2, sim) for f1, f2, sim in pairs_data]
    
    # Run evaluation
    evaluator = SimilarityEvaluator()
    results = evaluator.evaluate_all_metrics(features, pairs)
    
    # Save results
    output_file = os.path.join(args.log_directory, "similarity_evaluation_results.json")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Display top results
    sorted_results = sorted(results.items(), key=lambda x: x[1]['f1_score'], reverse=True)
    
    print(f"✅ Evaluation complete! Results for {len(results)} metrics:")
    print(f"📁 Detailed results saved to: {output_file}")
    print("\n🥇 Top 5 performing metrics:")
    
    for i, (metric_name, metrics) in enumerate(sorted_results[:5]):
        print(f"   {i+1}. {metric_name}:")
        print(f"      F1-Score: {metrics['f1_score']:.3f}")
        print(f"      Accuracy: {metrics['accuracy']:.3f}")
        print(f"      Precision: {metrics['precision']:.3f}")
        print(f"      Recall: {metrics['recall']:.3f}")


def full_analysis_command(args):
    """Run complete analysis pipeline."""
    print("🚀 Running complete CARLA similarity analysis pipeline...")
    print("=" * 70)
    
    # Step 1: Extract features
    print("\n📊 Step 1: Feature extraction")
    extract_features_command(args)
    
    # Step 2: Generate basic ground truth
    print("\n🏷️ Step 2: Basic ground truth generation")
    generate_basic_gt_command(args)
    
    # Step 3: Generate multi-dimensional ground truth
    print("\n🧠 Step 3: Multi-dimensional ground truth generation")
    generate_multi_gt_command(args)
    
    # Step 4: Evaluate metrics
    print("\n🏆 Step 4: Similarity metrics evaluation")
    evaluate_metrics_command(args)
    
    print("\n✨ Complete analysis finished!")
    print("📁 Check the log directory for all generated files:")
    print("   - extracted_features.json")
    print("   - basic_ground_truth.json") 
    print("   - multi_dimensional_ground_truth.json")
    print("   - similarity_evaluation_results.json")


def main():
    """Main CLI interface."""
    parser = argparse.ArgumentParser(
        description="CARLA Scenario Similarity Analysis Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py extract-features /path/to/logs/
  python main.py basic-gt /path/to/logs/
  python main.py multi-gt /path/to/logs/
  python main.py evaluate /path/to/logs/
  python main.py full-analysis /path/to/logs/
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Extract features command
    extract_parser = subparsers.add_parser('extract-features', 
                                         help='Extract 37-dimensional features from log files')
    extract_parser.add_argument('log_directory', help='Directory containing CARLA log files')
    
    # Basic ground truth command
    basic_gt_parser = subparsers.add_parser('basic-gt',
                                          help='Generate basic ground truth (filename-based)')
    basic_gt_parser.add_argument('log_directory', help='Directory containing CARLA log files')
    
    # Multi-dimensional ground truth command  
    multi_gt_parser = subparsers.add_parser('multi-gt',
                                          help='Generate multi-dimensional ground truth (behavioral)')
    multi_gt_parser.add_argument('log_directory', help='Directory containing CARLA log files')
    
    # Evaluate metrics command
    evaluate_parser = subparsers.add_parser('evaluate',
                                          help='Evaluate all similarity metrics')
    evaluate_parser.add_argument('log_directory', help='Directory containing CARLA log files')
    
    # Full analysis command
    full_parser = subparsers.add_parser('full-analysis',
                                      help='Run complete analysis pipeline')
    full_parser.add_argument('log_directory', help='Directory containing CARLA log files')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Validate log directory
    if not os.path.exists(args.log_directory):
        print(f"❌ Error: Log directory '{args.log_directory}' does not exist")
        return
    
    # Execute command
    commands = {
        'extract-features': extract_features_command,
        'basic-gt': generate_basic_gt_command,
        'multi-gt': generate_multi_gt_command,
        'evaluate': evaluate_metrics_command,
        'full-analysis': full_analysis_command
    }
    
    try:
        commands[args.command](args)
    except Exception as e:
        print(f"❌ Error executing {args.command}: {e}")
        print("Make sure CARLA is running and accessible.")


if __name__ == "__main__":
    main()
