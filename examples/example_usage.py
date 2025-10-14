#!/usr/bin/env python3
"""
CARLA Scenario Similarity Analysis Framework - Example Usage

This script demonstrates how to use the consolidated CARLA similarity analysis framework
for extracting features, generating ground truth, and evaluating similarity metrics.
"""

import os
import sys

# Add the parent directory to the path so we can import carla_similarity
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from carla_similarity import (
        FeatureExtractor, 
        SimilarityEvaluator,
        BasicGroundTruth, 
        MultiDimensionalGroundTruth,
        DistanceBasedMetrics,
        SequenceBasedMetrics,
        SetBasedMetrics
    )
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure CARLA is installed: pip install carla==0.9.15")
    sys.exit(1)


def main():
    """
    Example workflow showing how to use the CARLA similarity analysis framework.
    """
    # Path to log files - update this to your actual log directory
    log_directory = "../log_files"  # Adjust path as needed
    
    if not os.path.exists(log_directory):
        print(f"Log directory not found: {log_directory}")
        print("Please update the log_directory path in this script.")
        return
    
    print("🚀 CARLA Scenario Similarity Analysis Framework - Example Usage")
    print("=" * 70)
    
    # Step 1: Extract 37-dimensional features
    print("\n📊 Step 1: Extracting 37-dimensional features from log files...")
    try:
        extractor = FeatureExtractor()
        features = extractor.extract_features_from_logs(log_directory)
        print(f"✅ Successfully extracted features from {len(features)} scenarios")
        print(f"   Feature dimensions: {len(next(iter(features.values()))['features'])} per scenario")
    except Exception as e:
        print(f"❌ Feature extraction failed: {e}")
        print("Make sure CARLA server is running and log files are accessible.")
        return
    
    # Step 2: Generate basic ground truth (filename-based)
    print("\n🏷️ Step 2: Generating basic ground truth (filename-based)...")
    try:
        basic_gt = BasicGroundTruth()
        log_files = list(features.keys())
        basic_pairs = basic_gt.generate_ground_truth(log_files)
        print(f"✅ Generated {len(basic_pairs)} ground truth pairs (basic)")
        similar_count = sum(1 for _, _, sim in basic_pairs if sim)
        print(f"   Similar pairs: {similar_count}, Different pairs: {len(basic_pairs) - similar_count}")
    except Exception as e:
        print(f"❌ Basic ground truth generation failed: {e}")
        return
    
    # Step 3: Generate multi-dimensional ground truth (behavioral)
    print("\n🧠 Step 3: Generating multi-dimensional ground truth (behavioral)...")
    try:
        multi_gt = MultiDimensionalGroundTruth()
        multi_pairs = multi_gt.generate_ground_truth(features)
        print(f"✅ Generated {len(multi_pairs)} ground truth pairs (multi-dimensional)")
        similar_count = sum(1 for _, _, sim in multi_pairs if sim)
        print(f"   Similar pairs: {similar_count}, Different pairs: {len(multi_pairs) - similar_count}")
    except Exception as e:
        print(f"❌ Multi-dimensional ground truth generation failed: {e}")
        return
    
    # Step 4: Demonstrate individual similarity metrics
    print("\n📏 Step 4: Testing individual similarity metric categories...")
    
    # Distance-based metrics
    try:
        distance_metrics = DistanceBasedMetrics()
        scenario_names = list(features.keys())
        if len(scenario_names) >= 2:
            feat1 = features[scenario_names[0]]['features']
            feat2 = features[scenario_names[1]]['features']
            
            cosine_sim = distance_metrics.cosine_similarity(feat1, feat2)
            euclidean_sim = distance_metrics.euclidean_similarity(feat1, feat2)
            
            print(f"   Distance metrics between '{scenario_names[0]}' and '{scenario_names[1]}':")
            print(f"     Cosine similarity: {cosine_sim:.3f}")
            print(f"     Euclidean similarity: {euclidean_sim:.3f}")
    except Exception as e:
        print(f"   ⚠️ Distance metrics demo failed: {e}")
    
    # Sequence-based metrics  
    try:
        sequence_metrics = SequenceBasedMetrics()
        if len(scenario_names) >= 2:
            # Create simple test sequences for demo
            seq1 = ['STOP', 'ACCELERATE', 'CRUISE', 'TURN_RIGHT']
            seq2 = ['STOP', 'ACCELERATE', 'TURN_RIGHT', 'CRUISE']
            
            lcs_sim = sequence_metrics.lcs_similarity(seq1, seq2)
            edit_sim = sequence_metrics.edit_distance_similarity(seq1, seq2)
            
            print(f"   Sequence metrics between sample action sequences:")
            print(f"     LCS similarity: {lcs_sim:.3f}")
            print(f"     Edit distance similarity: {edit_sim:.3f}")
    except Exception as e:
        print(f"   ⚠️ Sequence metrics demo failed: {e}")
    
    # Set-based metrics
    try:
        set_metrics = SetBasedMetrics()
        if len(scenario_names) >= 2:
            # Create simple test sets for demo  
            set1 = {'CRUISE', 'TURN', 'BRAKE', 'ACCELERATE'}
            set2 = {'CRUISE', 'TURN', 'STOP', 'ACCELERATE'}
            
            jaccard_sim = set_metrics.jaccard_similarity(set1, set2)
            overlap_sim = set_metrics.overlap_coefficient(set1, set2)
            
            print(f"   Set metrics between sample behavior sets:")
            print(f"     Jaccard similarity: {jaccard_sim:.3f}")
            print(f"     Overlap coefficient: {overlap_sim:.3f}")
    except Exception as e:
        print(f"   ⚠️ Set metrics demo failed: {e}")
    
    # Step 5: Comprehensive evaluation
    print("\n🏆 Step 5: Running comprehensive similarity evaluation...")
    try:
        evaluator = SimilarityEvaluator()
        results = evaluator.evaluate_all_metrics(features, multi_pairs)
        
        print(f"✅ Evaluation complete! Results for {len(results)} metrics:")
        
        # Show top 3 performing metrics
        sorted_results = sorted(results.items(), key=lambda x: x[1]['f1_score'], reverse=True)
        print("\n   🥇 Top 3 performing metrics:")
        for i, (metric_name, metrics) in enumerate(sorted_results[:3]):
            print(f"   {i+1}. {metric_name}:")
            print(f"      F1-Score: {metrics['f1_score']:.3f}")
            print(f"      Accuracy: {metrics['accuracy']:.3f}")
            print(f"      Precision: {metrics['precision']:.3f}")
            print(f"      Recall: {metrics['recall']:.3f}")
            print()
        
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        return
    
    # Step 6: Summary
    print("\n📋 Summary:")
    print(f"   • Processed {len(features)} scenarios")
    print(f"   • Extracted 37-dimensional features per scenario")
    print(f"   • Generated ground truth with both basic and multi-dimensional methods")
    print(f"   • Evaluated {len(results)} similarity metrics")
    print(f"   • Best metric: {sorted_results[0][0]} (F1: {sorted_results[0][1]['f1_score']:.3f})")
    
    print("\n✨ Framework demonstration complete!")
    print("   For production use, consider running the full CLI:")
    print("   python carla_similarity/main.py full-analysis /path/to/log/files")


if __name__ == "__main__":
    main()
    print("\n" + "="*60)
    print("EXAMPLE: Movement Analysis")
    print("="*60)
    
    # This would typically import and run detailed_movement_analysis
    print("To run movement analysis:")
    print("python scripts/detailed_movement_analysis.py")
    print("- Outputs movement statistics")
    print("- Creates visualization plots")
    print("- Analyzes speed, acceleration, steering patterns")

def example_scenario_reduction():
    """Example: Scenario reduction analysis"""
    print("\n" + "="*60)
    print("EXAMPLE: Scenario Reduction")
    print("="*60)
    
    print("To run scenario reduction analysis:")
    print("python scripts/quick_scenario_analysis.py")
    print("- Analyzes multiple log files")


if __name__ == "__main__":
    main()
