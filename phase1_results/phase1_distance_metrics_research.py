#!/usr/bin/env python

"""
CARLA Scenario Similarity Metrics Research Framework
===================================================

Phase 1: Distance-Based Similarity Metrics Implementation
- Euclidean Distance
- Manhattan Distance  
- Cosine Similarity
- Minkowski Distance

Following IEEE Standards for experimental validation and ACM guidelines.
"""

import os
import sys
import carla
import math
import json
import glob
import time
import numpy as np
import pandas as pd
from collections import defaultdict, Counter
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.spatial.distance import euclidean, cityblock, cosine, minkowski
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix
from sklearn.model_selection import cross_val_score, KFold
import warnings
warnings.filterwarnings('ignore')

# Add the scenario runner to the path
sys.path.append('/home/ads/ads_testing/scenario_runner/scenario_runner-0.9.15')

from srunner.metrics.tools.metrics_log import MetricsLog

class SimilarityMetricsResearchFramework:
    """
    Comprehensive research framework for evaluating similarity metrics
    in CARLA scenario redundancy detection
    """
    
    def __init__(self, log_files_dir):
        self.log_files_dir = log_files_dir
        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(10.0)
        
        # Research data storage
        self.scenarios_features = {}
        self.similarity_matrices = {}
        self.evaluation_results = {}
        self.ground_truth = {}
        
        # Experimental parameters
        self.distance_metrics = {
            'euclidean': self._euclidean_similarity,
            'manhattan': self._manhattan_similarity,
            'cosine': self._cosine_similarity,
            'minkowski_p3': lambda x, y: self._minkowski_similarity(x, y, p=3),
            'minkowski_p0.5': lambda x, y: self._minkowski_similarity(x, y, p=0.5)
        }
        
        # Feature normalization settings
        self.normalization_methods = ['none', 'min_max', 'z_score', 'robust']
        
    def extract_comprehensive_features(self, log_file):
        """Extract comprehensive feature vectors for similarity analysis"""
        try:
            recorder_file = f"{os.getenv('SCENARIO_RUNNER_ROOT', './')}/{log_file}"
            recorder_str = self.client.show_recorder_file_info(recorder_file, True)
            log = MetricsLog(recorder_str)
            
            ego_id = log.get_ego_vehicle_id()
            start_frame, end_frame = log.get_actor_alive_frames(ego_id)
            
            # Initialize comprehensive feature vector
            features = {
                'temporal_features': [],
                'behavioral_features': [],
                'spatial_features': [],
                'speed_features': [],
                'traffic_features': [],
                'combined_vector': []
            }
            
            # Sample frames for analysis
            sample_frames = range(start_frame, end_frame, max(1, (end_frame - start_frame) // 100))
            
            speeds = []
            positions = []
            accelerations = []
            steering_angles = []
            behaviors = []
            
            prev_speed = 0
            prev_pos = None
            
            for frame in sample_frames:
                try:
                    # Velocity and speed
                    velocity = log.get_actor_velocity(ego_id, frame)
                    speed = math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
                    speeds.append(speed)
                    
                    # Position and spatial info
                    transform = log.get_actor_transform(ego_id, frame)
                    pos = (transform.location.x, transform.location.y)
                    positions.append(pos)
                    
                    # Acceleration calculation
                    if prev_speed is not None:
                        acceleration = speed - prev_speed
                        accelerations.append(acceleration)
                    
                    # Control inputs
                    try:
                        control = log.get_vehicle_control(ego_id, frame)
                        steer = control.steer if control else 0
                        steering_angles.append(steer)
                    except:
                        steering_angles.append(0)
                    
                    # Behavioral classification
                    behavior_code = self._encode_behavior(speed, prev_speed, 
                                                        steering_angles[-1] if steering_angles else 0)
                    behaviors.append(behavior_code)
                    
                    prev_speed = speed
                    prev_pos = pos
                    
                except Exception:
                    continue
            
            # Extract feature vectors
            features['temporal_features'] = self._extract_temporal_features(
                speeds, accelerations, len(sample_frames)
            )
            
            features['behavioral_features'] = self._extract_behavioral_features(
                behaviors, steering_angles
            )
            
            features['spatial_features'] = self._extract_spatial_features(positions)
            
            features['speed_features'] = self._extract_speed_features(speeds, accelerations)
            
            # Traffic context
            vehicle_ids = log.get_actor_ids_with_type_id("vehicle.*")
            traffic_count = len([vid for vid in vehicle_ids if vid != ego_id])
            features['traffic_features'] = [traffic_count, 
                                          min(traffic_count, 10),  # Capped complexity
                                          1 if traffic_count > 0 else 0]  # Binary indicator
            
            # Create combined normalized feature vector
            features['combined_vector'] = (
                features['temporal_features'] + 
                features['behavioral_features'] + 
                features['spatial_features'] + 
                features['speed_features'] + 
                features['traffic_features']
            )
            
            return features
            
        except Exception as e:
            print(f"Error extracting features from {log_file}: {e}")
            return None
    
    def _encode_behavior(self, speed, prev_speed, steer):
        """Encode behavior into numerical representation"""
        if speed < 0.5:
            return 0  # Stop
        elif speed > prev_speed + 1.0:
            return 1  # Accelerate
        elif speed < prev_speed - 1.0:
            return 2  # Decelerate
        elif abs(steer) > 0.1:
            return 3 if steer > 0 else 4  # Left/Right turn
        elif speed > 10.0:
            return 5  # Fast cruise
        elif speed > 2.0:
            return 6  # Normal cruise
        else:
            return 7  # Idle
    
    def _extract_temporal_features(self, speeds, accelerations, frame_count):
        """Extract temporal characteristics"""
        if not speeds:
            return [0] * 6
        
        duration = frame_count * 0.05  # Assuming 20 FPS
        speed_changes = sum(1 for i in range(1, len(speeds)) 
                           if abs(speeds[i] - speeds[i-1]) > 1.0)
        
        return [
            duration,
            len(speeds),
            speed_changes,
            speed_changes / len(speeds) if speeds else 0,
            np.std(speeds) if len(speeds) > 1 else 0,
            len([a for a in accelerations if abs(a) > 2.0])  # Significant accelerations
        ]
    
    def _extract_behavioral_features(self, behaviors, steering_angles):
        """Extract behavioral pattern features"""
        if not behaviors:
            return [0] * 10
        
        behavior_counts = Counter(behaviors)
        behavior_transitions = len(set(zip(behaviors[:-1], behaviors[1:]))) if len(behaviors) > 1 else 0
        
        avg_steer = np.mean([abs(s) for s in steering_angles]) if steering_angles else 0
        max_steer = max([abs(s) for s in steering_angles]) if steering_angles else 0
        
        return [
            behavior_counts.get(0, 0),  # Stops
            behavior_counts.get(1, 0),  # Accelerations
            behavior_counts.get(2, 0),  # Decelerations
            behavior_counts.get(3, 0) + behavior_counts.get(4, 0),  # Turns
            behavior_counts.get(5, 0) + behavior_counts.get(6, 0),  # Cruise
            behavior_transitions,
            len(set(behaviors)),  # Unique behaviors
            avg_steer,
            max_steer,
            len(behaviors)  # Total behavior count
        ]
    
    def _extract_spatial_features(self, positions):
        """Extract spatial movement characteristics"""
        if len(positions) < 2:
            return [0] * 8
        
        # Path length
        path_length = sum(math.sqrt((positions[i][0] - positions[i-1][0])**2 + 
                                   (positions[i][1] - positions[i-1][1])**2)
                         for i in range(1, len(positions)))
        
        # Bounding box
        xs = [p[0] for p in positions]
        ys = [p[1] for p in positions]
        bbox_width = max(xs) - min(xs)
        bbox_height = max(ys) - min(ys)
        
        # Displacement
        total_displacement = math.sqrt((positions[-1][0] - positions[0][0])**2 + 
                                     (positions[-1][1] - positions[0][1])**2)
        
        # Curvature estimation
        direction_changes = 0
        for i in range(2, len(positions)):
            v1 = (positions[i-1][0] - positions[i-2][0], positions[i-1][1] - positions[i-2][1])
            v2 = (positions[i][0] - positions[i-1][0], positions[i][1] - positions[i-1][1])
            cross_product = abs(v1[0] * v2[1] - v1[1] * v2[0])
            if cross_product > 50:  # Threshold for significant direction change
                direction_changes += 1
        
        return [
            path_length,
            total_displacement,
            path_length / total_displacement if total_displacement > 0 else 1,  # Tortuosity
            bbox_width,
            bbox_height,
            bbox_width * bbox_height,  # Area
            direction_changes,
            direction_changes / len(positions) if positions else 0  # Curvature density
        ]
    
    def _extract_speed_features(self, speeds, accelerations):
        """Extract speed profile characteristics"""
        if not speeds:
            return [0] * 10
        
        speeds_array = np.array(speeds)
        acc_array = np.array(accelerations) if accelerations else np.array([0])
        
        return [
            np.mean(speeds_array),
            np.std(speeds_array),
            np.min(speeds_array),
            np.max(speeds_array),
            np.median(speeds_array),
            np.percentile(speeds_array, 75) - np.percentile(speeds_array, 25),  # IQR
            np.mean(acc_array),
            np.std(acc_array),
            len([s for s in speeds if s > 10]),  # High speed count
            len([a for a in accelerations if abs(a) > 3])  # Hard accel/decel count
        ]
    
    def normalize_features(self, features_dict, method='z_score'):
        """Normalize feature vectors using different methods"""
        if method == 'none':
            return features_dict
        
        # Collect all feature vectors
        all_vectors = []
        scenario_names = []
        
        for name, features in features_dict.items():
            if features and 'combined_vector' in features:
                all_vectors.append(features['combined_vector'])
                scenario_names.append(name)
        
        if not all_vectors:
            return features_dict
        
        feature_matrix = np.array(all_vectors)
        
        if method == 'min_max':
            # Min-max normalization [0, 1]
            min_vals = np.min(feature_matrix, axis=0)
            max_vals = np.max(feature_matrix, axis=0)
            normalized_matrix = (feature_matrix - min_vals) / (max_vals - min_vals + 1e-8)
            
        elif method == 'z_score':
            # Z-score normalization (standardization)
            mean_vals = np.mean(feature_matrix, axis=0)
            std_vals = np.std(feature_matrix, axis=0)
            normalized_matrix = (feature_matrix - mean_vals) / (std_vals + 1e-8)
            
        elif method == 'robust':
            # Robust normalization using median and IQR
            median_vals = np.median(feature_matrix, axis=0)
            q75 = np.percentile(feature_matrix, 75, axis=0)
            q25 = np.percentile(feature_matrix, 25, axis=0)
            iqr = q75 - q25
            normalized_matrix = (feature_matrix - median_vals) / (iqr + 1e-8)
        
        # Update the features dictionary
        normalized_features = {}
        for i, name in enumerate(scenario_names):
            normalized_features[name] = features_dict[name].copy()
            normalized_features[name]['combined_vector'] = normalized_matrix[i].tolist()
        
        # Add scenarios that couldn't be normalized
        for name, features in features_dict.items():
            if name not in normalized_features:
                normalized_features[name] = features
        
        return normalized_features
    
    # Distance-based similarity methods
    def _euclidean_similarity(self, vec1, vec2):
        """Euclidean distance converted to similarity [0,1]"""
        distance = euclidean(vec1, vec2)
        # Convert to similarity using exponential decay
        return math.exp(-distance / (len(vec1) + 1e-8))
    
    def _manhattan_similarity(self, vec1, vec2):
        """Manhattan distance (cityblock) converted to similarity [0,1]"""
        distance = cityblock(vec1, vec2)
        return math.exp(-distance / (2 * len(vec1) + 1e-8))
    
    def _cosine_similarity(self, vec1, vec2):
        """Cosine similarity [0,1]"""
        # Handle zero vectors
        if np.linalg.norm(vec1) == 0 or np.linalg.norm(vec2) == 0:
            return 0.0
        return (1 + (1 - cosine(vec1, vec2))) / 2  # Convert [-1,1] to [0,1]
    
    def _minkowski_similarity(self, vec1, vec2, p=3):
        """Minkowski distance converted to similarity [0,1]"""
        distance = minkowski(vec1, vec2, p=p)
        return math.exp(-distance / (len(vec1)**(1/p) + 1e-8))
    
    def compute_similarity_matrices(self, features_dict, normalization='z_score'):
        """Compute similarity matrices for all distance metrics"""
        print(f"\n🔍 Computing similarity matrices with {normalization} normalization...")
        
        # Normalize features
        normalized_features = self.normalize_features(features_dict, normalization)
        
        # Get scenarios with valid features
        valid_scenarios = [(name, features['combined_vector']) 
                          for name, features in normalized_features.items() 
                          if features and 'combined_vector' in features]
        
        scenario_names = [name for name, _ in valid_scenarios]
        feature_vectors = [vec for _, vec in valid_scenarios]
        
        print(f"Computing similarities for {len(scenario_names)} scenarios...")
        
        similarity_results = {}
        
        for metric_name, metric_func in self.distance_metrics.items():
            print(f"  Computing {metric_name} similarities...")
            
            # Initialize similarity matrix
            n = len(scenario_names)
            similarity_matrix = np.zeros((n, n))
            
            # Compute pairwise similarities
            for i in range(n):
                for j in range(i, n):
                    if i == j:
                        similarity_matrix[i][j] = 1.0
                    else:
                        sim = metric_func(feature_vectors[i], feature_vectors[j])
                        similarity_matrix[i][j] = sim
                        similarity_matrix[j][i] = sim
            
            similarity_results[metric_name] = {
                'matrix': similarity_matrix,
                'scenario_names': scenario_names,
                'normalization': normalization
            }
        
        return similarity_results
    
    def create_ground_truth_labels(self, scenario_names):
        """Create ground truth similarity labels based on scenario naming patterns"""
        print("\n📋 Creating ground truth labels based on scenario patterns...")
        
        n = len(scenario_names)
        ground_truth_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i, n):
                if i == j:
                    ground_truth_matrix[i][j] = 1.0
                else:
                    # Extract similarity patterns from filenames
                    name1 = scenario_names[i]
                    name2 = scenario_names[j]
                    
                    similarity = 0.0
                    
                    # Same scenario family (e.g., ZAM_Tjunction)
                    if name1.split('-')[0] == name2.split('-')[0]:
                        similarity += 0.4
                    
                    # Same country/region
                    if name1.split('_')[0] == name2.split('_')[0]:
                        similarity += 0.2
                    
                    # Same location
                    if '_'.join(name1.split('_')[:2]) == '_'.join(name2.split('_')[:2]):
                        similarity += 0.3
                    
                    # Same scenario number pattern
                    if name1.split('_')[-1] == name2.split('_')[-1]:
                        similarity += 0.1
                    
                    ground_truth_matrix[i][j] = similarity
                    ground_truth_matrix[j][i] = similarity
        
        return ground_truth_matrix
    
    def evaluate_similarity_metric(self, similarity_matrix, ground_truth_matrix, 
                                  threshold=0.7, metric_name="Unknown"):
        """Evaluate similarity metric performance"""
        
        # Convert to binary predictions
        predictions = (similarity_matrix >= threshold).astype(int)
        ground_truth_binary = (ground_truth_matrix >= 0.6).astype(int)
        
        # Flatten matrices (excluding diagonal)
        n = len(predictions)
        pred_flat = []
        gt_flat = []
        
        for i in range(n):
            for j in range(i+1, n):
                pred_flat.append(predictions[i][j])
                gt_flat.append(ground_truth_binary[i][j])
        
        # Calculate metrics
        accuracy = accuracy_score(gt_flat, pred_flat)
        precision, recall, f1, _ = precision_recall_fscore_support(
            gt_flat, pred_flat, average='binary', zero_division=0
        )
        
        # Calculate correlation with continuous ground truth
        sim_flat = [similarity_matrix[i][j] for i in range(n) for j in range(i+1, n)]
        gt_cont_flat = [ground_truth_matrix[i][j] for i in range(n) for j in range(i+1, n)]
        
        correlation = stats.pearsonr(sim_flat, gt_cont_flat)[0]
        spearman_corr = stats.spearmanr(sim_flat, gt_cont_flat)[0]
        
        return {
            'metric_name': metric_name,
            'threshold': threshold,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'pearson_correlation': correlation,
            'spearman_correlation': spearman_corr,
            'num_pairs': len(pred_flat),
            'num_similar_pred': sum(pred_flat),
            'num_similar_gt': sum(gt_flat)
        }
    
    def run_phase1_experiments(self, max_scenarios=None):
        """Run Phase 1: Distance-based similarity metrics experiments"""
        
        print("🚀 STARTING PHASE 1: Distance-Based Similarity Metrics")
        print("="*80)
        
        # Get log files
        log_files = glob.glob(os.path.join(self.log_files_dir, "*.log"))
        if max_scenarios:
            log_files = log_files[:max_scenarios]
        
        print(f"📁 Processing {len(log_files)} scenario files...")
        
        # Extract features
        print("\n📊 Extracting comprehensive features...")
        start_time = time.time()
        
        features_dict = {}
        successful = 0
        
        for i, log_file in enumerate(log_files, 1):
            filename = os.path.basename(log_file)
            print(f"  [{i:3d}/{len(log_files)}] Processing {filename}")
            
            features = self.extract_comprehensive_features(f"log_files/{filename}")
            if features:
                features_dict[filename] = features
                successful += 1
        
        feature_time = time.time() - start_time
        print(f"\n✅ Feature extraction complete: {successful}/{len(log_files)} successful")
        print(f"⏱️  Time taken: {feature_time:.1f} seconds")
        
        if not features_dict:
            print("❌ No features extracted. Exiting.")
            return
        
        # Create ground truth
        scenario_names = list(features_dict.keys())
        ground_truth_matrix = self.create_ground_truth_labels(scenario_names)
        
        # Experiment with different normalizations
        results_summary = []
        
        for normalization in self.normalization_methods:
            print(f"\n🔬 EXPERIMENT: {normalization.upper()} NORMALIZATION")
            print("-" * 60)
            
            # Compute similarity matrices
            similarity_results = self.compute_similarity_matrices(features_dict, normalization)
            
            # Evaluate each metric
            for metric_name, result_data in similarity_results.items():
                similarity_matrix = result_data['matrix']
                
                # Test multiple thresholds
                thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
                best_f1 = 0
                best_threshold = 0.7
                best_eval = None
                
                for threshold in thresholds:
                    evaluation = self.evaluate_similarity_metric(
                        similarity_matrix, ground_truth_matrix, threshold, 
                        f"{normalization}_{metric_name}"
                    )
                    
                    if evaluation['f1_score'] > best_f1:
                        best_f1 = evaluation['f1_score']
                        best_threshold = threshold
                        best_eval = evaluation
                
                # Store best result
                if best_eval:
                    best_eval['normalization'] = normalization
                    results_summary.append(best_eval)
                    
                    print(f"  {metric_name:15s} | F1: {best_f1:.3f} | "
                          f"Acc: {best_eval['accuracy']:.3f} | "
                          f"Corr: {best_eval['pearson_correlation']:.3f} | "
                          f"Thresh: {best_threshold}")
        
        # Save results
        self._save_phase1_results(results_summary, features_dict, ground_truth_matrix)
        
        # Generate summary report
        self._generate_phase1_report(results_summary)
        
        print(f"\n🎯 PHASE 1 COMPLETE!")
        print(f"Results saved to phase1_distance_metrics_results.json")
        
        return results_summary
    
    def _save_phase1_results(self, results_summary, features_dict, ground_truth_matrix):
        """Save Phase 1 experimental results"""
        
        results_data = {
            'experiment_info': {
                'phase': 'Phase 1: Distance-Based Metrics',
                'timestamp': datetime.now().isoformat(),
                'num_scenarios': len(features_dict),
                'metrics_tested': list(self.distance_metrics.keys()),
                'normalization_methods': self.normalization_methods
            },
            'evaluation_results': results_summary,
            'feature_statistics': self._compute_feature_statistics(features_dict),
            'ground_truth_info': {
                'matrix_shape': ground_truth_matrix.shape,
                'similarity_distribution': {
                    'mean': float(np.mean(ground_truth_matrix)),
                    'std': float(np.std(ground_truth_matrix)),
                    'min': float(np.min(ground_truth_matrix)),
                    'max': float(np.max(ground_truth_matrix))
                }
            }
        }
        
        output_file = '/home/ads/ads_testing/phase1_distance_metrics_results.json'
        with open(output_file, 'w') as f:
            json.dump(results_data, f, indent=2, default=str)
    
    def _compute_feature_statistics(self, features_dict):
        """Compute statistics about extracted features"""
        if not features_dict:
            return {}
        
        # Collect all feature vectors
        all_vectors = [f['combined_vector'] for f in features_dict.values() 
                      if f and 'combined_vector' in f]
        
        if not all_vectors:
            return {}
        
        feature_matrix = np.array(all_vectors)
        
        return {
            'num_features': feature_matrix.shape[1],
            'feature_means': feature_matrix.mean(axis=0).tolist(),
            'feature_stds': feature_matrix.std(axis=0).tolist(),
            'feature_ranges': {
                'min': feature_matrix.min(axis=0).tolist(),
                'max': feature_matrix.max(axis=0).tolist()
            }
        }
    
    def _generate_phase1_report(self, results_summary):
        """Generate comprehensive Phase 1 report"""
        
        # Sort results by F1 score
        sorted_results = sorted(results_summary, key=lambda x: x['f1_score'], reverse=True)
        
        print(f"\n📈 PHASE 1 RESULTS SUMMARY")
        print("="*80)
        print(f"{'Rank':<4} {'Method':<25} {'F1':<6} {'Acc':<6} {'Prec':<6} {'Rec':<6} {'Corr':<6}")
        print("-"*80)
        
        for i, result in enumerate(sorted_results[:10], 1):
            method_name = f"{result['normalization']}_{result['metric_name'].split('_', 1)[-1]}"
            print(f"{i:<4} {method_name:<25} {result['f1_score']:<6.3f} "
                  f"{result['accuracy']:<6.3f} {result['precision']:<6.3f} "
                  f"{result['recall']:<6.3f} {result['pearson_correlation']:<6.3f}")
        
        # Best performing metrics
        best_overall = sorted_results[0]
        best_correlation = max(sorted_results, key=lambda x: x['pearson_correlation'])
        best_precision = max(sorted_results, key=lambda x: x['precision'])
        
        print(f"\n🏆 BEST PERFORMING METRICS:")
        print(f"Best Overall (F1):     {best_overall['normalization']}_{best_overall['metric_name'].split('_', 1)[-1]} (F1: {best_overall['f1_score']:.3f})")
        print(f"Best Correlation:      {best_correlation['normalization']}_{best_correlation['metric_name'].split('_', 1)[-1]} (Corr: {best_correlation['pearson_correlation']:.3f})")
        print(f"Best Precision:        {best_precision['normalization']}_{best_precision['metric_name'].split('_', 1)[-1]} (Prec: {best_precision['precision']:.3f})")

def main():
    """Main function to run Phase 1 experiments"""
    
    # Set environment variable
    os.environ['SCENARIO_RUNNER_ROOT'] = '/home/ads/ads_testing'
    
    print("🔬 CARLA Scenario Similarity Metrics Research - Phase 1")
    print("Distance-Based Metrics Evaluation")
    print("="*80)
    
    # Initialize research framework
    framework = SimilarityMetricsResearchFramework('/home/ads/ads_testing/log_files')
    
    # Ask for number of scenarios to test
    print("Select experiment size:")
    print("1. Small test (20 scenarios)")
    print("2. Medium test (50 scenarios)")  
    print("3. Large test (100 scenarios)")
    print("4. Full dataset (174 scenarios)")
    
    choice = input("Enter choice (1-4) or press Enter for small test: ").strip()
    
    max_scenarios = {
        '1': 20,
        '2': 50, 
        '3': 100,
        '4': None,
        '': 20
    }.get(choice, 20)
    
    try:
        # Run Phase 1 experiments
        results = framework.run_phase1_experiments(max_scenarios)
        
        print(f"\n✅ Phase 1 experiments completed successfully!")
        print(f"📊 Evaluated {len(framework.distance_metrics)} distance metrics")
        print(f"🔧 Tested {len(framework.normalization_methods)} normalization methods")
        print(f"📈 Generated {len(results)} evaluation results")
        
    except KeyboardInterrupt:
        print("\n⏹️  Experiments interrupted by user")
    except Exception as e:
        print(f"\n❌ Experiments failed: {e}")
        raise

if __name__ == "__main__":
    main()
