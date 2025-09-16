#!/usr/bin/env python

"""
Phase 2: Set-Based Similarity Metrics Research
==============================================

Implementing Jaccard similarity coefficient for CARLA scenario analysis.
This phase extracts meaningful feature sets from the same 37-dimensional feature vectors
used in Phase 1, then applies set-based similarity analysis.
"""

import os
import sys
import carla
import math
import json
import glob
import numpy as np
import pandas as pd
from collections import defaultdict, Counter
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Add scenario runner to path
sys.path.append('/home/ads/ads_testing/scenario_runner/scenario_runner-0.9.15')

from srunner.metrics.tools.metrics_log import MetricsLog

class Phase2SetBasedSimilarityAnalysis:
    """
    Phase 2: Set-based similarity analysis using Jaccard coefficient
    Converts 37-dimensional feature vectors into meaningful feature sets
    """
    
    def __init__(self, log_files_dir):
        self.log_files_dir = log_files_dir
        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(10.0)
        
        # Feature thresholds for set membership (adjusted to be more reasonable)
        self.feature_thresholds = {
            'high_speed': 8.0,         # m/s (more realistic for urban scenarios)
            'frequent_stops': 2,       # number of stops
            'many_turns': 3,           # number of turns
            'high_acceleration': 1.5,  # significant acceleration events
            'long_duration': 15.0,     # seconds
            'complex_path': 1.5,       # path tortuosity ratio
            'traffic_present': 1,      # other vehicles present
            'high_curvature': 2,       # direction changes
            'speed_variability': 2.0,  # speed standard deviation
            'aggressive_steering': 0.2 # steering angle threshold
        }
        
    def extract_comprehensive_features(self, log_file):
        """Extract the same 37-dimensional features as Phase 1"""
        try:
            recorder_file = f"{self.log_files_dir}/{log_file}"
            recorder_str = self.client.show_recorder_file_info(recorder_file, True)
            log = MetricsLog(recorder_str)
            
            ego_id = log.get_ego_vehicle_id()
            start_frame, end_frame = log.get_actor_alive_frames(ego_id)
            
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
            
            # Extract the same features as Phase 1
            temporal_features = self._extract_temporal_features(speeds, accelerations, len(sample_frames))
            behavioral_features = self._extract_behavioral_features(behaviors, steering_angles)
            spatial_features = self._extract_spatial_features(positions)
            speed_features = self._extract_speed_features(speeds, accelerations)
            
            # Traffic context
            vehicle_ids = log.get_actor_ids_with_type_id("vehicle.*")
            traffic_count = len([vid for vid in vehicle_ids if vid != ego_id])
            traffic_features = [traffic_count, min(traffic_count, 10), 1 if traffic_count > 0 else 0]
            
            # Combine all features (37 dimensions total)
            combined_vector = (temporal_features + behavioral_features + 
                             spatial_features + speed_features + traffic_features)
            
            return {
                'temporal': temporal_features,
                'behavioral': behavioral_features,
                'spatial': spatial_features,
                'speed': speed_features,
                'traffic': traffic_features,
                'combined': combined_vector
            }
            
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
        """Extract temporal characteristics (6 features)"""
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
            len([a for a in accelerations if abs(a) > 2.0])
        ]
    
    def _extract_behavioral_features(self, behaviors, steering_angles):
        """Extract behavioral pattern features (10 features)"""
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
        """Extract spatial movement characteristics (8 features)"""
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
            if cross_product > 50:
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
        """Extract speed profile characteristics (10 features)"""
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
    
    def convert_features_to_sets(self, features_dict):
        """
        Convert 37-dimensional feature vectors into categorical feature sets
        for Jaccard similarity analysis
        """
        scenario_sets = {}
        
        for scenario_name, features in features_dict.items():
            if not features or 'combined' not in features:
                continue
                
            feature_set = set()
            
            # Extract individual feature components
            temporal = features['temporal']      # [0:6]
            behavioral = features['behavioral']  # [6:16] 
            spatial = features['spatial']        # [16:24]
            speed = features['speed']           # [24:34]
            traffic = features['traffic']       # [34:37]
            
            # Convert temporal features to categorical sets
            if temporal[0] > self.feature_thresholds['long_duration']:  # Duration
                feature_set.add('long_scenario')
            if temporal[2] > self.feature_thresholds['frequent_stops']:  # Speed changes
                feature_set.add('dynamic_speed')
            if temporal[4] > self.feature_thresholds['speed_variability']:  # Speed std
                feature_set.add('variable_speed')
                
            # Convert behavioral features to categorical sets
            if behavioral[0] > self.feature_thresholds['frequent_stops']:  # Stops
                feature_set.add('frequent_stopping')
            if behavioral[3] > self.feature_thresholds['many_turns']:  # Turns
                feature_set.add('turning_scenario')
            if behavioral[7] > self.feature_thresholds['aggressive_steering']:  # Avg steering
                feature_set.add('active_steering')
            if behavioral[8] > self.feature_thresholds['aggressive_steering']:  # Max steering
                feature_set.add('sharp_maneuvers')
                
            # Convert spatial features to categorical sets
            if spatial[2] > self.feature_thresholds['complex_path']:  # Tortuosity
                feature_set.add('complex_path')
            if spatial[6] > self.feature_thresholds['high_curvature']:  # Direction changes
                feature_set.add('curved_trajectory')
            if spatial[0] > 500:  # Path length threshold
                feature_set.add('long_distance')
                
            # Convert speed features to categorical sets
            if speed[0] > self.feature_thresholds['high_speed']:  # Mean speed
                feature_set.add('high_speed_scenario')
            if speed[3] > self.feature_thresholds['high_speed']:  # Max speed
                feature_set.add('speed_peaks')
            if speed[9] > self.feature_thresholds['high_acceleration']:  # Hard accel/decel
                feature_set.add('aggressive_driving')
                
            # Convert traffic features to categorical sets
            if traffic[2] > 0:  # Traffic present
                feature_set.add('traffic_scenario')
            if traffic[0] > 5:  # Many vehicles
                feature_set.add('dense_traffic')
                
            # Add basic categorical features based on scenario name patterns
            scenario_lower = scenario_name.lower()
            if 'arg' in scenario_lower:
                feature_set.add('location_argentina')
            elif 'deu' in scenario_lower:
                feature_set.add('location_germany')
            elif 'bel' in scenario_lower:
                feature_set.add('location_belgium')
            elif 'chn' in scenario_lower:
                feature_set.add('location_china')
                
            scenario_sets[scenario_name] = feature_set
        
        return scenario_sets
    
    def jaccard_similarity(self, set1, set2):
        """Calculate Jaccard similarity coefficient between two sets"""
        if len(set1) == 0 and len(set2) == 0:
            return 1.0  # Both empty sets are identical
        
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        return intersection / union if union > 0 else 0.0
    
    def create_ground_truth(self, scenario_names):
        """
        Create ground truth based on scenario naming patterns
        Same approach as Phase 1 for consistency
        """
        ground_truth = {}
        
        for i, scenario1 in enumerate(scenario_names):
            for j, scenario2 in enumerate(scenario_names):
                if i >= j:
                    continue
                
                # Extract base names for comparison (same logic as Phase 1)
                # For names like "ARG_Carcarana-11_2_I-1-1", extract "ARG_Carcarana"
                if '-' in scenario1 and '-' in scenario2:
                    base1 = scenario1.split('-')[0]  # Get part before first dash
                    base2 = scenario2.split('-')[0]
                else:
                    base1 = '_'.join(scenario1.split('_')[:-1])  # Fallback to old method
                    base2 = '_'.join(scenario2.split('_')[:-1])
                
                # Scenarios are similar if they share the same base pattern
                is_similar = (base1 == base2) and (base1 != '' and base2 != '')
                ground_truth[(scenario1, scenario2)] = is_similar
        
        return ground_truth
    
    def evaluate_similarity_performance(self, scenario_sets, ground_truth, threshold=0.3):
        """Evaluate Jaccard similarity performance against ground truth"""
        predictions = []
        true_labels = []
        similarity_scores = []
        
        for (scenario1, scenario2), true_label in ground_truth.items():
            if scenario1 in scenario_sets and scenario2 in scenario_sets:
                similarity = self.jaccard_similarity(
                    scenario_sets[scenario1], 
                    scenario_sets[scenario2]
                )
                
                prediction = similarity >= threshold
                
                predictions.append(prediction)
                true_labels.append(true_label)
                similarity_scores.append(similarity)
        
        # Calculate metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_labels, predictions, average='binary'
        )
        accuracy = accuracy_score(true_labels, predictions)
        
        return {
            'threshold': threshold,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'accuracy': accuracy,
            'similarity_scores': similarity_scores,
            'predictions': predictions,
            'true_labels': true_labels
        }
    
    def run_analysis(self):
        """Run complete Phase 2 set-based similarity analysis"""
        print("=" * 60)
        print("Phase 2: Set-Based Similarity Analysis with Jaccard Coefficient")
        print("=" * 60)
        
        # Get log files
        log_files = [f for f in os.listdir(self.log_files_dir) if f.endswith('.log')]
        print(f"Found {len(log_files)} log files")
        
        # Extract features from each scenario
        print("\n1. Extracting 37-dimensional feature vectors...")
        features_dict = {}
        
        for i, log_file in enumerate(log_files):
            features = self.extract_comprehensive_features(log_file)
            if features:
                scenario_name = log_file.replace('.log', '')
                features_dict[scenario_name] = features
            
            if (i + 1) % 20 == 0:
                print(f"   Processed {i + 1}/{len(log_files)} scenarios")
        
        print(f"Successfully extracted features from {len(features_dict)} scenarios")
        
        # Convert to feature sets
        print("\n2. Converting features to categorical sets...")
        scenario_sets = self.convert_features_to_sets(features_dict)
        
        # Analyze feature set statistics
        print(f"\n3. Feature set statistics:")
        all_features = set()
        set_sizes = []
        
        for scenario_name, feature_set in scenario_sets.items():
            all_features.update(feature_set)
            set_sizes.append(len(feature_set))
        
        print(f"   Total unique features: {len(all_features)}")
        print(f"   Average set size: {np.mean(set_sizes):.2f}")
        print(f"   Set size range: {min(set_sizes)} - {max(set_sizes)}")
        print(f"   Most common features:")
        
        # Count feature frequencies
        feature_counts = Counter()
        for feature_set in scenario_sets.values():
            feature_counts.update(feature_set)
        
        for feature, count in feature_counts.most_common(10):
            print(f"     {feature}: {count} scenarios ({count/len(scenario_sets)*100:.1f}%)")
        
        # Create ground truth
        print("\n4. Creating ground truth from scenario naming patterns...")
        scenario_names = list(scenario_sets.keys())
        ground_truth = self.create_ground_truth(scenario_names)
        
        total_pairs = len(ground_truth)
        similar_pairs = sum(ground_truth.values())
        print(f"   Total scenario pairs: {total_pairs}")
        print(f"   Similar pairs (ground truth): {similar_pairs} ({similar_pairs/total_pairs*100:.1f}%)")
        
        # Test multiple thresholds
        print("\n5. Evaluating Jaccard similarity performance...")
        thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]
        results = []
        
        for threshold in thresholds:
            result = self.evaluate_similarity_performance(scenario_sets, ground_truth, threshold)
            results.append(result)
            print(f"   Threshold {threshold}: F1={result['f1']:.3f}, Acc={result['accuracy']:.3f}")
        
        # Find best threshold
        best_result = max(results, key=lambda x: x['f1'])
        print(f"\n6. Best performance: Threshold={best_result['threshold']}, F1={best_result['f1']:.3f}")
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed results
        results_data = {
            'experiment_info': {
                'phase': 'Phase 2: Set-Based Jaccard Similarity',
                'timestamp': timestamp,
                'total_scenarios': len(scenario_sets),
                'total_pairs': total_pairs,
                'similar_pairs': similar_pairs
            },
            'feature_statistics': {
                'total_unique_features': len(all_features),
                'average_set_size': float(np.mean(set_sizes)),
                'set_size_range': [int(min(set_sizes)), int(max(set_sizes))],
                'feature_frequencies': dict(feature_counts.most_common())
            },
            'threshold_results': [
                {
                    'threshold': r['threshold'],
                    'precision': float(r['precision']),
                    'recall': float(r['recall']),
                    'f1': float(r['f1']),
                    'accuracy': float(r['accuracy'])
                }
                for r in results
            ],
            'best_result': {
                'threshold': best_result['threshold'],
                'precision': float(best_result['precision']),
                'recall': float(best_result['recall']),
                'f1': float(best_result['f1']),
                'accuracy': float(best_result['accuracy'])
            },
            'scenario_sets': {name: list(feature_set) for name, feature_set in scenario_sets.items()}
        }
        
        output_file = f"phase2_jaccard_results_{timestamp}.json"
        with open(output_file, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        print(f"\n7. Results saved to: {output_file}")
        
        # Create visualization
        self.create_visualizations(results, scenario_sets, best_result, timestamp)
        
        return results_data
    
    def create_visualizations(self, results, scenario_sets, best_result, timestamp):
        """Create comprehensive visualizations for Phase 2 results"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Phase 2: Set-Based Jaccard Similarity Analysis Results', fontsize=16)
        
        # 1. Threshold performance curve
        thresholds = [r['threshold'] for r in results]
        f1_scores = [r['f1'] for r in results]
        accuracies = [r['accuracy'] for r in results]
        
        axes[0,0].plot(thresholds, f1_scores, 'o-', label='F1 Score', linewidth=2)
        axes[0,0].plot(thresholds, accuracies, 's-', label='Accuracy', linewidth=2)
        axes[0,0].axvline(x=best_result['threshold'], color='red', linestyle='--', alpha=0.7, label='Best F1')
        axes[0,0].set_xlabel('Jaccard Similarity Threshold')
        axes[0,0].set_ylabel('Performance Score')
        axes[0,0].set_title('Performance vs Threshold')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        
        # 2. Feature set size distribution
        set_sizes = [len(feature_set) for feature_set in scenario_sets.values()]
        axes[0,1].hist(set_sizes, bins=15, alpha=0.7, edgecolor='black')
        axes[0,1].set_xlabel('Feature Set Size')
        axes[0,1].set_ylabel('Number of Scenarios')
        axes[0,1].set_title('Distribution of Feature Set Sizes')
        axes[0,1].grid(True, alpha=0.3)
        
        # 3. Similarity score distribution
        similarity_scores = best_result['similarity_scores']
        axes[1,0].hist(similarity_scores, bins=20, alpha=0.7, edgecolor='black', color='green')
        axes[1,0].axvline(x=best_result['threshold'], color='red', linestyle='--', label='Best Threshold')
        axes[1,0].set_xlabel('Jaccard Similarity Score')
        axes[1,0].set_ylabel('Number of Scenario Pairs')
        axes[1,0].set_title('Distribution of Jaccard Similarity Scores')
        axes[1,0].legend()
        axes[1,0].grid(True, alpha=0.3)
        
        # 4. Feature frequency analysis
        feature_counts = Counter()
        for feature_set in scenario_sets.values():
            feature_counts.update(feature_set)
        
        top_features = feature_counts.most_common(10)
        features, counts = zip(*top_features)
        
        axes[1,1].barh(range(len(features)), counts, color='skyblue', edgecolor='black')
        axes[1,1].set_yticks(range(len(features)))
        axes[1,1].set_yticklabels(features)
        axes[1,1].set_xlabel('Number of Scenarios')
        axes[1,1].set_title('Top 10 Most Common Features')
        axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save the plot
        plot_filename = f"phase2_jaccard_analysis_{timestamp}.png"
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"   Visualization saved to: {plot_filename}")
        
        plt.show()

def main():
    """Main execution function"""
    # Configuration
    log_files_dir = "/home/ads/ads_testing/log_files"
    
    # Verify log files directory exists
    if not os.path.exists(log_files_dir):
        print(f"Error: Log files directory not found: {log_files_dir}")
        return
    
    # Run Phase 2 analysis
    analyzer = Phase2SetBasedSimilarityAnalysis(log_files_dir)
    results = analyzer.run_analysis()
    
    print("\nPhase 2 Set-Based Jaccard Similarity Analysis Complete!")
    print(f"Best F1 Score: {results['best_result']['f1']:.3f}")
    print(f"Best Accuracy: {results['best_result']['accuracy']:.3f}")

if __name__ == "__main__":
    main()
