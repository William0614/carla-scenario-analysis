"""
Phase 3: Sequence-Based Similarity Analysis for CARLA Scenarios

This module implements sequence-based similarity metrics to analyze CARLA scenario logs.
Unlike Phase 1 (distance-based on 37D vectors) and Phase 2 (set-based Jaccard),
Phase 3 focuses on the temporal sequence of driving actions.

Sequence-based metrics:
1. Edit Distance (Levenshtein) - minimum operations to transform one sequence to another
2. Longest Common Subsequence (LCS) - length of longest shared subsequence
3. Dynamic Time Warping (DTW) - optimal alignment between sequences
4. N-gram Jaccard - similarity of sequence fragments
5. Sequence Alignment Score - biological sequence alignment adapted for driving

Author: Research Team
Date: September 2024
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from collections import defaultdict, Counter
import warnings
warnings.filterwarnings('ignore')

# Import CARLA and scenario runner modules
import sys
sys.path.append('/home/ads/ads_testing/scenario_runner/scenario_runner-0.9.15')
from srunner.metrics.tools.metrics_log import MetricsLog
import carla

class Phase3SequenceAnalyzer:
    """
    Phase 3: Sequence-based similarity analysis for CARLA scenarios.
    
    Extracts driving action sequences and applies various sequence similarity metrics
    to identify scenarios with similar behavioral patterns.
    """
    
    def __init__(self, log_directory="/home/ads/ads_testing/log_files"):
        self.log_directory = log_directory
        self.extracted_sequences = {}
        self.similarity_results = {}
        
        # Simple action classification thresholds (from our earlier discussion)
        self.action_thresholds = {
            'speed_threshold_stop': 0.5,
            'speed_threshold_cruise': 1.0,
            'speed_threshold_fast': 5.0,
            'brake_threshold': 0.3,
            'acceleration_threshold': 1.0,
            'steer_threshold_light': 0.05,
            'steer_threshold_turn': 0.15
        }
        
    def classify_action(self, speed, speed_change, throttle, brake, steer):
        """
        Classify driving action based on vehicle control inputs.
        
        Uses simple action classification (9 action types) as discussed:
        - STOP, BRAKE, ACCELERATE
        - TURN_LEFT, TURN_RIGHT, STEER_LEFT, STEER_RIGHT  
        - CRUISE_FAST, CRUISE, IDLE
        """
        
        # Priority order: Stop > Brake > Acceleration > Turning > Cruising
        
        if speed < self.action_thresholds['speed_threshold_stop']:
            return "STOP"
        
        if brake > self.action_thresholds['brake_threshold']:
            return "BRAKE"
        
        if speed_change > self.action_thresholds['acceleration_threshold']:
            return "ACCELERATE"
        
        # Steering actions (with speed context)
        if abs(steer) > self.action_thresholds['steer_threshold_turn']:
            if steer > 0:
                return "TURN_RIGHT"
            else:
                return "TURN_LEFT"
        elif abs(steer) > self.action_thresholds['steer_threshold_light']:
            if steer > 0:
                return "STEER_RIGHT"
            else:
                return "STEER_LEFT"
        
        # Speed-based actions
        if speed > self.action_thresholds['speed_threshold_fast']:
            return "CRUISE_FAST"
        elif speed > self.action_thresholds['speed_threshold_cruise']:
            return "CRUISE"
        else:
            return "IDLE"
    
    def extract_action_sequence(self, log_file):
        """
        Extract sequence of driving actions from a CARLA log file.
        
        Returns:
            list: Sequence of action strings representing driving behavior
        """
        log_path = os.path.join(self.log_directory, log_file)
        
        try:
            # Use CARLA client to get recorder info
            import carla
            client = carla.Client('localhost', 2000)
            client.set_timeout(10.0)
            
            # Get recorder info as string
            recorder_info = client.show_recorder_file_info(log_path, True)
            
            # Initialize MetricsLog with recorder info string
            log = MetricsLog(recorder_info)
            
            # Get ego vehicle ID
            ego_id = log.get_ego_vehicle_id()
            if ego_id is None:
                # Try alternative method to find ego vehicle
                vehicle_ids = log.get_actor_ids_with_type_id("vehicle.*")
                if vehicle_ids:
                    ego_id = vehicle_ids[0]  # Use first vehicle as fallback
            
            if ego_id is None:
                return []
            
            # Extract frame-by-frame data
            frame_count = log.get_total_frame_count()
            if frame_count <= 10:  # Skip very short scenarios
                return []
            
            actions = []
            prev_speed = 0
            
            # Sample every 5th frame to reduce noise while maintaining sequence
            sample_interval = max(1, frame_count // 100)  # Aim for ~100 actions per scenario
            
            for frame in range(0, frame_count, sample_interval):
                try:
                    # Get vehicle state
                    velocity = log.get_actor_velocity(ego_id, frame)
                    control = log.get_vehicle_control(ego_id, frame)
                    
                    if velocity is None or control is None:
                        continue
                    
                    # Calculate speed and speed change
                    speed = (velocity.x**2 + velocity.y**2 + velocity.z**2)**0.5
                    speed_change = speed - prev_speed
                    prev_speed = speed
                    
                    # Extract control inputs
                    throttle = getattr(control, 'throttle', 0)
                    brake = getattr(control, 'brake', 0)
                    steer = getattr(control, 'steer', 0)
                    
                    # Classify action
                    action = self.classify_action(speed, speed_change, throttle, brake, steer)
                    actions.append(action)
                    
                except Exception as e:
                    continue
            
            # Remove consecutive duplicates to focus on behavioral changes
            if actions:
                compressed_actions = [actions[0]]
                for action in actions[1:]:
                    if action != compressed_actions[-1]:
                        compressed_actions.append(action)
                return compressed_actions
            
            return []
            
        except Exception as e:
            print(f"Error processing {log_file}: {e}")
            return []
    
    def edit_distance(self, seq1, seq2):
        """
        Calculate Edit Distance (Levenshtein) between two sequences.
        
        Returns normalized distance [0, 1] where 0 = identical, 1 = completely different
        """
        if not seq1 and not seq2:
            return 0.0
        if not seq1 or not seq2:
            return 1.0
        
        m, n = len(seq1), len(seq2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        # Initialize base cases
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
        
        # Fill DP table
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i-1] == seq2[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
        
        # Normalize by maximum possible distance
        max_len = max(m, n)
        return dp[m][n] / max_len if max_len > 0 else 0.0
    
    def longest_common_subsequence(self, seq1, seq2):
        """
        Calculate Longest Common Subsequence (LCS) similarity.
        
        Returns normalized similarity [0, 1] where 1 = high similarity, 0 = no common subsequence
        """
        if not seq1 or not seq2:
            return 0.0
        
        m, n = len(seq1), len(seq2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i-1] == seq2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        
        lcs_length = dp[m][n]
        max_len = max(m, n)
        return lcs_length / max_len if max_len > 0 else 0.0
    
    def dynamic_time_warping(self, seq1, seq2):
        """
        Calculate Dynamic Time Warping (DTW) similarity between sequences.
        
        Uses action-based cost matrix for driving behaviors.
        Returns normalized similarity [0, 1] where 1 = high similarity
        """
        if not seq1 or not seq2:
            return 0.0
        
        # Define cost matrix for action transitions
        actions = ["STOP", "BRAKE", "ACCELERATE", "TURN_LEFT", "TURN_RIGHT", 
                  "STEER_LEFT", "STEER_RIGHT", "CRUISE_FAST", "CRUISE", "IDLE"]
        
        # Create cost matrix (semantic similarity between actions)
        cost_matrix = {}
        for a1 in actions:
            for a2 in actions:
                if a1 == a2:
                    cost_matrix[(a1, a2)] = 0.0  # Identical actions
                elif (a1, a2) in [("CRUISE", "CRUISE_FAST"), ("STEER_LEFT", "TURN_LEFT"), 
                                ("STEER_RIGHT", "TURN_RIGHT"), ("BRAKE", "STOP")]:
                    cost_matrix[(a1, a2)] = 0.3  # Similar actions
                elif a1.split('_')[0] == a2.split('_')[0]:  # Same action family
                    cost_matrix[(a1, a2)] = 0.5
                else:
                    cost_matrix[(a1, a2)] = 1.0  # Different actions
        
        m, n = len(seq1), len(seq2)
        dtw = [[float('inf')] * (n + 1) for _ in range(m + 1)]
        dtw[0][0] = 0
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                cost = cost_matrix.get((seq1[i-1], seq2[j-1]), 1.0)
                dtw[i][j] = cost + min(dtw[i-1][j],    # insertion
                                     dtw[i][j-1],    # deletion
                                     dtw[i-1][j-1])  # match
        
        # Normalize by path length
        max_cost = max(m, n)
        dtw_distance = dtw[m][n] / max_cost if max_cost > 0 else 0
        return max(0, 1 - dtw_distance)  # Convert distance to similarity
    
    def ngram_jaccard(self, seq1, seq2, n=2):
        """
        Calculate N-gram Jaccard similarity between sequences.
        
        Breaks sequences into n-grams and applies Jaccard coefficient.
        """
        def get_ngrams(sequence, n):
            if len(sequence) < n:
                return set([tuple(sequence)])
            return set([tuple(sequence[i:i+n]) for i in range(len(sequence) - n + 1)])
        
        ngrams1 = get_ngrams(seq1, n)
        ngrams2 = get_ngrams(seq2, n)
        
        if not ngrams1 and not ngrams2:
            return 1.0
        if not ngrams1 or not ngrams2:
            return 0.0
        
        intersection = len(ngrams1.intersection(ngrams2))
        union = len(ngrams1.union(ngrams2))
        
        return intersection / union if union > 0 else 0.0
    
    def sequence_alignment_score(self, seq1, seq2):
        """
        Calculate sequence alignment score using Smith-Waterman algorithm.
        
        Adapted for driving action sequences with match/mismatch scoring.
        """
        if not seq1 or not seq2:
            return 0.0
        
        match_score = 2
        mismatch_score = -1
        gap_score = -1
        
        m, n = len(seq1), len(seq2)
        score_matrix = [[0] * (n + 1) for _ in range(m + 1)]
        max_score = 0
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                match = score_matrix[i-1][j-1] + (match_score if seq1[i-1] == seq2[j-1] else mismatch_score)
                delete = score_matrix[i-1][j] + gap_score
                insert = score_matrix[i][j-1] + gap_score
                
                score_matrix[i][j] = max(0, match, delete, insert)
                max_score = max(max_score, score_matrix[i][j])
        
        # Normalize by theoretical maximum score
        theoretical_max = min(m, n) * match_score
        return max_score / theoretical_max if theoretical_max > 0 else 0.0
    
    def create_ground_truth(self, log_files):
        """
        Create ground truth similarity labels based on scenario naming patterns.
        
        Same approach as Phase 1 and Phase 2 for consistency.
        """
        ground_truth = {}
        
        for i, file1 in enumerate(log_files):
            for j, file2 in enumerate(log_files[i+1:], i+1):
                # Extract scenario identifiers
                base1 = file1.replace('.log', '').rsplit('_', 2)[0] if '_' in file1 else file1
                base2 = file2.replace('.log', '').rsplit('_', 2)[0] if '_' in file2 else file2
                
                # Calculate similarity based on naming patterns
                similarity_score = 0
                
                # Same scenario base name
                if base1 == base2:
                    similarity_score += 0.4
                
                # Same country/region
                if file1.split('_')[0] == file2.split('_')[0]:
                    similarity_score += 0.2
                
                # Same location (country + city)
                if '_'.join(file1.split('_')[:2]) == '_'.join(file2.split('_')[:2]):
                    similarity_score += 0.3
                
                # Same scenario pattern
                if file1.split('.')[-2].split('_')[-1] == file2.split('.')[-2].split('_')[-1]:
                    similarity_score += 0.1
                
                # Binary classification (similar if score >= 0.6)
                is_similar = similarity_score >= 0.6
                ground_truth[(file1, file2)] = is_similar
        
        return ground_truth
    
    def evaluate_performance(self, similarities, ground_truth, thresholds):
        """
        Evaluate performance of similarity metrics against ground truth.
        
        Returns performance metrics for each threshold.
        """
        results = {}
        
        for threshold in thresholds:
            tp = fp = tn = fn = 0
            
            for (file1, file2), predicted_sim in similarities.items():
                actual_similar = ground_truth.get((file1, file2), False)
                predicted_similar = predicted_sim >= threshold
                
                if actual_similar and predicted_similar:
                    tp += 1
                elif actual_similar and not predicted_similar:
                    fn += 1
                elif not actual_similar and predicted_similar:
                    fp += 1
                else:
                    tn += 1
            
            # Calculate metrics
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            accuracy = (tp + tn) / (tp + fp + tn + fn) if (tp + fp + tn + fn) > 0 else 0
            
            results[threshold] = {
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'accuracy': accuracy,
                'tp': tp,
                'fp': fp,
                'tn': tn,
                'fn': fn
            }
        
        return results
    
    def run_analysis(self):
        """
        Run complete Phase 3 sequence-based similarity analysis.
        """
        print("=" * 60)
        print("Phase 3: Sequence-Based Similarity Analysis")
        print("=" * 60)
        
        # Get log files
        log_files = [f for f in os.listdir(self.log_directory) if f.endswith('.log')]
        print(f"Found {len(log_files)} log files\n")
        
        # Extract action sequences
        print("1. Extracting action sequences from log files...")
        successful_extractions = 0
        
        for i, log_file in enumerate(log_files):
            if (i + 1) % 20 == 0:
                print(f"   Processed {i + 1}/{len(log_files)} scenarios")
            
            sequence = self.extract_action_sequence(log_file)
            if sequence:
                self.extracted_sequences[log_file] = sequence
                successful_extractions += 1
        
        print(f"Successfully extracted sequences from {successful_extractions} scenarios\n")
        
        if successful_extractions < 2:
            print("❌ Not enough sequences extracted for analysis")
            return None
        
        # Analyze sequence characteristics
        print("2. Analyzing sequence characteristics...")
        
        all_sequences = list(self.extracted_sequences.values())
        sequence_lengths = [len(seq) for seq in all_sequences]
        all_actions = [action for seq in all_sequences for action in seq]
        action_counts = Counter(all_actions)
        
        print(f"   Average sequence length: {np.mean(sequence_lengths):.1f}")
        print(f"   Sequence length range: {min(sequence_lengths)} - {max(sequence_lengths)}")
        print(f"   Total unique actions: {len(action_counts)}")
        print(f"   Most common actions:")
        for action, count in action_counts.most_common(5):
            percentage = 100 * count / len(all_actions)
            print(f"     {action}: {count} occurrences ({percentage:.1f}%)")
        print()
        
        # Calculate similarities using different sequence metrics
        print("3. Calculating sequence similarities...")
        
        sequence_files = list(self.extracted_sequences.keys())
        metrics = {
            'edit_distance': {},
            'lcs_similarity': {},
            'dtw_similarity': {},
            'ngram_jaccard': {},
            'alignment_score': {}
        }
        
        total_pairs = len(sequence_files) * (len(sequence_files) - 1) // 2
        pair_count = 0
        
        for i, file1 in enumerate(sequence_files):
            for j, file2 in enumerate(sequence_files[i+1:], i+1):
                pair_count += 1
                if pair_count % 1000 == 0:
                    print(f"   Processed {pair_count}/{total_pairs} pairs")
                
                seq1 = self.extracted_sequences[file1]
                seq2 = self.extracted_sequences[file2]
                
                # Calculate different similarity metrics
                # Edit distance (convert distance to similarity)
                edit_dist = self.edit_distance(seq1, seq2)
                metrics['edit_distance'][(file1, file2)] = 1 - edit_dist
                
                # LCS similarity
                metrics['lcs_similarity'][(file1, file2)] = self.longest_common_subsequence(seq1, seq2)
                
                # DTW similarity
                metrics['dtw_similarity'][(file1, file2)] = self.dynamic_time_warping(seq1, seq2)
                
                # N-gram Jaccard
                metrics['ngram_jaccard'][(file1, file2)] = self.ngram_jaccard(seq1, seq2, n=2)
                
                # Sequence alignment
                metrics['alignment_score'][(file1, file2)] = self.sequence_alignment_score(seq1, seq2)
        
        print(f"Calculated similarities for {pair_count} scenario pairs\n")
        
        # Create ground truth
        print("4. Creating ground truth from scenario naming patterns...")
        ground_truth = self.create_ground_truth(sequence_files)
        
        similar_pairs = sum(ground_truth.values())
        total_pairs = len(ground_truth)
        similarity_rate = similar_pairs / total_pairs
        
        print(f"   Total scenario pairs: {total_pairs}")
        print(f"   Similar pairs (ground truth): {similar_pairs} ({similarity_rate:.1%})")
        print()
        
        # Evaluate each metric
        print("5. Evaluating sequence similarity metrics...")
        thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        
        metric_results = {}
        
        for metric_name, similarities in metrics.items():
            print(f"\n   Evaluating {metric_name}:")
            results = self.evaluate_performance(similarities, ground_truth, thresholds)
            metric_results[metric_name] = results
            
            # Find best threshold
            best_threshold = max(results.keys(), key=lambda t: results[t]['f1'])
            best_f1 = results[best_threshold]['f1']
            best_acc = results[best_threshold]['accuracy']
            
            print(f"     Best threshold: {best_threshold}")
            print(f"     Best F1 score: {best_f1:.3f}")
            print(f"     Best accuracy: {best_acc:.3f}")
        
        # Find overall best metric
        print("\n6. Best performing sequence metrics:")
        best_metrics = []
        
        for metric_name, results in metric_results.items():
            best_threshold = max(results.keys(), key=lambda t: results[t]['f1'])
            best_f1 = results[best_threshold]['f1']
            best_acc = results[best_threshold]['accuracy']
            best_metrics.append((metric_name, best_f1, best_acc, best_threshold))
        
        best_metrics.sort(key=lambda x: x[1], reverse=True)  # Sort by F1 score
        
        for i, (metric, f1, acc, threshold) in enumerate(best_metrics, 1):
            print(f"   {i}. {metric}: F1={f1:.3f}, Acc={acc:.1%} (threshold={threshold})")
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"phase3_sequence_results_{timestamp}.json"
        
        results_data = {
            'timestamp': timestamp,
            'method': 'Phase 3: Sequence-Based Similarity Analysis',
            'dataset_size': successful_extractions,
            'total_pairs': total_pairs,
            'ground_truth_similarity_rate': similarity_rate,
            'sequence_statistics': {
                'avg_length': float(np.mean(sequence_lengths)),
                'length_range': [int(min(sequence_lengths)), int(max(sequence_lengths))],
                'unique_actions': len(action_counts),
                'action_distribution': dict(action_counts.most_common(10))
            },
            'metric_performance': {}
        }
        
        for metric_name, results in metric_results.items():
            best_threshold = max(results.keys(), key=lambda t: results[t]['f1'])
            results_data['metric_performance'][metric_name] = {
                'best_threshold': best_threshold,
                'best_f1': results[best_threshold]['f1'],
                'best_accuracy': results[best_threshold]['accuracy'],
                'best_precision': results[best_threshold]['precision'],
                'best_recall': results[best_threshold]['recall'],
                'all_thresholds': results
            }
        
        # Save all results
        with open(results_file, 'w') as f:
            json.dump(results_data, f, indent=2, default=str)
        
        # Create visualization
        self.create_visualizations(metric_results, timestamp)
        
        print(f"\n7. Results saved to: {results_file}")
        print(f"   Visualization saved to: phase3_sequence_analysis_{timestamp}.png")
        print("\nPhase 3 Sequence-Based Similarity Analysis Complete!")
        
        # Return best performing metric info
        best_metric_name, best_f1, best_acc, best_threshold = best_metrics[0]
        print(f"Best Performing Metric: {best_metric_name}")
        print(f"Best F1 Score: {best_f1:.3f}")
        print(f"Best Accuracy: {best_acc:.1%}")
        
        return results_data
    
    def create_visualizations(self, metric_results, timestamp):
        """Create comprehensive visualizations for Phase 3 results."""
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Phase 3: Sequence-Based Similarity Metrics Performance', fontsize=16, fontweight='bold')
        
        # 1. F1 Score Comparison
        ax1 = axes[0, 0]
        metrics_names = list(metric_results.keys())
        best_f1_scores = []
        best_thresholds = []
        
        for metric_name in metrics_names:
            results = metric_results[metric_name]
            best_threshold = max(results.keys(), key=lambda t: results[t]['f1'])
            best_f1_scores.append(results[best_threshold]['f1'])
            best_thresholds.append(best_threshold)
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(metrics_names)))
        bars = ax1.bar(range(len(metrics_names)), best_f1_scores, color=colors)
        ax1.set_xlabel('Sequence Similarity Metrics')
        ax1.set_ylabel('Best F1 Score')
        ax1.set_title('F1 Score Comparison')
        ax1.set_xticks(range(len(metrics_names)))
        ax1.set_xticklabels([name.replace('_', '\n') for name in metrics_names], rotation=45, ha='right')
        ax1.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, score in zip(bars, best_f1_scores):
            height = bar.get_height()
            ax1.annotate(f'{score:.3f}', xy=(bar.get_x() + bar.get_width()/2, height),
                        ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # 2. Accuracy Comparison
        ax2 = axes[0, 1]
        best_accuracies = []
        
        for metric_name in metrics_names:
            results = metric_results[metric_name]
            best_threshold = max(results.keys(), key=lambda t: results[t]['f1'])
            best_accuracies.append(results[best_threshold]['accuracy'])
        
        bars = ax2.bar(range(len(metrics_names)), best_accuracies, color=colors)
        ax2.set_xlabel('Sequence Similarity Metrics')
        ax2.set_ylabel('Best Accuracy')
        ax2.set_title('Accuracy Comparison')
        ax2.set_xticks(range(len(metrics_names)))
        ax2.set_xticklabels([name.replace('_', '\n') for name in metrics_names], rotation=45, ha='right')
        ax2.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, acc in zip(bars, best_accuracies):
            height = bar.get_height()
            ax2.annotate(f'{acc:.1%}', xy=(bar.get_x() + bar.get_width()/2, height),
                        ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # 3. Threshold Analysis
        ax3 = axes[0, 2]
        thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        
        for metric_name, color in zip(metrics_names, colors):
            results = metric_results[metric_name]
            f1_scores = [results[t]['f1'] for t in thresholds]
            ax3.plot(thresholds, f1_scores, marker='o', label=metric_name.replace('_', ' '), 
                    color=color, linewidth=2, markersize=4)
        
        ax3.set_xlabel('Similarity Threshold')
        ax3.set_ylabel('F1 Score')
        ax3.set_title('F1 Score vs Threshold')
        ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax3.grid(True, alpha=0.3)
        
        # 4. Precision-Recall Analysis
        ax4 = axes[1, 0]
        
        for metric_name, color in zip(metrics_names, colors):
            results = metric_results[metric_name]
            precisions = [results[t]['precision'] for t in thresholds]
            recalls = [results[t]['recall'] for t in thresholds]
            ax4.plot(recalls, precisions, marker='o', label=metric_name.replace('_', ' '), 
                    color=color, linewidth=2, markersize=4)
        
        ax4.set_xlabel('Recall')
        ax4.set_ylabel('Precision')
        ax4.set_title('Precision-Recall Curves')
        ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax4.grid(True, alpha=0.3)
        
        # 5. Performance Heatmap
        ax5 = axes[1, 1]
        
        # Create performance matrix
        performance_matrix = np.zeros((len(metrics_names), len(thresholds)))
        
        for i, metric_name in enumerate(metrics_names):
            results = metric_results[metric_name]
            for j, threshold in enumerate(thresholds):
                performance_matrix[i, j] = results[threshold]['f1']
        
        im = ax5.imshow(performance_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
        ax5.set_xlabel('Similarity Threshold')
        ax5.set_ylabel('Similarity Metrics')
        ax5.set_title('F1 Score Heatmap')
        ax5.set_xticks(range(len(thresholds)))
        ax5.set_xticklabels(thresholds)
        ax5.set_yticks(range(len(metrics_names)))
        ax5.set_yticklabels([name.replace('_', ' ') for name in metrics_names])
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax5)
        cbar.set_label('F1 Score', rotation=270, labelpad=15)
        
        # Add text annotations
        for i in range(len(metrics_names)):
            for j in range(len(thresholds)):
                text = ax5.text(j, i, f'{performance_matrix[i, j]:.2f}',
                              ha="center", va="center", color="black", fontsize=8)
        
        # 6. Summary Statistics
        ax6 = axes[1, 2]
        ax6.axis('off')
        
        # Create summary table
        summary_text = "Phase 3 Summary\n" + "="*20 + "\n\n"
        
        # Best performers
        best_metrics = []
        for metric_name in metrics_names:
            results = metric_results[metric_name]
            best_threshold = max(results.keys(), key=lambda t: results[t]['f1'])
            best_f1 = results[best_threshold]['f1']
            best_acc = results[best_threshold]['accuracy']
            best_metrics.append((metric_name, best_f1, best_acc, best_threshold))
        
        best_metrics.sort(key=lambda x: x[1], reverse=True)
        
        summary_text += "Top 3 Performers:\n"
        for i, (metric, f1, acc, threshold) in enumerate(best_metrics[:3], 1):
            summary_text += f"{i}. {metric.replace('_', ' ').title()}\n"
            summary_text += f"   F1: {f1:.3f}\n"
            summary_text += f"   Acc: {acc:.1%}\n"
            summary_text += f"   Threshold: {threshold}\n\n"
        
        # Add key insights
        summary_text += "Key Insights:\n"
        best_metric = best_metrics[0]
        summary_text += f"• Best: {best_metric[0].replace('_', ' ').title()}\n"
        summary_text += f"• F1 improvement needed vs Phase 1/2\n"
        summary_text += f"• Sequence patterns captured\n"
        summary_text += f"• {len(thresholds)} thresholds tested\n"
        
        ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes, fontsize=11,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(f'phase3_sequence_analysis_{timestamp}.png', dpi=300, bbox_inches='tight')
        plt.close()

if __name__ == "__main__":
    # Run Phase 3 analysis
    analyzer = Phase3SequenceAnalyzer()
    results = analyzer.run_analysis()
    
    if results:
        print("\n" + "="*60)
        print("PHASE 3 SEQUENCE ANALYSIS SUMMARY")
        print("="*60)
        
        best_metric = max(results['metric_performance'].items(), 
                         key=lambda x: x[1]['best_f1'])
        
        print(f"Dataset: {results['dataset_size']} scenarios processed")
        print(f"Total pairs analyzed: {results['total_pairs']:,}")
        print(f"Ground truth similarity: {results['ground_truth_similarity_rate']:.1%}")
        print()
        print(f"🏆 BEST PERFORMING METRIC: {best_metric[0].upper()}")
        print(f"   F1 Score: {best_metric[1]['best_f1']:.3f}")
        print(f"   Accuracy: {best_metric[1]['best_accuracy']:.1%}")
        print(f"   Optimal Threshold: {best_metric[1]['best_threshold']}")
        print()
        print("Sequence characteristics:")
        stats = results['sequence_statistics']
        print(f"   Average sequence length: {stats['avg_length']:.1f} actions")
        print(f"   Length range: {stats['length_range'][0]}-{stats['length_range'][1]} actions")
        print(f"   Unique action types: {stats['unique_actions']}")
        print()
        print("Phase 3 Complete! 🎯")
