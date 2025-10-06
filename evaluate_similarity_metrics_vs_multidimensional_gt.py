#!/usr/bin/env python3
"""
Comprehensive Similarity Metric Evaluation vs Multi-Dimensional Ground Truth

This script evaluates ALL similarity metrics from Phases 1, 2, and 3 against
our new multi-dimensional ground truth to determine which method performs best
with the improved ground truth baseline.

Previously, N-gram Jaccard achieved F1=0.702 vs filename-based ground truth.
Now we test all methods vs behavioral + spatial + traffic + contextual ground truth.
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from collections import Counter, defaultdict
import warnings
warnings.filterwarnings('ignore')

# Import CARLA and scenario runner modules
sys.path.append('/home/ads/CARLA_0.9.15/scenario_runner/scenario_runner-0.9.15')
from srunner.metrics.tools.metrics_log import MetricsLog
import carla

# Import similarity calculation utilities
from scipy.spatial.distance import euclidean, cityblock, cosine, minkowski
from scipy.stats import pearsonr
from scipy.spatial import distance
import math
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

class ComprehensiveSimilarityEvaluator:
    """
    Evaluate all similarity metrics against multi-dimensional ground truth.
    """
    
    def __init__(self, multidim_gt_file):
        """Load multi-dimensional ground truth for comparison."""
        print(f"Loading multi-dimensional ground truth from: {multidim_gt_file}")
        
        with open(multidim_gt_file, 'r') as f:
            self.gt_data = json.load(f)
        
        self.scenario_files = self.gt_data['scenario_files']
        self.gt_binary_matrix = np.array(self.gt_data['binary_matrix'])
        self.gt_similarity_matrix = np.array(self.gt_data['similarity_matrix'])
        
        print(f"✅ Ground truth loaded: {len(self.scenario_files)} scenarios")
        print(f"   Binary threshold: {self.gt_data['metadata']['similarity_threshold']}")
        print(f"   Similarity rate: {self.gt_data['metadata']['similarity_rate']:.1%}")
        
        # Initialize CARLA client for feature extraction
        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(10.0)
        
        # Cache for extracted features to avoid recomputation
        self.feature_cache = {}
        if 'extracted_features' in self.gt_data:
            print("✅ Using cached features from ground truth file")
            self.feature_cache = self.gt_data['extracted_features']
        
    def extract_scenario_features(self, log_file):
        """Extract features needed for similarity calculations."""
        if log_file in self.feature_cache:
            return self.feature_cache[log_file]
        
        # Same extraction logic as multi-dimensional ground truth
        # (Implementation would be similar to MultiDimensionalGroundTruth.extract_scenario_features)
        # For now, use cached features from the ground truth file
        return None
    
    def evaluate_phase1_distance_metrics(self):
        """Evaluate Phase 1 distance-based similarity metrics."""
        print("\n🔍 EVALUATING PHASE 1 DISTANCE METRICS")
        print("=" * 50)
        
        results = {}
        
        # Extract statistical features for distance calculations
        statistical_features = self._extract_statistical_features()
        
        if not statistical_features:
            print("❌ Could not extract statistical features")
            return results
        
        # Distance metrics from Phase 1
        distance_metrics = {
            'euclidean': self._euclidean_similarity,
            'manhattan': self._manhattan_similarity, 
            'cosine': self._cosine_similarity,
            'minkowski_p0.5': lambda x, y: self._minkowski_similarity(x, y, p=0.5),
            'minkowski_p3': lambda x, y: self._minkowski_similarity(x, y, p=3)
        }
        
        # Normalization methods from Phase 1
        normalization_methods = {
            'none': lambda x: x,
            'min_max': self._min_max_normalize,
            'z_score': self._z_score_normalize,
            'robust': self._robust_normalize
        }
        
        for norm_name, norm_func in normalization_methods.items():
            print(f"\n  Testing {norm_name} normalization...")
            
            # Apply normalization
            normalized_features = {}
            if norm_name == 'none':
                normalized_features = statistical_features.copy()
            else:
                all_vectors = list(statistical_features.values())
                normalized_vectors = norm_func(all_vectors)
                normalized_features = dict(zip(statistical_features.keys(), normalized_vectors))
            
            # Test each distance metric
            for metric_name, metric_func in distance_metrics.items():
                print(f"    {metric_name}...", end='')
                
                try:
                    # Calculate similarity matrix
                    similarity_matrix = self._calculate_pairwise_similarities(
                        normalized_features, metric_func
                    )
                    
                    # Evaluate against ground truth with multiple thresholds
                    best_result = self._evaluate_similarity_matrix(similarity_matrix)
                    
                    full_name = f"{norm_name}_{metric_name}"
                    results[full_name] = best_result
                    
                    print(f" F1={best_result['f1_score']:.3f}")
                    
                except Exception as e:
                    print(f" ERROR: {e}")
                    continue
        
        return results
    
    def evaluate_phase2_set_metrics(self):
        """Evaluate Phase 2 set-based similarity metrics."""
        print("\n🔍 EVALUATING PHASE 2 SET METRICS")
        print("=" * 50)
        
        results = {}
        
        # Extract set-based features
        set_features = self._extract_set_features()
        
        if not set_features:
            print("❌ Could not extract set features")
            return results
        
        # Set-based metrics from Phase 2
        set_metrics = {
            'jaccard': self._jaccard_similarity,
            'dice': self._dice_similarity,
            'overlap': self._overlap_similarity,
            'cosine_sets': self._cosine_set_similarity
        }
        
        for metric_name, metric_func in set_metrics.items():
            print(f"  Testing {metric_name}...", end='')
            
            try:
                # Calculate similarity matrix
                similarity_matrix = self._calculate_pairwise_set_similarities(
                    set_features, metric_func
                )
                
                # Evaluate against ground truth
                best_result = self._evaluate_similarity_matrix(similarity_matrix)
                results[metric_name] = best_result
                
                print(f" F1={best_result['f1_score']:.3f}")
                
            except Exception as e:
                print(f" ERROR: {e}")
                continue
        
        return results
    
    def evaluate_phase3_sequence_metrics(self):
        """Evaluate Phase 3 sequence-based similarity metrics."""
        print("\n🔍 EVALUATING PHASE 3 SEQUENCE METRICS")
        print("=" * 50)
        
        results = {}
        
        # Extract action sequences
        action_sequences = self._extract_action_sequences()
        
        if not action_sequences:
            print("❌ Could not extract action sequences")
            return results
        
        # Sequence-based metrics from Phase 3
        sequence_metrics = {
            'edit_distance': self._edit_distance_similarity,
            'lcs': self._lcs_similarity,
            'dtw': self._dtw_similarity,
            'ngram_jaccard': self._ngram_jaccard_similarity,
            'global_alignment': self._global_alignment_similarity
        }
        
        for metric_name, metric_func in sequence_metrics.items():
            print(f"  Testing {metric_name}...", end='')
            
            try:
                # Calculate similarity matrix
                similarity_matrix = self._calculate_pairwise_sequence_similarities(
                    action_sequences, metric_func
                )
                
                # Evaluate against ground truth
                best_result = self._evaluate_similarity_matrix(similarity_matrix)
                results[metric_name] = best_result
                
                print(f" F1={best_result['f1_score']:.3f}")
                
            except Exception as e:
                print(f" ERROR: {e}")
                continue
        
        return results
    
    def _extract_statistical_features(self):
        """Extract statistical features for Phase 1 distance metrics."""
        features = {}
        
        if not self.feature_cache:
            return features
        
        for scenario_file in self.scenario_files:
            if scenario_file not in self.feature_cache:
                continue
            
            scenario_features = self.feature_cache[scenario_file]
            
            # Extract statistical vector (similar to Phase 1)
            feature_vector = []
            
            # Behavioral stats
            if 'behavioral' in scenario_features and 'behavioral_stats' in scenario_features['behavioral']:
                stats = scenario_features['behavioral']['behavioral_stats']
                feature_vector.extend([
                    stats.get('avg_speed', 0),
                    stats.get('speed_std', 0), 
                    stats.get('max_speed', 0),
                    stats.get('avg_throttle', 0),
                    stats.get('avg_brake', 0),
                    stats.get('steering_variance', 0)
                ])
            
            # Spatial stats
            if 'spatial' in scenario_features and 'spatial_stats' in scenario_features['spatial']:
                stats = scenario_features['spatial']['spatial_stats']
                feature_vector.extend([
                    stats.get('total_distance', 0),
                    stats.get('displacement', 0),
                    stats.get('bbox_width', 0),
                    stats.get('bbox_height', 0),
                    stats.get('avg_curvature', 0)
                ])
            
            # Traffic stats
            if 'traffic' in scenario_features and 'traffic_stats' in scenario_features['traffic']:
                stats = scenario_features['traffic']['traffic_stats']
                feature_vector.extend([
                    stats.get('avg_vehicles', 0),
                    stats.get('max_vehicles', 0),
                    stats.get('traffic_variance', 0)
                ])
            
            if feature_vector:
                features[scenario_file] = np.array(feature_vector)
        
        return features
    
    def _extract_set_features(self):
        """Extract set-based features for Phase 2 metrics.""" 
        features = {}
        
        for scenario_file in self.scenario_files:
            if scenario_file not in self.feature_cache:
                continue
            
            scenario_features = self.feature_cache[scenario_file]
            
            # Convert action sequence to set
            if ('behavioral' in scenario_features and 
                'action_sequence' in scenario_features['behavioral']):
                action_set = set(scenario_features['behavioral']['action_sequence'])
                features[scenario_file] = action_set
        
        return features
    
    def _extract_action_sequences(self):
        """Extract action sequences for Phase 3 metrics."""
        sequences = {}
        
        for scenario_file in self.scenario_files:
            if scenario_file not in self.feature_cache:
                continue
            
            scenario_features = self.feature_cache[scenario_file]
            
            if ('behavioral' in scenario_features and 
                'action_sequence' in scenario_features['behavioral']):
                sequences[scenario_file] = scenario_features['behavioral']['action_sequence']
        
        return sequences
    
    def _calculate_pairwise_similarities(self, features, similarity_func):
        """Calculate pairwise similarities using given function."""
        n = len(self.scenario_files)
        similarity_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i + 1, n):
                file1, file2 = self.scenario_files[i], self.scenario_files[j]
                
                if file1 in features and file2 in features:
                    sim = similarity_func(features[file1], features[file2])
                    similarity_matrix[i, j] = sim
                    similarity_matrix[j, i] = sim
        
        # Set diagonal to 1.0
        np.fill_diagonal(similarity_matrix, 1.0)
        return similarity_matrix
    
    def _calculate_pairwise_set_similarities(self, set_features, similarity_func):
        """Calculate pairwise set similarities."""
        return self._calculate_pairwise_similarities(set_features, similarity_func)
    
    def _calculate_pairwise_sequence_similarities(self, sequences, similarity_func):
        """Calculate pairwise sequence similarities."""
        return self._calculate_pairwise_similarities(sequences, similarity_func)
    
    def _evaluate_similarity_matrix(self, similarity_matrix):
        """Evaluate similarity matrix against ground truth with multiple thresholds."""
        best_result = {'f1_score': 0, 'accuracy': 0, 'precision': 0, 'recall': 0, 'threshold': 0}
        
        # Test multiple thresholds
        thresholds = np.arange(0.1, 1.0, 0.1)
        
        for threshold in thresholds:
            # Convert to binary predictions
            binary_pred = (similarity_matrix >= threshold).astype(int)
            
            # Flatten matrices for comparison (exclude diagonal)
            gt_flat = []
            pred_flat = []
            
            n = len(self.scenario_files)
            for i in range(n):
                for j in range(i + 1, n):
                    gt_flat.append(self.gt_binary_matrix[i, j])
                    pred_flat.append(binary_pred[i, j])
            
            if len(set(pred_flat)) < 2:  # Skip if all predictions are the same
                continue
            
            # Calculate metrics
            f1 = f1_score(gt_flat, pred_flat)
            accuracy = accuracy_score(gt_flat, pred_flat)
            precision = precision_score(gt_flat, pred_flat)
            recall = recall_score(gt_flat, pred_flat)
            
            # Update best result
            if f1 > best_result['f1_score']:
                best_result = {
                    'f1_score': f1,
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'threshold': threshold
                }
        
        return best_result
    
    # Similarity metric implementations
    def _euclidean_similarity(self, vec1, vec2):
        """Euclidean distance converted to similarity."""
        distance = euclidean(vec1, vec2)
        return 1 / (1 + distance)
    
    def _manhattan_similarity(self, vec1, vec2):
        """Manhattan distance converted to similarity."""
        distance = cityblock(vec1, vec2)
        return 1 / (1 + distance)
    
    def _cosine_similarity(self, vec1, vec2):
        """Cosine similarity."""
        return 1 - cosine(vec1, vec2)
    
    def _minkowski_similarity(self, vec1, vec2, p=2):
        """Minkowski distance converted to similarity."""
        distance = minkowski(vec1, vec2, p=p)
        return 1 / (1 + distance)
    
    def _jaccard_similarity(self, set1, set2):
        """Jaccard similarity for sets."""
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        return intersection / union if union > 0 else 0
    
    def _dice_similarity(self, set1, set2):
        """Dice coefficient for sets."""
        intersection = len(set1 & set2)
        total = len(set1) + len(set2)
        return 2 * intersection / total if total > 0 else 0
    
    def _overlap_similarity(self, set1, set2):
        """Overlap coefficient for sets."""
        intersection = len(set1 & set2)
        min_size = min(len(set1), len(set2))
        return intersection / min_size if min_size > 0 else 0
    
    def _cosine_set_similarity(self, set1, set2):
        """Cosine similarity for sets."""
        intersection = len(set1 & set2)
        magnitude = math.sqrt(len(set1) * len(set2))
        return intersection / magnitude if magnitude > 0 else 0
    
    def _edit_distance_similarity(self, seq1, seq2):
        """Edit distance converted to similarity."""
        if not seq1 and not seq2:
            return 1.0
        if not seq1 or not seq2:
            return 0.0
        
        distance = self._edit_distance(seq1, seq2)
        max_len = max(len(seq1), len(seq2))
        return 1 - (distance / max_len) if max_len > 0 else 0
    
    def _lcs_similarity(self, seq1, seq2):
        """Longest common subsequence similarity."""
        if not seq1 and not seq2:
            return 1.0
        if not seq1 or not seq2:
            return 0.0
        
        lcs_length = self._lcs_length(seq1, seq2)
        max_len = max(len(seq1), len(seq2))
        return lcs_length / max_len if max_len > 0 else 0
    
    def _dtw_similarity(self, seq1, seq2):
        """Dynamic Time Warping similarity (simplified)."""
        if not seq1 and not seq2:
            return 1.0
        if not seq1 or not seq2:
            return 0.0
        
        # Simplified DTW implementation
        dtw_distance = self._simplified_dtw(seq1, seq2)
        max_len = max(len(seq1), len(seq2))
        return 1 - (dtw_distance / max_len) if max_len > 0 else 0
    
    def _ngram_jaccard_similarity(self, seq1, seq2):
        """N-gram Jaccard similarity (the Phase 3 winner)."""
        if not seq1 and not seq2:
            return 1.0
        if not seq1 or not seq2:
            return 0.0
        
        # Generate 2-grams
        ngrams1 = self._generate_ngrams(seq1, 2)
        ngrams2 = self._generate_ngrams(seq2, 2)
        
        if not ngrams1 and not ngrams2:
            return 1.0
        if not ngrams1 or not ngrams2:
            return 0.0
        
        set1, set2 = set(ngrams1), set(ngrams2)
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        return intersection / union if union > 0 else 0
    
    def _global_alignment_similarity(self, seq1, seq2):
        """Global sequence alignment similarity."""
        if not seq1 and not seq2:
            return 1.0
        if not seq1 or not seq2:
            return 0.0
        
        # Simplified alignment score
        alignment_score = self._alignment_score(seq1, seq2)
        max_possible = max(len(seq1), len(seq2))
        return alignment_score / max_possible if max_possible > 0 else 0
    
    # Helper methods (implementations of distance/similarity calculations)
    def _edit_distance(self, seq1, seq2):
        """Calculate edit distance."""
        m, n = len(seq1), len(seq2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i-1] == seq2[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
        
        return dp[m][n]
    
    def _lcs_length(self, seq1, seq2):
        """Calculate longest common subsequence length."""
        m, n = len(seq1), len(seq2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i-1] == seq2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        
        return dp[m][n]
    
    def _simplified_dtw(self, seq1, seq2):
        """Simplified DTW distance."""
        m, n = len(seq1), len(seq2)
        dtw = [[float('inf')] * (n + 1) for _ in range(m + 1)]
        dtw[0][0] = 0
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                cost = 0 if seq1[i-1] == seq2[j-1] else 1
                dtw[i][j] = cost + min(dtw[i-1][j], dtw[i][j-1], dtw[i-1][j-1])
        
        return dtw[m][n]
    
    def _generate_ngrams(self, sequence, n):
        """Generate n-grams from sequence."""
        if len(sequence) < n:
            return []
        return [tuple(sequence[i:i+n]) for i in range(len(sequence) - n + 1)]
    
    def _alignment_score(self, seq1, seq2):
        """Simple alignment score."""
        matches = sum(1 for a, b in zip(seq1, seq2) if a == b)
        return matches
    
    # Normalization methods
    def _min_max_normalize(self, vectors):
        """Min-max normalization."""
        vectors = np.array(vectors)
        min_vals = np.min(vectors, axis=0)
        max_vals = np.max(vectors, axis=0)
        range_vals = max_vals - min_vals
        range_vals[range_vals == 0] = 1  # Avoid division by zero
        return (vectors - min_vals) / range_vals
    
    def _z_score_normalize(self, vectors):
        """Z-score normalization."""
        vectors = np.array(vectors)
        mean_vals = np.mean(vectors, axis=0)
        std_vals = np.std(vectors, axis=0)
        std_vals[std_vals == 0] = 1  # Avoid division by zero
        return (vectors - mean_vals) / std_vals
    
    def _robust_normalize(self, vectors):
        """Robust normalization using median and IQR."""
        vectors = np.array(vectors)
        median_vals = np.median(vectors, axis=0)
        q75 = np.percentile(vectors, 75, axis=0)
        q25 = np.percentile(vectors, 25, axis=0)
        iqr = q75 - q25
        iqr[iqr == 0] = 1  # Avoid division by zero
        return (vectors - median_vals) / iqr
    
    def run_comprehensive_evaluation(self):
        """Run complete evaluation of all similarity metrics."""
        print("🔬 COMPREHENSIVE SIMILARITY METRIC EVALUATION")
        print("=" * 60)
        print(f"Ground Truth: Multi-Dimensional ({self.gt_data['metadata']['similarity_rate']:.1%} similarity rate)")
        print(f"Scenarios: {len(self.scenario_files)}")
        print(f"Total Pairs: {self.gt_data['metadata']['total_pairs']}")
        
        all_results = {}
        
        # Evaluate all phases
        try:
            phase1_results = self.evaluate_phase1_distance_metrics()
            all_results.update({f"Phase1_{k}": v for k, v in phase1_results.items()})
        except Exception as e:
            print(f"❌ Phase 1 evaluation failed: {e}")
        
        try:
            phase2_results = self.evaluate_phase2_set_metrics()
            all_results.update({f"Phase2_{k}": v for k, v in phase2_results.items()})
        except Exception as e:
            print(f"❌ Phase 2 evaluation failed: {e}")
        
        try:
            phase3_results = self.evaluate_phase3_sequence_metrics()
            all_results.update({f"Phase3_{k}": v for k, v in phase3_results.items()})
        except Exception as e:
            print(f"❌ Phase 3 evaluation failed: {e}")
        
        # Rank all results
        if all_results:
            self._report_final_results(all_results)
            return all_results
        else:
            print("❌ No evaluation results obtained")
            return {}
    
    def _report_final_results(self, all_results):
        """Generate final ranking report."""
        print("\n" + "="*70)
        print("🏆 FINAL RESULTS: BEST SIMILARITY METRICS VS MULTI-DIMENSIONAL GT")
        print("="*70)
        
        # Sort by F1 score
        sorted_results = sorted(
            all_results.items(), 
            key=lambda x: x[1]['f1_score'], 
            reverse=True
        )
        
        print(f"{'Rank':<4} {'F1':<6} {'Acc':<6} {'Prec':<6} {'Rec':<6} {'Thresh':<6} {'Method'}")
        print("-" * 70)
        
        for i, (method_name, result) in enumerate(sorted_results[:20], 1):
            print(f"{i:<4} "
                  f"{result['f1_score']:.3f}  "
                  f"{result['accuracy']:.3f}  "
                  f"{result['precision']:.3f}  "
                  f"{result['recall']:.3f}  "
                  f"{result['threshold']:.2f}   "
                  f"{method_name}")
        
        # Highlight best performers
        print(f"\n🥇 WINNER: {sorted_results[0][0]} (F1={sorted_results[0][1]['f1_score']:.3f})")
        
        if len(sorted_results) >= 2:
            print(f"🥈 Runner-up: {sorted_results[1][0]} (F1={sorted_results[1][1]['f1_score']:.3f})")
        
        if len(sorted_results) >= 3:
            print(f"🥉 Third place: {sorted_results[2][0]} (F1={sorted_results[2][1]['f1_score']:.3f})")
        
        # Compare with original Phase 3 winner
        ngram_result = None
        for method_name, result in sorted_results:
            if 'ngram_jaccard' in method_name.lower():
                ngram_result = result
                break
        
        if ngram_result:
            print(f"\n📊 COMPARISON:")
            print(f"   N-gram Jaccard vs Filename GT: F1=0.702")
            print(f"   N-gram Jaccard vs Multi-D GT:  F1={ngram_result['f1_score']:.3f}")
            print(f"   New best method vs Multi-D GT: F1={sorted_results[0][1]['f1_score']:.3f}")
        
        # Save results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = f"similarity_metrics_evaluation_{timestamp}.json"
        
        output_data = {
            'metadata': {
                'evaluation_date': datetime.now().isoformat(),
                'ground_truth_file': 'multi_dimensional_ground_truth.json',
                'num_scenarios': len(self.scenario_files),
                'ground_truth_similarity_rate': self.gt_data['metadata']['similarity_rate']
            },
            'results': all_results,
            'ranking': [(name, result) for name, result in sorted_results]
        }
        
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2, default=str)
        
        print(f"\n💾 Results saved to: {output_file}")

def main():
    """Run comprehensive similarity metric evaluation."""
    
    # Find the most recent multi-dimensional ground truth file
    gt_files = [f for f in os.listdir('.') if f.startswith('multi_dimensional_ground_truth_') and f.endswith('.json')]
    
    if not gt_files:
        print("❌ No multi-dimensional ground truth file found!")
        print("   Please run multi_dimensional_ground_truth.py first")
        return
    
    # Use the most recent ground truth file
    gt_file = sorted(gt_files)[-1]
    print(f"Using ground truth file: {gt_file}")
    
    # Run evaluation
    evaluator = ComprehensiveSimilarityEvaluator(gt_file)
    results = evaluator.run_comprehensive_evaluation()
    
    if results:
        print("\n✅ Evaluation completed successfully!")
        print("   Check the output JSON file for detailed results")
    else:
        print("\n❌ Evaluation failed")

if __name__ == "__main__":
    main()