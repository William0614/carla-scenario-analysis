#!/usr/bin/env python3
"""
Evaluation Module for CARLA Scenario Similarity Analysis

Comprehensive evaluation framework for comparing similarity metrics
against ground truth implementations.
"""

import json
import numpy as np
from datetime import datetime
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from scipy.stats import pearsonr

from .similarity_metrics import (
    DistanceBasedMetrics, 
    SequenceBasedMetrics, 
    SetBasedMetrics,
    NormalizationUtils
)


class SimilarityEvaluator:
    """
    Comprehensive evaluation framework for similarity metrics validation.
    
    Evaluates different categories of similarity metrics:
    - Distance-based metrics (Euclidean, Cosine, etc.)
    - Sequence-based metrics (LCS, Edit Distance, etc.)
    - Set-based metrics (Jaccard, Dice, etc.)
    """
    
    def __init__(self, similarity_threshold=0.6):
        """
        Initialize the evaluator.
        
        Args:
            similarity_threshold (float): Threshold for binary similarity classification
        """
        self.similarity_threshold = similarity_threshold
        self.distance_metrics = DistanceBasedMetrics()
        self.sequence_metrics = SequenceBasedMetrics()
        self.set_metrics = SetBasedMetrics()
        
    def evaluate_all_metrics(self, features_dict, ground_truth, normalized=True):
        """
        Evaluate all similarity metrics against ground truth.
        
        Args:
            features_dict (dict): Dictionary of extracted features per scenario
            ground_truth (dict): Ground truth similarity pairs
            normalized (bool): Whether to use normalized features
            
        Returns:
            dict: Comprehensive evaluation results
        """
        print("Starting comprehensive similarity metrics evaluation...")
        
        # Prepare features
        if normalized:
            features_dict = NormalizationUtils.z_score_normalize(features_dict)
        
        scenarios = list(features_dict.keys())
        
        # Initialize results
        results = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'total_scenarios': len(scenarios),
                'total_comparisons': len(scenarios) * (len(scenarios) - 1) // 2,
                'similarity_threshold': self.similarity_threshold,
                'normalized_features': normalized
            },
            'distance_based': {},
            'sequence_based': {},
            'set_based': {},
            'summary': {}
        }
        
        # Evaluate distance-based metrics
        print("Evaluating distance-based metrics...")
        results['distance_based'] = self._evaluate_distance_metrics(
            features_dict, ground_truth, scenarios
        )
        
        # Evaluate sequence-based metrics
        print("Evaluating sequence-based metrics...")
        results['sequence_based'] = self._evaluate_sequence_metrics(
            features_dict, ground_truth, scenarios
        )
        
        # Evaluate set-based metrics (if categorical features available)
        print("Evaluating set-based metrics...")
        results['set_based'] = self._evaluate_set_metrics(
            features_dict, ground_truth, scenarios
        )
        
        # Generate summary
        results['summary'] = self._generate_summary(results)
        
        print(f"Evaluation complete. Best overall metric: {results['summary']['best_metric']}")
        
        return results
    
    def _evaluate_distance_metrics(self, features_dict, ground_truth, scenarios):
        """Evaluate all distance-based similarity metrics."""
        metrics_to_test = {
            'euclidean': lambda f1, f2: 1 / (1 + self.distance_metrics.euclidean_distance(f1, f2)),
            'manhattan': lambda f1, f2: 1 / (1 + self.distance_metrics.manhattan_distance(f1, f2)),
            'cosine': self.distance_metrics.cosine_similarity,
            'chebyshev': lambda f1, f2: 1 / (1 + self.distance_metrics.chebyshev_distance(f1, f2)),
            'minkowski_p3': lambda f1, f2: 1 / (1 + self.distance_metrics.minkowski_distance(f1, f2, p=3)),
            'pearson': self.distance_metrics.pearson_correlation,
            'spearman': self.distance_metrics.spearman_correlation
        }
        
        results = {}
        
        for metric_name, metric_func in metrics_to_test.items():
            print(f"  Testing {metric_name}...")
            
            # Calculate similarities for all pairs
            predicted_similarities = []
            true_labels = []
            
            for i, scenario1 in enumerate(scenarios):
                for j, scenario2 in enumerate(scenarios[i+1:], i+1):
                    # Get features
                    features1 = features_dict[scenario1]['combined_vector']
                    features2 = features_dict[scenario2]['combined_vector']
                    
                    # Calculate similarity
                    similarity = metric_func(features1, features2)
                    predicted_similarities.append(similarity)
                    
                    # Get ground truth
                    gt_key = f"{scenario1}_{scenario2}"
                    true_label = 1 if gt_key in ground_truth else 0
                    true_labels.append(true_label)
            
            # Evaluate performance
            predicted_labels = [1 if sim >= self.similarity_threshold else 0 
                              for sim in predicted_similarities]
            
            results[metric_name] = self._calculate_performance_metrics(
                true_labels, predicted_labels, predicted_similarities
            )
        
        return results
    
    def _evaluate_sequence_metrics(self, features_dict, ground_truth, scenarios):
        """Evaluate sequence-based similarity metrics."""
        metrics_to_test = {
            'lcs': self.sequence_metrics.longest_common_subsequence,
            'edit_distance': self.sequence_metrics.edit_distance_similarity,
            'sequence_matcher': self.sequence_metrics.sequence_matcher_similarity,
            'ngram_jaccard': lambda s1, s2: self.sequence_metrics.ngram_jaccard_similarity(s1, s2, n=2),
            'dtw': self.sequence_metrics.dtw_similarity
        }
        
        results = {}
        
        # Extract behavioral sequences for all scenarios
        sequences = {}
        for scenario, features in features_dict.items():
            sequences[scenario] = self._extract_sequence_from_features(features)
        
        for metric_name, metric_func in metrics_to_test.items():
            print(f"  Testing {metric_name}...")
            
            predicted_similarities = []
            true_labels = []
            
            for i, scenario1 in enumerate(scenarios):
                for j, scenario2 in enumerate(scenarios[i+1:], i+1):
                    # Get sequences
                    seq1 = sequences.get(scenario1, [])
                    seq2 = sequences.get(scenario2, [])
                    
                    # Calculate similarity
                    similarity = metric_func(seq1, seq2) if seq1 and seq2 else 0.0
                    predicted_similarities.append(similarity)
                    
                    # Get ground truth
                    gt_key = f"{scenario1}_{scenario2}"
                    true_label = 1 if gt_key in ground_truth else 0
                    true_labels.append(true_label)
            
            # Evaluate performance
            predicted_labels = [1 if sim >= self.similarity_threshold else 0 
                              for sim in predicted_similarities]
            
            results[metric_name] = self._calculate_performance_metrics(
                true_labels, predicted_labels, predicted_similarities
            )
        
        return results
    
    def _evaluate_set_metrics(self, features_dict, ground_truth, scenarios):
        """Evaluate set-based similarity metrics."""
        metrics_to_test = {
            'jaccard': self.set_metrics.jaccard_similarity,
            'dice': self.set_metrics.dice_similarity,
            'overlap': self.set_metrics.overlap_similarity,
            'cosine_set': self.set_metrics.cosine_set_similarity
        }
        
        results = {}
        
        # Convert features to categorical sets
        categorical_sets = {}
        for scenario, features in features_dict.items():
            categorical_sets[scenario] = self._convert_features_to_sets(features)
        
        for metric_name, metric_func in metrics_to_test.items():
            print(f"  Testing {metric_name}...")
            
            predicted_similarities = []
            true_labels = []
            
            for i, scenario1 in enumerate(scenarios):
                for j, scenario2 in enumerate(scenarios[i+1:], i+1):
                    # Get sets
                    set1 = categorical_sets.get(scenario1, set())
                    set2 = categorical_sets.get(scenario2, set())
                    
                    # Calculate similarity
                    similarity = metric_func(set1, set2)
                    predicted_similarities.append(similarity)
                    
                    # Get ground truth
                    gt_key = f"{scenario1}_{scenario2}"
                    true_label = 1 if gt_key in ground_truth else 0
                    true_labels.append(true_label)
            
            # Evaluate performance
            predicted_labels = [1 if sim >= self.similarity_threshold else 0 
                              for sim in predicted_similarities]
            
            results[metric_name] = self._calculate_performance_metrics(
                true_labels, predicted_labels, predicted_similarities
            )
        
        return results
    
    def _calculate_performance_metrics(self, true_labels, predicted_labels, predicted_scores):
        """Calculate comprehensive performance metrics."""
        try:
            # Classification metrics
            accuracy = accuracy_score(true_labels, predicted_labels)
            precision = precision_score(true_labels, predicted_labels, zero_division=0)
            recall = recall_score(true_labels, predicted_labels, zero_division=0)
            f1 = f1_score(true_labels, predicted_labels, zero_division=0)
            
            # Correlation with ground truth
            correlation = 0.0
            if len(set(predicted_scores)) > 1 and len(set(true_labels)) > 1:
                try:
                    correlation, _ = pearsonr(predicted_scores, true_labels)
                    if np.isnan(correlation):
                        correlation = 0.0
                except:
                    correlation = 0.0
            
            return {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'correlation': correlation,
                'predicted_positive': sum(predicted_labels),
                'true_positive': sum(true_labels)
            }
            
        except Exception as e:
            print(f"Error calculating performance metrics: {e}")
            return {
                'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0,
                'f1_score': 0.0, 'correlation': 0.0,
                'predicted_positive': 0, 'true_positive': 0
            }
    
    def _extract_sequence_from_features(self, features):
        """Extract behavioral sequence from feature vector."""
        if not features or 'behavioral_features' not in features:
            return []
        
        behavioral_features = features['behavioral_features']
        
        # Reconstruct sequence from behavioral feature counts
        sequence = []
        
        if len(behavioral_features) >= 10:
            stops = int(behavioral_features[0])
            accelerations = int(behavioral_features[1])
            decelerations = int(behavioral_features[2])
            turns = int(behavioral_features[3])
            cruise = int(behavioral_features[4])
            
            # Create representative sequence
            sequence.extend([0] * min(stops, 5))
            sequence.extend([1] * min(accelerations, 5))
            sequence.extend([2] * min(decelerations, 5))
            sequence.extend([3] * min(turns, 5))
            sequence.extend([5] * min(cruise, 5))
        
        return sequence
    
    def _convert_features_to_sets(self, features):
        """Convert features to categorical sets for set-based metrics."""
        feature_set = set()
        
        if not features:
            return feature_set
        
        # Convert temporal features
        temporal = features.get('temporal_features', [])
        if len(temporal) >= 6:
            if temporal[0] > 15.0:  # Long duration
                feature_set.add('long_scenario')
            if temporal[4] > 2.0:  # High speed variability
                feature_set.add('variable_speed')
        
        # Convert behavioral features
        behavioral = features.get('behavioral_features', [])
        if len(behavioral) >= 10:
            if behavioral[0] > 2:  # Frequent stops
                feature_set.add('frequent_stopping')
            if behavioral[3] > 3:  # Many turns
                feature_set.add('turning_scenario')
        
        # Convert spatial features
        spatial = features.get('spatial_features', [])
        if len(spatial) >= 8:
            if spatial[2] > 1.5:  # Complex path
                feature_set.add('complex_path')
        
        # Convert speed features
        speed = features.get('speed_features', [])
        if len(speed) >= 10:
            if speed[0] > 8.0:  # High mean speed
                feature_set.add('high_speed_scenario')
        
        # Convert traffic features
        traffic = features.get('traffic_features', [])
        if len(traffic) >= 3:
            if traffic[2] > 0:  # Traffic present
                feature_set.add('traffic_present')
        
        return feature_set
    
    def _generate_summary(self, results):
        """Generate summary of best performing metrics by category."""
        summary = {
            'best_by_category': {},
            'best_metric': None,
            'best_f1_score': 0.0
        }
        
        # Find best in each category
        for category in ['distance_based', 'sequence_based', 'set_based']:
            if category in results and results[category]:
                best_metric = max(
                    results[category].items(),
                    key=lambda x: x[1]['f1_score']
                )
                summary['best_by_category'][category] = {
                    'metric': best_metric[0],
                    'f1_score': best_metric[1]['f1_score'],
                    'accuracy': best_metric[1]['accuracy']
                }
        
        # Find overall best
        all_metrics = []
        for category, metrics in results.items():
            if category not in ['metadata', 'summary']:
                for metric_name, performance in metrics.items():
                    all_metrics.append((f"{category}_{metric_name}", performance['f1_score']))
        
        if all_metrics:
            best_overall = max(all_metrics, key=lambda x: x[1])
            summary['best_metric'] = best_overall[0]
            summary['best_f1_score'] = best_overall[1]
        
        return summary
    
    def save_results(self, results, filepath):
        """Save evaluation results to JSON file."""
        try:
            with open(filepath, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            print(f"Evaluation results saved to {filepath}")
        except Exception as e:
            print(f"Error saving results: {e}")
    
    def print_summary(self, results):
        """Print a formatted summary of evaluation results."""
        print("\n" + "="*80)
        print("SIMILARITY METRICS EVALUATION SUMMARY")
        print("="*80)
        
        if 'best_by_category' in results.get('summary', {}):
            print("\nBest Performers by Category:")
            for category, best in results['summary']['best_by_category'].items():
                print(f"  {category.replace('_', ' ').title()}: {best['metric']} "
                      f"(F1={best['f1_score']:.3f}, Acc={best['accuracy']:.3f})")
        
        if 'best_metric' in results.get('summary', {}):
            print(f"\nOverall Best Metric: {results['summary']['best_metric']} "
                  f"(F1={results['summary']['best_f1_score']:.3f})")
        
        print("\n" + "="*80)