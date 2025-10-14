#!/usr/bin/env python3
"""
Ground Truth Module for CARLA Scenario Analysis

Provides both basic filename-based ground truth and multi-dimensional
behavioral ground truth implementations.
"""

import os
import re
import math
import numpy as np
from collections import defaultdict, Counter
from .similarity_metrics import SequenceBasedMetrics


class BasicGroundTruth:
    """
    Basic ground truth based on filename patterns and geographical similarity.
    Used as baseline for comparison with multi-dimensional approaches.
    """
    
    def __init__(self):
        self.similarity_threshold = 0.7
    
    def generate_ground_truth(self, scenario_files):
        """
        Generate basic ground truth similarity matrix based on filenames.
        
        Args:
            scenario_files (list): List of scenario filenames
            
        Returns:
            dict: Ground truth similarity pairs
        """
        ground_truth = {}
        
        for i, file1 in enumerate(scenario_files):
            for j, file2 in enumerate(scenario_files[i+1:], i+1):
                similarity = self._calculate_filename_similarity(file1, file2)
                
                if similarity >= self.similarity_threshold:
                    key = f"{file1}_{file2}"
                    ground_truth[key] = {
                        'similarity': similarity,
                        'similar': True,
                        'method': 'filename_pattern'
                    }
        
        return ground_truth
    
    def _calculate_filename_similarity(self, file1, file2):
        """Calculate similarity based on filename patterns."""
        # Extract components from filenames
        components1 = self._extract_filename_components(file1)
        components2 = self._extract_filename_components(file2)
        
        # Calculate component-wise similarities
        similarities = []
        
        # Country/location similarity
        if components1.get('country') == components2.get('country'):
            similarities.append(0.4)  # High weight for same country
        
        # City similarity
        if components1.get('city') == components2.get('city'):
            similarities.append(0.4)  # High weight for same city
        
        # Scenario type similarity
        if components1.get('scenario_type') == components2.get('scenario_type'):
            similarities.append(0.2)  # Moderate weight for scenario type
        
        return sum(similarities) if similarities else 0.0
    
    def _extract_filename_components(self, filename):
        """Extract structured components from CARLA scenario filename."""
        # Example: "DEU_Flensburg-28_1_I-1-1.log"
        # Pattern: COUNTRY_CITY-SCENARIO_VARIANT_TYPE-ITERATION-RUN.log
        
        components = {}
        
        # Remove .log extension
        name = filename.replace('.log', '')
        
        # Split by underscores and dashes
        parts = re.split('[_-]', name)
        
        if len(parts) >= 2:
            components['country'] = parts[0]  # DEU, FRA, etc.
            components['city'] = parts[1].split('-')[0] if '-' in parts[1] else parts[1]
        
        if len(parts) >= 3:
            # Extract numeric scenario identifier
            scenario_match = re.search(r'(\d+)', parts[1])
            if scenario_match:
                components['scenario_type'] = scenario_match.group(1)
        
        return components


class MultiDimensionalGroundTruth:
    """
    Multi-dimensional ground truth based on 4-dimensional behavioral analysis:
    1. Behavioral similarity (LCS-based)
    2. Spatial similarity  
    3. Traffic similarity
    4. Contextual similarity
    
    This is the validated ground truth methodology from the research.
    """
    
    def __init__(self, similarity_threshold=0.6):
        """
        Initialize multi-dimensional ground truth generator.
        
        Args:
            similarity_threshold (float): Minimum similarity for ground truth pairs
        """
        self.similarity_threshold = similarity_threshold
        self.lcs_metrics = SequenceBasedMetrics()
        
        # Similarity calculation parameters (optimized from research)
        self.weights = {
            'behavioral': 0.4,   # Primary weight for behavioral patterns
            'spatial': 0.25,     # Secondary weight for spatial patterns  
            'traffic': 0.2,      # Moderate weight for traffic context
            'contextual': 0.15   # Lower weight for contextual factors
        }
        
        # Thresholds for individual similarity dimensions
        self.thresholds = {
            'spatial_distance': 500.0,      # meters
            'duration_diff': 10.0,          # seconds
            'speed_diff': 3.0,              # m/s
            'traffic_diff': 3               # vehicle count
        }
    
    def generate_ground_truth(self, features_dict):
        """
        Generate multi-dimensional ground truth similarity pairs.
        
        Args:
            features_dict (dict): Dictionary of extracted features per scenario
            
        Returns:
            dict: Ground truth similarity pairs with detailed analysis
        """
        scenarios = list(features_dict.keys())
        ground_truth = {}
        
        total_comparisons = 0
        similar_pairs = 0
        
        for i, scenario1 in enumerate(scenarios):
            for j, scenario2 in enumerate(scenarios[i+1:], i+1):
                total_comparisons += 1
                
                # Calculate multi-dimensional similarity
                similarity_result = self._calculate_multidimensional_similarity(
                    scenario1, scenario2, features_dict
                )
                
                if similarity_result['overall_similarity'] >= self.similarity_threshold:
                    key = f"{scenario1}_{scenario2}"
                    ground_truth[key] = {
                        'overall_similarity': similarity_result['overall_similarity'],
                        'similar': True,
                        'dimensions': similarity_result['dimensions'],
                        'method': 'multidimensional'
                    }
                    similar_pairs += 1
        
        # Add summary statistics
        similarity_rate = similar_pairs / total_comparisons if total_comparisons > 0 else 0
        ground_truth['_metadata'] = {
            'total_comparisons': total_comparisons,
            'similar_pairs': similar_pairs,
            'similarity_rate': similarity_rate,
            'threshold': self.similarity_threshold
        }
        
        return ground_truth
    
    def _calculate_multidimensional_similarity(self, scenario1, scenario2, features_dict):
        """Calculate 4-dimensional similarity between two scenarios."""
        features1 = features_dict.get(scenario1, {})
        features2 = features_dict.get(scenario2, {})
        
        # Initialize dimension similarities
        dimensions = {}
        
        # 1. Behavioral Similarity (LCS-based)
        dimensions['behavioral'] = self._calculate_behavioral_similarity(features1, features2)
        
        # 2. Spatial Similarity
        dimensions['spatial'] = self._calculate_spatial_similarity(features1, features2)
        
        # 3. Traffic Similarity
        dimensions['traffic'] = self._calculate_traffic_similarity(features1, features2)
        
        # 4. Contextual Similarity (duration, speed patterns)
        dimensions['contextual'] = self._calculate_contextual_similarity(features1, features2)
        
        # Calculate weighted overall similarity
        overall_similarity = sum(
            self.weights[dim] * score for dim, score in dimensions.items()
        )
        
        return {
            'overall_similarity': overall_similarity,
            'dimensions': dimensions
        }
    
    def _calculate_behavioral_similarity(self, features1, features2):
        """Calculate behavioral similarity using LCS on action sequences."""
        # Extract behavioral sequences from features
        seq1 = self._extract_behavioral_sequence(features1)
        seq2 = self._extract_behavioral_sequence(features2)
        
        if not seq1 or not seq2:
            return 0.0
        
        # Use optimized LCS similarity
        return self.lcs_metrics.longest_common_subsequence(seq1, seq2)
    
    def _extract_behavioral_sequence(self, features):
        """Extract behavioral action sequence from feature vector."""
        if not features or 'behavioral_features' not in features:
            return []
        
        behavioral_features = features['behavioral_features']
        
        # Reconstruct sequence from behavioral feature counts
        # This is a simplified reconstruction - in practice, you'd store the original sequence
        sequence = []
        
        # Add actions based on counts (simplified)
        if len(behavioral_features) >= 10:
            stops = int(behavioral_features[0])
            accelerations = int(behavioral_features[1])
            decelerations = int(behavioral_features[2])
            turns = int(behavioral_features[3])
            cruise = int(behavioral_features[4])
            
            # Create a representative sequence
            sequence.extend([0] * min(stops, 5))        # Stops
            sequence.extend([1] * min(accelerations, 5)) # Accelerations
            sequence.extend([2] * min(decelerations, 5)) # Decelerations
            sequence.extend([3] * min(turns, 5))         # Turns
            sequence.extend([5] * min(cruise, 5))        # Cruise
        
        return sequence
    
    def _calculate_spatial_similarity(self, features1, features2):
        """Calculate spatial similarity based on path characteristics."""
        if not features1 or not features2:
            return 0.0
        
        spatial1 = features1.get('spatial_features', [])
        spatial2 = features2.get('spatial_features', [])
        
        if len(spatial1) < 8 or len(spatial2) < 8:
            return 0.0
        
        # Compare key spatial metrics
        path_length_sim = 1 - abs(spatial1[0] - spatial2[0]) / max(spatial1[0], spatial2[0], 1)
        displacement_sim = 1 - abs(spatial1[1] - spatial2[1]) / max(spatial1[1], spatial2[1], 1)
        tortuosity_sim = 1 - abs(spatial1[2] - spatial2[2]) / max(spatial1[2], spatial2[2], 1)
        
        # Average spatial similarity
        spatial_similarity = np.mean([path_length_sim, displacement_sim, tortuosity_sim])
        
        return max(0.0, min(1.0, spatial_similarity))
    
    def _calculate_traffic_similarity(self, features1, features2):
        """Calculate traffic context similarity."""
        if not features1 or not features2:
            return 0.0
        
        traffic1 = features1.get('traffic_features', [])
        traffic2 = features2.get('traffic_features', [])
        
        if len(traffic1) < 3 or len(traffic2) < 3:
            return 0.0
        
        # Compare traffic vehicle counts
        count1, count2 = traffic1[0], traffic2[0]
        traffic_diff = abs(count1 - count2)
        
        if traffic_diff <= self.thresholds['traffic_diff']:
            return 1.0 - (traffic_diff / self.thresholds['traffic_diff'])
        else:
            return 0.0
    
    def _calculate_contextual_similarity(self, features1, features2):
        """Calculate contextual similarity (duration, speed patterns)."""
        if not features1 or not features2:
            return 0.0
        
        temporal1 = features1.get('temporal_features', [])
        temporal2 = features2.get('temporal_features', [])
        speed1 = features1.get('speed_features', [])
        speed2 = features2.get('speed_features', [])
        
        similarities = []
        
        # Duration similarity
        if len(temporal1) >= 1 and len(temporal2) >= 1:
            duration_diff = abs(temporal1[0] - temporal2[0])
            if duration_diff <= self.thresholds['duration_diff']:
                duration_sim = 1.0 - (duration_diff / self.thresholds['duration_diff'])
                similarities.append(duration_sim)
        
        # Speed similarity
        if len(speed1) >= 1 and len(speed2) >= 1:
            speed_diff = abs(speed1[0] - speed2[0])  # Mean speed
            if speed_diff <= self.thresholds['speed_diff']:
                speed_sim = 1.0 - (speed_diff / self.thresholds['speed_diff'])
                similarities.append(speed_sim)
        
        return np.mean(similarities) if similarities else 0.0


# Utility functions
def load_ground_truth_from_file(filepath):
    """Load pre-computed ground truth from JSON file."""
    import json
    
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading ground truth from {filepath}: {e}")
        return {}

def save_ground_truth_to_file(ground_truth, filepath):
    """Save ground truth to JSON file."""
    import json
    
    try:
        with open(filepath, 'w') as f:
            json.dump(ground_truth, f, indent=2, default=str)
        print(f"Ground truth saved to {filepath}")
    except Exception as e:
        print(f"Error saving ground truth to {filepath}: {e}")