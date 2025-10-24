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
    Multi-dimensional ground truth based on 5-dimensional behavioral analysis:
    1. Temporal similarity - time-based patterns and event rates
    2. Motion similarity - speed, acceleration, and dynamics  
    3. Behavioral similarity - driving patterns and maneuvers (LCS-based)
    4. Spatial similarity - geometric and trajectory characteristics
    5. Context similarity - traffic and environmental factors
    
    Updated to align with refined 32-dimensional feature architecture.
    """
    
    def __init__(self, similarity_threshold=0.6):
        """
        Initialize multi-dimensional ground truth generator.
        
        Args:
            similarity_threshold (float): Minimum similarity for ground truth pairs
        """
        self.similarity_threshold = similarity_threshold
        self.lcs_metrics = SequenceBasedMetrics()
        
        # Similarity calculation parameters (updated for 32D architecture)
        self.weights = {
            'temporal': 0.15,    # Time-based patterns and event rates
            'motion': 0.25,      # Speed, acceleration, dynamics (high impact)
            'behavioral': 0.30,  # Primary weight for driving patterns
            'spatial': 0.20,     # Geometric and trajectory characteristics  
            'context': 0.10      # Environmental and traffic factors
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
        
        # 1. Temporal Similarity (time-based patterns)
        dimensions['temporal'] = self._calculate_temporal_similarity(features1, features2)
        
        # 2. Motion Similarity (speed, acceleration, dynamics)
        dimensions['motion'] = self._calculate_motion_similarity(features1, features2)
        
        # 3. Behavioral Similarity (LCS-based driving patterns)
        dimensions['behavioral'] = self._calculate_behavioral_similarity(features1, features2)
        
        # 4. Spatial Similarity (geometric characteristics)
        dimensions['spatial'] = self._calculate_spatial_similarity(features1, features2)
        
        # 5. Context Similarity (traffic and environment)
        dimensions['context'] = self._calculate_context_similarity(features1, features2)
        
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
        
        # Add actions based on counts (updated for 8D behavioral features)
        if len(behavioral_features) >= 8:
            stops = int(behavioral_features[0])          # Stop events
            accelerations = int(behavioral_features[1])   # Acceleration events  
            decelerations = int(behavioral_features[2])   # Deceleration events
            turns = int(behavioral_features[3])          # Turn maneuvers
            cruise = int(behavioral_features[4])         # Cruise periods
            transitions = int(behavioral_features[5])    # Behavior transitions
            
            # Create a representative sequence based on event counts
            sequence.extend([0] * min(stops, 5))        # Stops
            sequence.extend([1] * min(accelerations, 5)) # Accelerations
            sequence.extend([2] * min(decelerations, 5)) # Decelerations
            sequence.extend([3] * min(turns, 5))         # Turns
            sequence.extend([5] * min(cruise, 5))        # Cruise
            sequence.extend([6] * min(transitions, 3))   # Transitions (limited)
        
        return sequence
    
    def _calculate_temporal_similarity(self, features1, features2):
        """Calculate temporal similarity based on time-based patterns."""
        if not features1 or not features2:
            return 0.0
        
        temporal1 = features1.get('temporal_features', [])
        temporal2 = features2.get('temporal_features', [])
        
        if len(temporal1) < 4 or len(temporal2) < 4:
            return 0.0
        
        similarities = []
        
        # Duration similarity (feature 0)
        duration_diff = abs(temporal1[0] - temporal2[0])
        if duration_diff <= self.thresholds['duration_diff']:
            duration_sim = 1.0 - (duration_diff / self.thresholds['duration_diff'])
            similarities.append(duration_sim)
        
        # Frame count ratio similarity (feature 1) 
        frame_ratio1 = temporal1[1] / max(temporal1[0], 1)  # frames per second
        frame_ratio2 = temporal2[1] / max(temporal2[0], 1)
        frame_sim = 1 - abs(frame_ratio1 - frame_ratio2) / max(frame_ratio1, frame_ratio2, 1)
        similarities.append(max(0.0, frame_sim))
        
        # Event frequency similarity (feature 2)
        event_freq_sim = 1 - abs(temporal1[2] - temporal2[2]) / max(temporal1[2], temporal2[2], 1)
        similarities.append(max(0.0, event_freq_sim))
        
        # Temporal density similarity (feature 3)
        density_sim = 1 - abs(temporal1[3] - temporal2[3]) / max(temporal1[3], temporal2[3], 1)
        similarities.append(max(0.0, density_sim))
        
        return np.mean(similarities) if similarities else 0.0
    
    def _calculate_motion_similarity(self, features1, features2):
        """Calculate motion similarity based on speed and acceleration patterns."""
        if not features1 or not features2:
            return 0.0
        
        motion1 = features1.get('motion_features', [])
        motion2 = features2.get('motion_features', [])
        
        if len(motion1) < 8 or len(motion2) < 8:
            return 0.0
        
        similarities = []
        
        # Mean speed similarity (feature 0)
        speed_diff = abs(motion1[0] - motion2[0])
        if speed_diff <= self.thresholds['speed_diff']:
            speed_sim = 1.0 - (speed_diff / self.thresholds['speed_diff'])
            similarities.append(speed_sim)
        
        # Speed variability similarity (feature 1)
        speed_std_sim = 1 - abs(motion1[1] - motion2[1]) / max(motion1[1], motion2[1], 1)
        similarities.append(max(0.0, speed_std_sim))
        
        # Speed range similarity (features 2-4: min, max, range)
        range_sim = 1 - abs(motion1[4] - motion2[4]) / max(motion1[4], motion2[4], 1)
        similarities.append(max(0.0, range_sim))
        
        # Acceleration pattern similarity (features 5-6)
        acc_mean_sim = 1 - abs(motion1[5] - motion2[5]) / max(abs(motion1[5]), abs(motion2[5]), 1)
        similarities.append(max(0.0, acc_mean_sim))
        
        # Dynamic events similarity (feature 7)
        dynamic_sim = 1 - abs(motion1[7] - motion2[7]) / max(motion1[7], motion2[7], 1)
        similarities.append(max(0.0, dynamic_sim))
        
        return np.mean(similarities) if similarities else 0.0
    
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
    
    def _calculate_context_similarity(self, features1, features2):
        """Calculate context similarity (traffic and environmental factors)."""
        if not features1 or not features2:
            return 0.0
        
        context1 = features1.get('context_features', [])
        context2 = features2.get('context_features', [])
        
        if len(context1) < 4 or len(context2) < 4:
            return 0.0
        
        similarities = []
        
        # Traffic count similarity (feature 0)
        count1, count2 = context1[0], context2[0]
        traffic_diff = abs(count1 - count2)
        if traffic_diff <= self.thresholds['traffic_diff']:
            traffic_sim = 1.0 - (traffic_diff / self.thresholds['traffic_diff'])
            similarities.append(traffic_sim)
        
        # Traffic density similarity (feature 1)
        density_sim = 1 - abs(context1[1] - context2[1]) / max(context1[1], context2[1], 1)
        similarities.append(max(0.0, density_sim))
        
        # Traffic presence similarity (feature 2) - binary match
        presence_sim = 1.0 if context1[2] == context2[2] else 0.0
        similarities.append(presence_sim)
        
        # Scenario complexity similarity (feature 3)
        complexity_sim = 1 - abs(context1[3] - context2[3]) / max(context1[3], context2[3], 1)
        similarities.append(max(0.0, complexity_sim))
        
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