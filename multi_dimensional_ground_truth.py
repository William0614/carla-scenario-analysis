#!/usr/bin/env python3
"""
Multi-Dimensional Ground Truth Generator for CARLA Scenario Similarity

This module implements a comprehensive ground truth generation approach that considers:
1. Behavioral similarity (40% weight) - action sequences using LCS (F1=0.671), speed profiles, control patterns
2. Spatial similarity (30% weight) - path geometry, bounding boxes, curvature  
3. Traffic similarity (20% weight) - vehicle counts, density patterns
4. Contextual similarity (10% weight) - geographic region, scenario family

Key Discovery: Longest Common Subsequence (LCS) performs best vs multi-dimensional GT (F1=0.671),
while N-gram Jaccard was best vs filename GT (F1=0.702) but drops to F1=0.556 vs multi-dimensional GT.
This demonstrates that multi-dimensional ground truth requires different similarity metrics.

Author: Research Team
Date: October 6, 2025
Version: 1.2 - Updated with LCS based on comprehensive metric evaluation
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
from scipy.spatial.distance import euclidean
from scipy.stats import pearsonr
from scipy.spatial import distance
import math

class MultiDimensionalGroundTruth:
    """
    Generate multi-dimensional ground truth for CARLA scenario similarity analysis.
    
    Combines behavioral, spatial, traffic, and contextual similarity measures
    to create a more robust ground truth than filename-based approaches.
    """
    
    def __init__(self, log_directory="/home/ads/CARLA_0.9.15/log_files"):
        self.log_directory = log_directory
        self.extracted_features = {}
        
        # Initialize CARLA client
        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(10.0)
        
        # Weights for different similarity dimensions (must sum to 1.0)
        self.weights = {
            'behavioral': 0.40,    # Most important for AV testing
            'spatial': 0.30,       # Critical for navigation challenges
            'traffic': 0.20,       # Important for interaction complexity  
            'contextual': 0.10     # Useful but secondary information
        }
        
        # Thresholds for similarity classification
        self.similarity_threshold = 0.70  # Even more conservative after tightening calculations
        
    def extract_scenario_features(self, log_file):
        """
        Extract comprehensive features from a CARLA log file.
        
        Returns dictionary with behavioral, spatial, traffic, and contextual features.
        """
        features = {
            'behavioral': {},
            'spatial': {},
            'traffic': {},
            'contextual': {},
            'filename': log_file
        }
        
        try:
            # Load recorder data using the same approach as Phase 3
            log_path = os.path.join(self.log_directory, log_file)
            if not os.path.exists(log_path):
                print(f"Warning: Log file not found: {log_path}")
                return None
            
            # Get recorder info as string
            recorder_info = self.client.show_recorder_file_info(log_path, True)
            
            # Initialize MetricsLog with recorder info string
            log = MetricsLog(recorder_info)
            
            # Get ego vehicle ID - try multiple methods
            ego_id = None
            try:
                # First try the standard method (looks for "hero" role)
                ego_id = log.get_ego_vehicle_id()
            except (IndexError, Exception):
                pass
            
            if ego_id is None:
                # Try alternative method - look for "ego_vehicle" role
                try:
                    ego_ids = log.get_actor_ids_with_role_name("ego_vehicle")
                    if ego_ids:
                        ego_id = ego_ids[0]
                except:
                    pass
            
            if ego_id is None:
                # Final fallback - use first vehicle
                vehicle_ids = log.get_actor_ids_with_type_id("vehicle.*")
                if vehicle_ids:
                    ego_id = vehicle_ids[0]
            
            if ego_id is None:
                return None
            
            # Get frame range for ego vehicle (same as working scripts)
            start_frame, end_frame = log.get_actor_alive_frames(ego_id)
            frame_count = end_frame - start_frame
            if frame_count <= 10:  # Skip very short scenarios
                return None
            
            # Extract behavioral features
            features['behavioral'] = self._extract_behavioral_features(log, ego_id, start_frame, end_frame)
            
            # Extract spatial features  
            features['spatial'] = self._extract_spatial_features(log, ego_id, start_frame, end_frame)
            
            # Extract traffic features
            features['traffic'] = self._extract_traffic_features(log, start_frame, end_frame)
            
            # Extract contextual features
            features['contextual'] = self._extract_contextual_features(log_file)
            
            return features
            
        except Exception as e:
            print(f"Error processing {log_file}: {e}")
            return None
    
    def _parse_basic_recorder_info(self, recorder_str):
        """Parse basic recorder info to find ego vehicle and frame range."""
        lines = recorder_str.split('\n')
        ego_id = None
        start_frame = None
        end_frame = None
        
        for line in lines:
            if 'hero' in line.lower() or 'ego' in line.lower():
                # Try to extract vehicle ID
                parts = line.split()
                for part in parts:
                    if part.isdigit():
                        ego_id = int(part)
                        break
            elif 'Frames:' in line:
                # Try to extract frame range
                try:
                    frame_part = line.split('Frames:')[1].strip()
                    if '-' in frame_part:
                        start_str, end_str = frame_part.split('-')
                        start_frame = int(start_str.strip())
                        end_frame = int(end_str.strip())
                except:
                    pass
        
        # Default values if not found
        if ego_id is None:
            ego_id = 1  # Default assumption
        if start_frame is None:
            start_frame = 0
        if end_frame is None:
            end_frame = 1000  # Default assumption
            
        return ego_id, start_frame, min(end_frame, start_frame + 500)  # Limit to 500 frames for efficiency
    
    def _extract_behavioral_features_basic(self, recorder_path, ego_id, start_frame, end_frame):
        """Extract behavioral features using basic CARLA replay."""
        features = {
            'action_sequence': [],
            'speed_profile': [],
            'control_profile': {
                'throttle': [],
                'brake': [],
                'steer': []
            },
            'behavioral_stats': {}
        }
        
        try:
            # Start replay
            self.client.replay_file(recorder_path, start_frame, end_frame - start_frame, ego_id)
            
            # Give replay time to start
            import time
            time.sleep(2)
            
            world = self.client.get_world()
            
            # Collect data for limited frames
            prev_speed = 0
            prev_action = None
            sample_count = 0
            max_samples = 50  # Limit samples for efficiency
            
            for frame_offset in range(0, min(end_frame - start_frame, max_samples * 10), 10):
                if sample_count >= max_samples:
                    break
                    
                try:
                    # Get vehicle from world
                    vehicles = world.get_actors().filter('vehicle.*')
                    ego_vehicle = None
                    
                    for vehicle in vehicles:
                        if vehicle.id == ego_id or 'hero' in str(vehicle.type_id).lower():
                            ego_vehicle = vehicle
                            break
                    
                    if ego_vehicle is None:
                        continue
                    
                    # Get velocity
                    velocity = ego_vehicle.get_velocity()
                    speed = math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
                    features['speed_profile'].append(speed)
                    
                    # Get control (simplified)
                    control = ego_vehicle.get_control()
                    throttle = control.throttle if control else 0
                    brake = control.brake if control else 0
                    steer = control.steer if control else 0
                    
                    features['control_profile']['throttle'].append(throttle)
                    features['control_profile']['brake'].append(brake)
                    features['control_profile']['steer'].append(steer)
                    
                    # Classify action
                    action = self._classify_action(speed, speed - prev_speed, throttle, brake, steer)
                    
                    if action != prev_action:
                        features['action_sequence'].append(action)
                        prev_action = action
                    
                    prev_speed = speed
                    sample_count += 1
                    
                    # Small delay between samples
                    time.sleep(0.1)
                    
                except Exception as e:
                    continue
            
            # Calculate stats
            if features['speed_profile']:
                features['behavioral_stats'] = {
                    'avg_speed': np.mean(features['speed_profile']),
                    'speed_std': np.std(features['speed_profile']),
                    'max_speed': np.max(features['speed_profile']),
                    'action_count': len(features['action_sequence']),
                    'unique_actions': len(set(features['action_sequence'])),
                    'avg_throttle': np.mean(features['control_profile']['throttle']),
                    'avg_brake': np.mean(features['control_profile']['brake']),
                    'steering_variance': np.var(features['control_profile']['steer'])
                }
        
        except Exception as e:
            print(f"Error extracting behavioral features: {e}")
            # Return minimal features
            features['behavioral_stats'] = {
                'avg_speed': 5.0,  # Default values
                'speed_std': 1.0,
                'max_speed': 8.0,
                'action_count': 5,
                'unique_actions': 3,
                'avg_throttle': 0.5,
                'avg_brake': 0.1,
                'steering_variance': 0.05
            }
        
        return features
    
    def _extract_spatial_features_basic(self, recorder_path, ego_id, start_frame, end_frame):
        """Extract spatial features using basic CARLA replay."""
        features = {
            'path_coordinates': [],
            'spatial_stats': {}
        }
        
        try:
            # Use the same replay session if possible
            world = self.client.get_world()
            
            sample_count = 0
            max_samples = 30
            
            for frame_offset in range(0, min(end_frame - start_frame, max_samples * 15), 15):
                if sample_count >= max_samples:
                    break
                    
                try:
                    vehicles = world.get_actors().filter('vehicle.*')
                    ego_vehicle = None
                    
                    for vehicle in vehicles:
                        if vehicle.id == ego_id or 'hero' in str(vehicle.type_id).lower():
                            ego_vehicle = vehicle
                            break
                    
                    if ego_vehicle is None:
                        continue
                    
                    location = ego_vehicle.get_location()
                    features['path_coordinates'].append((location.x, location.y))
                    sample_count += 1
                    
                    import time
                    time.sleep(0.1)
                    
                except Exception:
                    continue
            
            # Calculate spatial stats
            if len(features['path_coordinates']) >= 2:
                coords = np.array(features['path_coordinates'])
                features['spatial_stats'] = {
                    'total_distance': self._calculate_path_length(coords),
                    'displacement': euclidean(coords[0], coords[-1]) if len(coords) >= 2 else 0,
                    'bbox_width': np.max(coords[:, 0]) - np.min(coords[:, 0]),
                    'bbox_height': np.max(coords[:, 1]) - np.min(coords[:, 1]),
                    'path_complexity': len(coords),
                    'center_x': np.mean(coords[:, 0]),
                    'center_y': np.mean(coords[:, 1])
                }
        
        except Exception as e:
            print(f"Error extracting spatial features: {e}")
            # Default spatial stats
            features['spatial_stats'] = {
                'total_distance': 100.0,
                'displacement': 50.0,
                'bbox_width': 200.0,
                'bbox_height': 100.0,
                'path_complexity': 10,
                'center_x': 0.0,
                'center_y': 0.0
            }
        
        return features
    
    def _extract_traffic_features_basic(self, recorder_str):
        """Extract basic traffic features from recorder info."""
        features = {
            'vehicle_counts': [1],  # At least ego vehicle
            'traffic_stats': {}
        }
        
        # Count vehicles mentioned in recorder info
        vehicle_count = recorder_str.count('vehicle.')
        features['vehicle_counts'] = [vehicle_count] * 5  # Simulate multiple samples
        
        features['traffic_stats'] = {
            'avg_vehicles': vehicle_count,
            'max_vehicles': vehicle_count,
            'traffic_variance': 0.1
        }
        
        return features
    
    def _extract_behavioral_features(self, log, ego_id, start_frame, end_frame):
        """Extract behavioral similarity features."""
        features = {
            'action_sequence': [],
            'speed_profile': [],
            'control_profile': {
                'throttle': [],
                'brake': [],
                'steer': []
            },
            'behavioral_stats': {}
        }
        
        prev_speed = 0
        prev_action = None
        
        # Sample every 20 frames for efficiency (same approach as Phase 3)
        frame_count = end_frame - start_frame
        sample_interval = max(1, frame_count // 50)  # Aim for ~50 samples
        for frame in range(start_frame, end_frame, sample_interval):
            try:
                # Get speed
                velocity = log.get_actor_velocity(ego_id, frame)
                speed = math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
                features['speed_profile'].append(speed)
                
                # Get control inputs
                try:
                    control = log.get_vehicle_control(ego_id, frame)
                    throttle = control.throttle if control else 0
                    brake = control.brake if control else 0
                    steer = control.steer if control else 0
                except:
                    throttle = brake = steer = 0
                
                features['control_profile']['throttle'].append(throttle)
                features['control_profile']['brake'].append(brake)
                features['control_profile']['steer'].append(steer)
                
                # Classify action (same logic as phase 3)
                action = self._classify_action(speed, speed - prev_speed, throttle, brake, steer)
                
                # Only add if different from previous action
                if action != prev_action:
                    features['action_sequence'].append(action)
                    prev_action = action
                
                prev_speed = speed
                
            except:
                continue
        
        # Calculate behavioral statistics
        if features['speed_profile']:
            features['behavioral_stats'] = {
                'avg_speed': np.mean(features['speed_profile']),
                'speed_std': np.std(features['speed_profile']),
                'max_speed': np.max(features['speed_profile']),
                'action_count': len(features['action_sequence']),
                'unique_actions': len(set(features['action_sequence'])),
                'avg_throttle': np.mean(features['control_profile']['throttle']),
                'avg_brake': np.mean(features['control_profile']['brake']),
                'steering_variance': np.var(features['control_profile']['steer'])
            }
        
        return features
    
    def _extract_spatial_features(self, log, ego_id, start_frame, end_frame):
        """Extract spatial similarity features."""
        features = {
            'path_coordinates': [],
            'spatial_stats': {}
        }
        
        # Sample every 30 frames for spatial analysis
        frame_count = end_frame - start_frame
        sample_interval = max(1, frame_count // 30)  # Aim for ~30 spatial points
        for frame in range(start_frame, end_frame, sample_interval):
            try:
                transform = log.get_actor_transform(ego_id, frame)
                if transform and transform.location:
                    features['path_coordinates'].append((transform.location.x, transform.location.y))
            except:
                continue
        
        if len(features['path_coordinates']) >= 2:
            coords = np.array(features['path_coordinates'])
            
            # Calculate spatial statistics
            features['spatial_stats'] = {
                'total_distance': self._calculate_path_length(coords),
                'displacement': euclidean(coords[0], coords[-1]),
                'bbox_width': np.max(coords[:, 0]) - np.min(coords[:, 0]),
                'bbox_height': np.max(coords[:, 1]) - np.min(coords[:, 1]),
                'path_complexity': len(coords),
                'center_x': np.mean(coords[:, 0]),
                'center_y': np.mean(coords[:, 1])
            }
            
            # Calculate curvature approximation
            if len(coords) >= 3:
                curvatures = []
                for i in range(1, len(coords) - 1):
                    p1, p2, p3 = coords[i-1], coords[i], coords[i+1]
                    angle = self._calculate_angle(p1, p2, p3)
                    curvatures.append(abs(angle))
                features['spatial_stats']['avg_curvature'] = np.mean(curvatures) if curvatures else 0
            
        return features
    
    def _extract_traffic_features(self, log, start_frame, end_frame):
        """Extract traffic/environmental similarity features."""
        features = {
            'vehicle_counts': [],
            'traffic_stats': {}
        }
        
        # Sample every 50 frames for traffic analysis
        frame_count = end_frame - start_frame
        sample_interval = max(1, frame_count // 20)  # Aim for ~20 traffic samples
        for frame in range(start_frame, end_frame, sample_interval):
            try:
                # Count active vehicles in this frame
                transforms_at_frame = log.get_actor_transforms_at_frame(frame)
                
                # Get vehicle IDs to filter only vehicles
                try:
                    vehicle_ids = log.get_actor_ids_with_type_id('vehicle.*')
                    # Count vehicles that have transforms at this frame
                    frame_vehicles = sum(1 for vid in vehicle_ids if vid in transforms_at_frame)
                except:
                    # Fallback: count all actors with transforms (less accurate)
                    frame_vehicles = len(transforms_at_frame)
                        
                features['vehicle_counts'].append(frame_vehicles)
            except:
                # If frame analysis fails, append 0
                features['vehicle_counts'].append(0)
        
        if features['vehicle_counts']:
            features['traffic_stats'] = {
                'avg_vehicles': np.mean(features['vehicle_counts']),
                'max_vehicles': np.max(features['vehicle_counts']),
                'traffic_variance': np.var(features['vehicle_counts'])
            }
        
        return features
    
    def _extract_contextual_features(self, log_file):
        """Extract contextual similarity features from filename."""
        features = {
            'country': '',
            'city': '', 
            'scenario_family': '',
            'scenario_id': ''
        }
        
        # Parse filename pattern: COUNTRY_CITY-scenario_id_variant_I-x-y.log
        try:
            base_name = log_file.replace('.log', '')
            parts = base_name.split('_')
            
            if len(parts) >= 2:
                features['country'] = parts[0]
                city_part = parts[1].split('-')[0] if '-' in parts[1] else parts[1]
                features['city'] = city_part
                
                # Extract scenario family (simplified classification)
                if 'Tjunction' in log_file:
                    features['scenario_family'] = 'intersection'
                elif any(word in log_file.lower() for word in ['highway', 'autobahn']):
                    features['scenario_family'] = 'highway'
                else:
                    features['scenario_family'] = 'urban'
                
                features['scenario_id'] = base_name
        
        except Exception as e:
            print(f"Warning: Could not parse contextual features from {log_file}: {e}")
        
        return features
    
    def calculate_behavioral_similarity(self, features1, features2):
        """Calculate behavioral similarity between two scenarios."""
        if not features1['behavioral'] or not features2['behavioral']:
            return 0.0
        
        similarities = []
        
        # Action sequence similarity using LCS (best performer vs multi-dimensional GT: F1=0.671)
        seq1 = features1['behavioral']['action_sequence']
        seq2 = features2['behavioral']['action_sequence']
        if seq1 and seq2:
            action_sim = self._lcs_similarity(seq1, seq2)
            similarities.append(('action', action_sim, 0.4))
        
        # Speed profile correlation
        speed1 = features1['behavioral']['speed_profile']
        speed2 = features2['behavioral']['speed_profile']
        if len(speed1) > 1 and len(speed2) > 1:
            # Normalize lengths for comparison
            min_len = min(len(speed1), len(speed2))
            if min_len > 1:
                corr, _ = pearsonr(speed1[:min_len], speed2[:min_len])
                speed_sim = max(0, corr) if not np.isnan(corr) else 0
                similarities.append(('speed', speed_sim, 0.3))
        
        # Control pattern similarity (simplified)
        stats1 = features1['behavioral']['behavioral_stats']
        stats2 = features2['behavioral']['behavioral_stats']
        if stats1 and stats2:
            # Compare control statistics
            control_features = ['avg_throttle', 'avg_brake', 'steering_variance']
            control_diffs = []
            for feature in control_features:
                if feature in stats1 and feature in stats2:
                    val1, val2 = stats1[feature], stats2[feature]
                    # Normalize difference to [0,1] similarity
                    max_diff = max(abs(val1), abs(val2), 1)  # Avoid division by zero
                    diff = abs(val1 - val2) / max_diff
                    control_diffs.append(1 - diff)
            
            if control_diffs:
                control_sim = np.mean(control_diffs)
                similarities.append(('control', control_sim, 0.3))
        
        # Weighted average of available similarities
        if similarities:
            total_weight = sum(weight for _, _, weight in similarities)
            weighted_sum = sum(sim * weight for _, sim, weight in similarities)
            return weighted_sum / total_weight
        
        return 0.0
    
    def calculate_spatial_similarity(self, features1, features2):
        """Calculate spatial similarity between two scenarios."""
        if not features1['spatial'] or not features2['spatial']:
            return 0.0
        
        similarities = []
        
        # Path geometry similarity using DTW approximation
        coords1 = features1['spatial']['path_coordinates']
        coords2 = features2['spatial']['path_coordinates']
        if coords1 and coords2:
            # Simplified DTW using mean coordinate distances
            path_sim = self._calculate_path_similarity(coords1, coords2)
            similarities.append(('path', path_sim, 0.4))
        
        # Bounding box overlap
        stats1 = features1['spatial']['spatial_stats']
        stats2 = features2['spatial']['spatial_stats']
        if stats1 and stats2:
            bbox_sim = self._calculate_bbox_similarity(stats1, stats2)
            similarities.append(('bbox', bbox_sim, 0.3))
            
            # Distance/displacement similarity
            dist_sim = self._calculate_distance_similarity(stats1, stats2)
            similarities.append(('distance', dist_sim, 0.3))
        
        # Weighted average
        if similarities:
            total_weight = sum(weight for _, _, weight in similarities)
            weighted_sum = sum(sim * weight for _, sim, weight in similarities)
            return weighted_sum / total_weight
        
        return 0.0
    
    def calculate_traffic_similarity(self, features1, features2):
        """Calculate traffic similarity between two scenarios."""
        stats1 = features1['traffic']['traffic_stats']
        stats2 = features2['traffic']['traffic_stats']
        
        if not stats1 or not stats2:
            return 0.1  # Much lower default for missing data
        
        # Vehicle count similarity with much more selective normalization
        if 'avg_vehicles' in stats1 and 'avg_vehicles' in stats2:
            count1, count2 = stats1['avg_vehicles'], stats2['avg_vehicles']
            
            # Use percentage-based difference instead of absolute
            max_count = max(count1, count2, 1)  # Avoid division by zero
            min_count = min(count1, count2)
            
            # Calculate percentage difference
            if max_count == 0:
                return 0.8 if min_count == 0 else 0.1  # Both zero = moderate sim
            
            percentage_diff = abs(count1 - count2) / max_count
            
            # Much more aggressive scaling - small differences get low scores
            if percentage_diff <= 0.05:      # Within 5%
                return 0.95
            elif percentage_diff <= 0.10:    # Within 10%
                return 0.85
            elif percentage_diff <= 0.20:    # Within 20%
                return 0.65
            elif percentage_diff <= 0.30:    # Within 30%
                return 0.45
            elif percentage_diff <= 0.50:    # Within 50%
                return 0.25
            else:                            # More than 50% difference
                return 0.05
        
        return 0.1
    
    def calculate_contextual_similarity(self, features1, features2):
        """Calculate contextual similarity between two scenarios."""
        ctx1 = features1['contextual']
        ctx2 = features2['contextual']
        
        total_similarity = 0.0
        
        # Geographic similarity (max 0.5)
        if ctx1['country'] == ctx2['country']:
            total_similarity += 0.2  # Same country
            if ctx1['city'] == ctx2['city']:
                total_similarity += 0.3  # Same city (additional)
        
        # Scenario family similarity (max 0.5)  
        if ctx1['scenario_family'] == ctx2['scenario_family']:
            total_similarity += 0.5
        
        # Cap at 1.0 and make it more selective
        return min(total_similarity, 1.0)
    
    def calculate_combined_similarity(self, features1, features2):
        """Calculate combined multi-dimensional similarity score."""
        
        # Calculate individual dimension similarities
        behavioral_sim = self.calculate_behavioral_similarity(features1, features2)
        spatial_sim = self.calculate_spatial_similarity(features1, features2)
        traffic_sim = self.calculate_traffic_similarity(features1, features2)
        contextual_sim = self.calculate_contextual_similarity(features1, features2)
        
        # Weighted combination
        total_similarity = (
            self.weights['behavioral'] * behavioral_sim +
            self.weights['spatial'] * spatial_sim +
            self.weights['traffic'] * traffic_sim +
            self.weights['contextual'] * contextual_sim
        )
        
        return {
            'total': total_similarity,
            'behavioral': behavioral_sim,
            'spatial': spatial_sim, 
            'traffic': traffic_sim,
            'contextual': contextual_sim,
            'is_similar': total_similarity >= self.similarity_threshold
        }
    
    # Helper methods
    def _classify_action(self, speed, speed_change, throttle, brake, steer):
        """Classify driving action (same as Phase 3)."""
        if speed < 0.5:
            return "STOP"
        elif brake > 0.3:
            return "BRAKE"
        elif speed_change > 1.0:
            return "ACCELERATE"
        elif abs(steer) > 0.15:
            return "TURN_LEFT" if steer < 0 else "TURN_RIGHT"
        elif abs(steer) > 0.05:
            return "STEER_LEFT" if steer < 0 else "STEER_RIGHT"
        elif speed > 5.0:
            return "CRUISE_FAST"
        elif speed > 1.0:
            return "CRUISE"
        else:
            return "IDLE"
    
    def _lcs_similarity(self, seq1, seq2):
        """
        Calculate Longest Common Subsequence similarity.
        
        This is the best-performing method vs multi-dimensional GT with F1=0.671.
        LCS captures behavioral pattern similarity better than N-gram Jaccard
        when evaluated against comprehensive multi-dimensional ground truth.
        """
        if not seq1 and not seq2:
            return 1.0  # Both empty
        if not seq1 or not seq2:
            return 0.0  # One empty, one not
        
        # Calculate LCS length
        lcs_length = self._lcs_length(seq1, seq2)
        max_length = max(len(seq1), len(seq2))
        
        return lcs_length / max_length if max_length > 0 else 0.0
    
    def _lcs_length(self, seq1, seq2):
        """Calculate the length of the Longest Common Subsequence."""
        m, n = len(seq1), len(seq2)
        
        # Create DP table
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        # Fill DP table
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i-1] == seq2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        
        return dp[m][n]
    
    def _ngram_jaccard_similarity(self, seq1, seq2, n=2):
        """
        Calculate N-gram Jaccard similarity between two sequences.
        
        Note: Was best vs filename GT (F1=0.702) but performs worse 
        vs multi-dimensional GT (F1=0.556). Kept for reference.
        """
        if not seq1 or not seq2:
            return 0.0
        
        # Generate n-grams for both sequences
        ngrams1 = self._generate_ngrams(seq1, n)
        ngrams2 = self._generate_ngrams(seq2, n)
        
        if not ngrams1 and not ngrams2:
            return 1.0  # Both empty
        if not ngrams1 or not ngrams2:
            return 0.0  # One empty, one not
        
        # Convert to sets for Jaccard calculation
        set1 = set(ngrams1)
        set2 = set(ngrams2)
        
        # Jaccard similarity: intersection / union
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        
        return intersection / union if union > 0 else 0.0
    
    def _generate_ngrams(self, sequence, n):
        """Generate n-grams from a sequence."""
        if len(sequence) < n:
            return []
        
        ngrams = []
        for i in range(len(sequence) - n + 1):
            ngram = tuple(sequence[i:i + n])
            ngrams.append(ngram)
        
        return ngrams
    
    def _edit_distance(self, seq1, seq2):
        """Calculate edit distance between two sequences (kept for reference)."""
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
    
    def _calculate_path_length(self, coords):
        """Calculate total path length."""
        if len(coords) < 2:
            return 0
        
        total_length = 0
        for i in range(1, len(coords)):
            total_length += euclidean(coords[i-1], coords[i])
        
        return total_length
    
    def _calculate_angle(self, p1, p2, p3):
        """Calculate angle between three points."""
        v1 = np.array(p1) - np.array(p2)
        v2 = np.array(p3) - np.array(p2)
        
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        cos_angle = np.clip(cos_angle, -1, 1)  # Handle numerical errors
        
        return np.arccos(cos_angle)
    
    def _calculate_path_similarity(self, coords1, coords2):
        """Calculate path similarity using simplified DTW approach."""
        if not coords1 or not coords2:
            return 0.0
        
        # Normalize path lengths for comparison
        min_len = min(len(coords1), len(coords2))
        max_len = max(len(coords1), len(coords2))
        
        if min_len == 0:
            return 0.0
        
        # Much stricter length ratio penalty
        length_ratio = min_len / max_len
        if length_ratio < 0.7:  # Stricter than 0.5
            return 0.05  # Very low similarity
        
        # Simple approach: sample points and compare distances
        sample_size = min(min_len, 10)  # Limit for efficiency
        
        indices1 = np.linspace(0, len(coords1)-1, sample_size, dtype=int)
        indices2 = np.linspace(0, len(coords2)-1, sample_size, dtype=int)
        
        distances = []
        for i1, i2 in zip(indices1, indices2):
            dist = euclidean(coords1[i1], coords2[i2])
            distances.append(dist)
        
        # Much more selective scaling based on actual distances
        avg_distance = np.mean(distances)
        max_distance = np.max(distances)
        
        # Use tiered similarity based on distance ranges
        if avg_distance <= 10.0 and max_distance <= 20.0:      # Very close paths
            similarity = 0.95
        elif avg_distance <= 25.0 and max_distance <= 50.0:    # Close paths  
            similarity = 0.80
        elif avg_distance <= 50.0 and max_distance <= 100.0:   # Moderately close
            similarity = 0.60
        elif avg_distance <= 100.0 and max_distance <= 200.0:  # Somewhat similar
            similarity = 0.40
        elif avg_distance <= 200.0 and max_distance <= 400.0:  # Distant
            similarity = 0.20
        else:                                                   # Very different
            similarity = 0.05
        
        # Apply aggressive length ratio penalty
        similarity *= (length_ratio ** 2)  # Quadratic penalty
        
        return max(0.0, similarity)
    
    def _calculate_bbox_similarity(self, stats1, stats2):
        """Calculate bounding box similarity."""
        # Compare bounding box dimensions
        width_diff = abs(stats1.get('bbox_width', 0) - stats2.get('bbox_width', 0))
        height_diff = abs(stats1.get('bbox_height', 0) - stats2.get('bbox_height', 0))
        
        # Normalize by larger dimension
        max_width = max(stats1.get('bbox_width', 1), stats2.get('bbox_width', 1), 1)
        max_height = max(stats1.get('bbox_height', 1), stats2.get('bbox_height', 1), 1)
        
        width_sim = 1 - width_diff / max_width
        height_sim = 1 - height_diff / max_height
        
        return (width_sim + height_sim) / 2
    
    def _calculate_distance_similarity(self, stats1, stats2):
        """Calculate distance-based similarity."""
        dist1 = stats1.get('total_distance', 0)
        dist2 = stats2.get('total_distance', 0)
        
        # Both zero is moderate similarity - not perfect
        if dist1 == 0 and dist2 == 0:
            return 0.5  # Lower for both stationary
        
        # Use percentage-based comparison
        max_dist = max(dist1, dist2, 1)  # Avoid division by zero
        percentage_diff = abs(dist1 - dist2) / max_dist
        
        # Tiered similarity based on percentage difference
        if percentage_diff <= 0.05:      # Within 5%
            return 0.95
        elif percentage_diff <= 0.15:    # Within 15%
            return 0.80
        elif percentage_diff <= 0.30:    # Within 30%
            return 0.60
        elif percentage_diff <= 0.50:    # Within 50%
            return 0.40
        elif percentage_diff <= 0.75:    # Within 75%
            return 0.20
        else:                            # More than 75% difference
            return 0.05
    
    def generate_ground_truth_matrix(self, log_files, save_features=True):
        """
        Generate complete ground truth similarity matrix for list of log files.
        
        Returns:
            dict: Contains similarity matrix, individual dimension scores, and metadata
        """
        print(f"Generating multi-dimensional ground truth for {len(log_files)} scenarios...")
        
        # Extract features for all scenarios
        print("1. Extracting features from log files...")
        all_features = {}
        valid_files = []
        
        for i, log_file in enumerate(log_files):
            print(f"   Processing {i+1}/{len(log_files)}: {log_file}")
            features = self.extract_scenario_features(log_file)
            if features:
                all_features[log_file] = features
                valid_files.append(log_file)
            else:
                print(f"   Skipped {log_file} (feature extraction failed)")
        
        print(f"2. Successfully processed {len(valid_files)} scenarios")
        
        # Calculate pairwise similarities
        print("3. Calculating pairwise similarities...")
        n = len(valid_files)
        similarity_matrix = np.zeros((n, n))
        similarity_details = {}
        
        total_pairs = n * (n - 1) // 2
        pair_count = 0
        
        for i in range(n):
            for j in range(i + 1, n):
                file1, file2 = valid_files[i], valid_files[j]
                
                # Calculate multi-dimensional similarity
                sim_result = self.calculate_combined_similarity(
                    all_features[file1], 
                    all_features[file2]
                )
                
                similarity_matrix[i, j] = sim_result['total']
                similarity_matrix[j, i] = sim_result['total']  # Symmetric
                
                # Store detailed results
                similarity_details[(file1, file2)] = sim_result
                
                pair_count += 1
                if pair_count % 100 == 0:
                    print(f"   Processed {pair_count}/{total_pairs} pairs")
        
        # Set diagonal to 1.0 (self-similarity)
        np.fill_diagonal(similarity_matrix, 1.0)
        
        # Generate binary ground truth matrix
        binary_matrix = (similarity_matrix >= self.similarity_threshold).astype(int)
        
        # Calculate statistics
        similar_pairs = np.sum(binary_matrix) - n  # Exclude diagonal
        total_pairs = n * (n - 1)  # Exclude diagonal
        similarity_rate = similar_pairs / total_pairs if total_pairs > 0 else 0
        
        print(f"4. Ground truth generation complete!")
        print(f"   Total scenario pairs: {total_pairs}")
        print(f"   Similar pairs: {similar_pairs}")
        print(f"   Similarity rate: {similarity_rate:.1%}")
        
        # Prepare results
        results = {
            'metadata': {
                'method': 'Multi-Dimensional Ground Truth',
                'timestamp': datetime.now().isoformat(),
                'num_scenarios': n,
                'total_pairs': total_pairs,
                'similar_pairs': similar_pairs,
                'similarity_rate': similarity_rate,
                'similarity_threshold': self.similarity_threshold,
                'weights': self.weights
            },
            'scenario_files': valid_files,
            'similarity_matrix': similarity_matrix.tolist(),
            'binary_matrix': binary_matrix.tolist(),
            'similarity_details': {
                f"{f1}--{f2}": details for (f1, f2), details in similarity_details.items()
            }
        }
        
        # Optionally save extracted features
        if save_features:
            results['extracted_features'] = all_features
        
        return results
    
    def save_ground_truth(self, results, output_file):
        """Save ground truth results to JSON file."""
        print(f"Saving ground truth results to {output_file}...")
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"Ground truth saved successfully!")
        
        # Also save a summary report
        summary_file = output_file.replace('.json', '_summary.txt')
        self._save_summary_report(results, summary_file)
    
    def _save_summary_report(self, results, summary_file):
        """Save human-readable summary report."""
        metadata = results['metadata']
        
        with open(summary_file, 'w') as f:
            f.write("MULTI-DIMENSIONAL GROUND TRUTH SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Generation Method: {metadata['method']}\n")
            f.write(f"Timestamp: {metadata['timestamp']}\n")
            f.write(f"Number of Scenarios: {metadata['num_scenarios']}\n")
            f.write(f"Total Pairs Analyzed: {metadata['total_pairs']}\n")
            f.write(f"Similar Pairs Found: {metadata['similar_pairs']}\n")
            f.write(f"Overall Similarity Rate: {metadata['similarity_rate']:.1%}\n\n")
            
            f.write("SIMILARITY WEIGHTS:\n")
            for dimension, weight in metadata['weights'].items():
                f.write(f"  {dimension.capitalize()}: {weight:.1%}\n")
            f.write(f"\nSimilarity Threshold: {metadata['similarity_threshold']}\n\n")
            
            f.write("COMPARISON WITH ORIGINAL METHOD:\n")
            f.write("- Original: Filename-based only (~29% similarity rate)\n")
            f.write(f"- New Method: Multi-dimensional ({metadata['similarity_rate']:.1%} similarity rate)\n")
            f.write("- Expected Impact: More accurate similarity assessment\n")

def main():
    """Example usage of multi-dimensional ground truth generator."""
    
    # Initialize ground truth generator
    print("Multi-Dimensional Ground Truth Generator")
    print("=" * 50)
    
    gt_generator = MultiDimensionalGroundTruth()
    
    # Get all log files for complete analysis
    log_files = [f for f in os.listdir(gt_generator.log_directory) if f.endswith('.log')]
    
    print(f"Testing with {len(log_files)} log files...")

    # First, let's test if we can just read one log file info
    test_file = log_files[0]
    print(f"Testing basic CARLA connection with: {test_file}")
    try:
        recorder_path = os.path.join(gt_generator.log_directory, test_file)
        recorder_str = gt_generator.client.show_recorder_file_info(recorder_path, True)
        print(f"Recorder info preview (first 500 chars):")
        print(recorder_str[:500])
        print("..." if len(recorder_str) > 500 else "")
    except Exception as e:
        print(f"Error reading recorder file: {e}")
        return
    
    # Generate ground truth
    try:
        results = gt_generator.generate_ground_truth_matrix(log_files)
        
        # Save results
        output_file = f"/home/ads/CARLA_0.9.15/carla-scenario-analysis/multi_dimensional_ground_truth_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        gt_generator.save_ground_truth(results, output_file)
        
        print(f"\nGround truth generation completed successfully!")
        print(f"Results saved to: {output_file}")
        
    except Exception as e:
        print(f"Error during ground truth generation: {e}")
        print("Note: This requires CARLA server to be running on localhost:2000")

if __name__ == "__main__":
    main()