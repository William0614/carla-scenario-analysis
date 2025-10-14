#!/usr/bin/env python3
"""
Feature Extraction Module for CARLA Scenario Analysis

Extracts 32-dimensional feature vectors from CARLA log files for similarity analysis.
"""

import os
import sys
import carla
import math
import numpy as np
from collections import Counter

# Add the scenario runner to the path
sys.path.append('/home/ads/ads_testing/scenario_runner/scenario_runner-0.9.15')

from srunner.metrics.tools.metrics_log import MetricsLog


class FeatureExtractor:
    """
    Comprehensive 32-dimensional feature extraction from CARLA scenario logs.
    
    Feature Categories:
        - Temporal Features (4 dimensions): Duration, timing patterns  
    - Motion Features (8 dimensions): Speed, acceleration, and dynamics
    - Behavioral Features (8 dimensions): Driving patterns, maneuvers
    - Spatial Features (8 dimensions): Geographic movement, trajectories
    - Context Features (4 dimensions): Traffic and environmental context
    """
    
    def __init__(self, carla_host='localhost', carla_port=2000):
        """Initialize the feature extractor with CARLA client connection."""
        self.client = carla.Client(carla_host, carla_port)
        self.client.set_timeout(10.0)
    
    def extract_features(self, log_file):
        """
        Extract comprehensive 32-dimensional feature vector from a log file.
        
        Args:
            log_file (str): Path to the CARLA log file
            
        Returns:
            dict: Feature dictionary with temporal, motion, behavioral, spatial, 
                  context features and combined 32-dimensional vector
        """
        try:
            recorder_file = f"{os.getenv('SCENARIO_RUNNER_ROOT', './')}/{log_file}"
            recorder_str = self.client.show_recorder_file_info(recorder_file, True)
            log = MetricsLog(recorder_str)
            
            ego_id = log.get_ego_vehicle_id()
            start_frame, end_frame = log.get_actor_alive_frames(ego_id)
            
            # Initialize feature dictionary
            features = {
                'temporal_features': [],
                'motion_features': [],
                'behavioral_features': [],
                'spatial_features': [],
                'context_features': [],
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
            
            # Extract raw data from frames
            for frame in sample_frames:
                try:
                    # Velocity and speed
                    velocity = log.get_actor_velocity(ego_id, frame)
                    speed = math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
                    speeds.append(speed)
                    
                    # Position data
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
            
            features['motion_features'] = self._extract_motion_features(speeds, accelerations)
            
            # Traffic context
            vehicle_ids = log.get_actor_ids_with_type_id("vehicle.*")
            traffic_count = len([vid for vid in vehicle_ids if vid != ego_id])
            
            features['context_features'] = self._extract_context_features(positions, traffic_count)
            
            # Create combined 32-dimensional vector
            features['combined_vector'] = (
                features['temporal_features'] + 
                features['motion_features'] + 
                features['behavioral_features'] + 
                features['spatial_features'] + 
                features['context_features']
            )
            
            return features
            
        except Exception as e:
            print(f"Error extracting features from {log_file}: {e}")
            return None
    
    def _encode_behavior(self, speed, prev_speed, steer):
        """Encode behavior into numerical representation."""
        speed_diff = speed - prev_speed if prev_speed is not None else 0
        
        if speed < 0.5:
            return 0  # Stop
        elif speed_diff > 1.0:
            return 1  # Accelerate
        elif speed_diff < -1.0:
            return 2  # Decelerate  
        elif steer > 0.15:
            return 3  # Left turn
        elif steer < -0.15:
            return 4  # Right turn
        elif speed > 2.0:
            return 5  # Normal cruise
        else:
            return 6  # Idle
    
    def _extract_temporal_features(self, speeds, accelerations, frame_count):
        """Extract temporal characteristics (4 features)."""
        if not speeds:
            return [0] * 4
        
        duration = frame_count * 0.05  # Assuming 20 FPS
        dynamic_events = (
            sum(1 for i in range(1, len(speeds)) if abs(speeds[i] - speeds[i-1]) > 1.0) +
            len([a for a in accelerations if abs(a) > 2.5])
        )
        
        return [
            duration,                                           # 1. Duration (seconds)
            frame_count,                                        # 2. Frame count
            dynamic_events / duration if duration > 0 else 0,   # 3. Event frequency (events/sec)
            dynamic_events / frame_count if frame_count > 0 else 0  # 4. Temporal density (events/frame)
        ]
    
    def _extract_motion_features(self, speeds, accelerations):
        """Extract motion characteristics (8 features)."""
        if not speeds:
            return [0] * 8
        
        speeds_array = np.array(speeds)
        acc_array = np.array(accelerations) if accelerations else np.array([0])
        
        return [
            np.mean(speeds_array),                             # 5. Mean speed
            np.std(speeds_array),                              # 6. Speed standard deviation
            np.min(speeds_array),                              # 7. Minimum speed
            np.max(speeds_array),                              # 8. Maximum speed
            np.max(speeds_array) - np.min(speeds_array),       # 9. Speed range
            np.mean(acc_array),                                # 10. Mean acceleration
            np.std(acc_array),                                 # 11. Acceleration standard deviation
            len([a for a in accelerations if abs(a) > 2.5])   # 12. Dynamic events count (unified threshold)
        ]
    
    def _extract_behavioral_features(self, behaviors, steering_angles):
        """Extract behavioral pattern features (8 features)."""
        if not behaviors:
            return [0] * 8
        
        behavior_counts = Counter(behaviors)
        behavior_transitions = len(set(zip(behaviors[:-1], behaviors[1:]))) if len(behaviors) > 1 else 0
        
        avg_steer = np.mean([abs(s) for s in steering_angles]) if steering_angles else 0
        max_steer = max([abs(s) for s in steering_angles]) if steering_angles else 0
        
        return [
            behavior_counts.get(0, 0),                          # 13. Stop events count
            behavior_counts.get(1, 0),                          # 14. Acceleration events count
            behavior_counts.get(2, 0),                          # 15. Deceleration events count
            behavior_counts.get(3, 0) + behavior_counts.get(4, 0),  # 16. Turn maneuvers count
            behavior_counts.get(5, 0) + behavior_counts.get(6, 0),  # 17. Cruise behavior count
            behavior_transitions,                               # 18. Behavior transitions count
            avg_steer,                                          # 19. Average steering magnitude
            max_steer                                           # 20. Maximum steering magnitude
        ]
    
    def _extract_spatial_features(self, positions):
        """Extract spatial movement characteristics (8 features)."""
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
        
        # Direction changes (curvature estimation)
        direction_changes = 0
        for i in range(2, len(positions)):
            v1 = (positions[i-1][0] - positions[i-2][0], positions[i-1][1] - positions[i-2][1])
            v2 = (positions[i][0] - positions[i-1][0], positions[i][1] - positions[i-1][1])
            cross_product = abs(v1[0] * v2[1] - v1[1] * v2[0])
            if cross_product > 50:  # Threshold for significant direction change
                direction_changes += 1
        
        return [
            path_length,                                        # 17. Total path length
            total_displacement,                                 # 18. Total displacement
            path_length / total_displacement if total_displacement > 0 else 1,  # 19. Path tortuosity ratio
            bbox_width,                                         # 20. Bounding box width
            bbox_height,                                        # 21. Bounding box height
            bbox_width * bbox_height,                          # 22. Bounding box area
            direction_changes,                                  # 23. Direction changes count
            direction_changes / len(positions) if positions else 0  # 24. Curvature density
        ]
    
    def _extract_context_features(self, positions, traffic_count):
        """Extract environmental context features (4 features)."""
        # Calculate spatial area for traffic density
        if len(positions) >= 2:
            xs = [p[0] for p in positions]
            ys = [p[1] for p in positions]
            bbox_area = (max(xs) - min(xs)) * (max(ys) - min(ys))
            traffic_density = traffic_count / max(bbox_area, 1) if bbox_area > 0 else 0
        else:
            traffic_density = 0
        
        # Simple complexity score combining spatial and traffic factors
        complexity_score = min(10, traffic_count + len(positions) / 100)
        
        return [
            traffic_count,                                     # 29. Traffic count
            traffic_density,                                   # 30. Traffic density (vehicles/area)  
            1 if traffic_count > 0 else 0,                     # 31. Traffic presence indicator
            complexity_score                                   # 32. Scenario complexity score
        ]


# Utility functions for batch processing
def extract_features_batch(log_files_dir, extractor=None):
    """
    Extract features from all log files in a directory.
    
    Args:
        log_files_dir (str): Directory containing CARLA log files
        extractor (FeatureExtractor): Optional pre-initialized extractor
        
    Returns:
        dict: Dictionary mapping filenames to their feature vectors
    """
    if extractor is None:
        extractor = FeatureExtractor()
    
    import glob
    log_files = glob.glob(os.path.join(log_files_dir, "*.log"))
    features_dict = {}
    
    for log_file in log_files:
        filename = os.path.basename(log_file)
        print(f"Processing {filename}...")
        
        features = extractor.extract_features(f"log_files/{filename}")
        if features is not None:
            features_dict[filename] = features
    
    return features_dict