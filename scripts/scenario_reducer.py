#!/usr/bin/env python

import os
import sys
import carla
import math
import json
import glob
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

# Add the scenario runner to the path
sys.path.append('/home/ads/ads_testing/scenario_runner/scenario_runner-0.9.15')

from srunner.metrics.tools.metrics_log import MetricsLog

class ScenarioReducer:
    """
    Comprehensive scenario reduction analysis tool
    """
    
    def __init__(self, log_files_dir):
        self.log_files_dir = log_files_dir
        self.scenarios = {}
        self.client = carla.Client('localhost', 2000)
        
    def extract_scenario_features(self, log_file):
        """Extract comprehensive features from a scenario log"""
        try:
            recorder_file = "{}/{}".format(os.getenv('SCENARIO_RUNNER_ROOT', "./"), log_file)
            recorder_str = self.client.show_recorder_file_info(recorder_file, True)
            log = MetricsLog(recorder_str)
            
            # Get ego vehicle
            ego_id = log.get_ego_vehicle_id()
            start_frame, end_frame = log.get_actor_alive_frames(ego_id)
            
            # Extract features
            features = {
                'file': log_file,
                'duration': log.get_elapsed_time(end_frame - 1) if end_frame > start_frame else 0,
                'total_frames': end_frame - start_frame,
                'behavioral_sequence': [],
                'speed_profile': [],
                'spatial_path': [],
                'vehicle_interactions': [],
                'traffic_complexity': 0,
                'maneuvers': {
                    'accelerations': 0,
                    'decelerations': 0,
                    'left_turns': 0,
                    'right_turns': 0,
                    'lane_changes': 0,
                    'stops': 0
                }
            }
            
            # Sample frames for analysis (every 10th frame for efficiency)
            sample_frames = range(start_frame, end_frame, 10)
            
            prev_speed = 0
            prev_steer = 0
            behavioral_sequence = []
            
            for frame in sample_frames:
                try:
                    # Get current state
                    velocity = log.get_actor_velocity(ego_id, frame)
                    speed = math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
                    
                    transform = log.get_actor_transform(ego_id, frame)
                    features['spatial_path'].append((transform.location.x, transform.location.y))
                    features['speed_profile'].append(speed)
                    
                    # Get control inputs
                    try:
                        control = log.get_vehicle_control(ego_id, frame)
                        throttle = control.throttle if control else 0
                        brake = control.brake if control else 0
                        steer = control.steer if control else 0
                    except:
                        throttle = brake = steer = 0
                    
                    # Behavioral pattern detection
                    behavior = self._detect_behavior(speed, prev_speed, steer, prev_steer, throttle, brake)
                    if behavior and (not behavioral_sequence or behavioral_sequence[-1] != behavior):
                        behavioral_sequence.append(behavior)
                        
                        # Count specific maneuvers
                        if behavior == 'accelerate':
                            features['maneuvers']['accelerations'] += 1
                        elif behavior == 'decelerate':
                            features['maneuvers']['decelerations'] += 1
                        elif behavior == 'left_turn':
                            features['maneuvers']['left_turns'] += 1
                        elif behavior == 'right_turn':
                            features['maneuvers']['right_turns'] += 1
                        elif behavior == 'stop':
                            features['maneuvers']['stops'] += 1
                    
                    prev_speed = speed
                    prev_steer = steer
                    
                except Exception as e:
                    continue
            
            features['behavioral_sequence'] = behavioral_sequence
            
            # Calculate traffic complexity (number of other vehicles)
            vehicle_ids = log.get_actor_ids_with_type_id("vehicle.*")
            features['traffic_complexity'] = len([vid for vid in vehicle_ids if vid != ego_id])
            
            # Calculate spatial complexity (path length and curvature)
            if len(features['spatial_path']) > 1:
                path_length = self._calculate_path_length(features['spatial_path'])
                curvature = self._calculate_path_curvature(features['spatial_path'])
                features['path_length'] = path_length
                features['path_curvature'] = curvature
            else:
                features['path_length'] = 0
                features['path_curvature'] = 0
            
            # Speed statistics
            if features['speed_profile']:
                features['avg_speed'] = np.mean(features['speed_profile'])
                features['max_speed'] = np.max(features['speed_profile'])
                features['speed_variance'] = np.var(features['speed_profile'])
            else:
                features['avg_speed'] = features['max_speed'] = features['speed_variance'] = 0
                
            return features
            
        except Exception as e:
            print(f"Error processing {log_file}: {e}")
            return None
    
    def _detect_behavior(self, speed, prev_speed, steer, prev_steer, throttle, brake):
        """Detect current driving behavior"""
        speed_threshold = 0.5  # m/s
        steer_threshold = 0.1
        
        if brake > 0.3:
            return 'hard_brake'
        elif speed < 0.5 and prev_speed < 0.5:
            return 'stop'
        elif speed > prev_speed + speed_threshold:
            return 'accelerate'
        elif speed < prev_speed - speed_threshold:
            return 'decelerate'
        elif steer > steer_threshold:
            return 'left_turn'
        elif steer < -steer_threshold:
            return 'right_turn'
        elif abs(steer - prev_steer) > 0.2:
            return 'lane_change'
        else:
            return 'cruise'
    
    def _calculate_path_length(self, path):
        """Calculate total path length"""
        length = 0
        for i in range(1, len(path)):
            dx = path[i][0] - path[i-1][0]
            dy = path[i][1] - path[i-1][1]
            length += math.sqrt(dx*dx + dy*dy)
        return length
    
    def _calculate_path_curvature(self, path):
        """Calculate average path curvature"""
        if len(path) < 3:
            return 0
        
        curvatures = []
        for i in range(1, len(path) - 1):
            # Calculate curvature using three consecutive points
            p1, p2, p3 = path[i-1], path[i], path[i+1]
            
            # Vectors
            v1 = (p2[0] - p1[0], p2[1] - p1[1])
            v2 = (p3[0] - p2[0], p3[1] - p2[1])
            
            # Cross product magnitude (proportional to curvature)
            cross = abs(v1[0] * v2[1] - v1[1] * v2[0])
            
            # Lengths
            len1 = math.sqrt(v1[0]**2 + v1[1]**2)
            len2 = math.sqrt(v2[0]**2 + v2[1]**2)
            
            if len1 > 0 and len2 > 0:
                curvature = cross / (len1 * len2)
                curvatures.append(curvature)
        
        return np.mean(curvatures) if curvatures else 0
    
    def analyze_all_scenarios(self):
        """Analyze all log files in the directory"""
        log_files = glob.glob(os.path.join(self.log_files_dir, "*.log"))
        print(f"Found {len(log_files)} log files to analyze...")
        
        for i, log_file in enumerate(log_files):
            print(f"Processing {i+1}/{len(log_files)}: {os.path.basename(log_file)}")
            
            features = self.extract_scenario_features(f"log_files/{os.path.basename(log_file)}")
            if features:
                self.scenarios[os.path.basename(log_file)] = features
        
        print(f"Successfully processed {len(self.scenarios)} scenarios")
    
    def find_behavioral_duplicates(self, similarity_threshold=0.8):
        """Find scenarios with similar behavioral sequences"""
        behavioral_groups = defaultdict(list)
        
        for scenario_name, features in self.scenarios.items():
            behavior_key = tuple(features['behavioral_sequence'])
            behavioral_groups[behavior_key].append(scenario_name)
        
        duplicates = {k: v for k, v in behavioral_groups.items() if len(v) > 1}
        
        print(f"\n=== BEHAVIORAL SEQUENCE DUPLICATES ===")
        reduction_potential = 0
        for sequence, scenarios in duplicates.items():
            print(f"Behavior: {' → '.join(sequence)}")
            print(f"  Scenarios ({len(scenarios)}): {', '.join(scenarios)}")
            print(f"  Reduction potential: {len(scenarios) - 1} scenarios")
            reduction_potential += len(scenarios) - 1
            print()
        
        print(f"Total behavioral reduction potential: {reduction_potential} scenarios")
        return duplicates
    
    def find_spatial_similarities(self, distance_threshold=100):
        """Find scenarios with similar spatial paths"""
        spatial_groups = []
        processed = set()
        
        scenarios_list = list(self.scenarios.items())
        
        for i, (name1, features1) in enumerate(scenarios_list):
            if name1 in processed:
                continue
                
            similar_group = [name1]
            path1 = features1['spatial_path']
            
            for j, (name2, features2) in enumerate(scenarios_list[i+1:], i+1):
                if name2 in processed:
                    continue
                    
                path2 = features2['spatial_path']
                similarity = self._calculate_path_similarity(path1, path2)
                
                if similarity > 0.7:  # 70% similarity threshold
                    similar_group.append(name2)
                    processed.add(name2)
            
            if len(similar_group) > 1:
                spatial_groups.append(similar_group)
            processed.add(name1)
        
        print(f"\n=== SPATIAL PATH SIMILARITIES ===")
        spatial_reduction = 0
        for group in spatial_groups:
            print(f"Similar paths ({len(group)} scenarios): {', '.join(group)}")
            spatial_reduction += len(group) - 1
        
        print(f"Total spatial reduction potential: {spatial_reduction} scenarios")
        return spatial_groups
    
    def _calculate_path_similarity(self, path1, path2):
        """Calculate similarity between two paths"""
        if not path1 or not path2:
            return 0
        
        # Simple approach: compare start and end points, and overall direction
        if len(path1) < 2 or len(path2) < 2:
            return 0
        
        # Compare start points
        start_dist = math.sqrt((path1[0][0] - path2[0][0])**2 + (path1[0][1] - path2[0][1])**2)
        
        # Compare end points
        end_dist = math.sqrt((path1[-1][0] - path2[-1][0])**2 + (path1[-1][1] - path2[-1][1])**2)
        
        # If start and end are very different, paths are different
        if start_dist > 200 or end_dist > 200:
            return 0
        
        # Calculate overall similarity (this is simplified)
        similarity = max(0, 1 - (start_dist + end_dist) / 400)
        return similarity
    
    def find_speed_profile_similarities(self):
        """Find scenarios with similar speed profiles"""
        speed_groups = []
        processed = set()
        
        scenarios_list = list(self.scenarios.items())
        
        for i, (name1, features1) in enumerate(scenarios_list):
            if name1 in processed:
                continue
            
            similar_group = [name1]
            
            for j, (name2, features2) in enumerate(scenarios_list[i+1:], i+1):
                if name2 in processed:
                    continue
                
                # Compare speed statistics
                if (abs(features1['avg_speed'] - features2['avg_speed']) < 1.0 and
                    abs(features1['max_speed'] - features2['max_speed']) < 2.0 and
                    abs(features1['speed_variance'] - features2['speed_variance']) < 1.0):
                    similar_group.append(name2)
                    processed.add(name2)
            
            if len(similar_group) > 1:
                speed_groups.append(similar_group)
            processed.add(name1)
        
        print(f"\n=== SPEED PROFILE SIMILARITIES ===")
        speed_reduction = 0
        for group in speed_groups:
            print(f"Similar speed profiles ({len(group)} scenarios): {', '.join(group)}")
            speed_reduction += len(group) - 1
        
        print(f"Total speed profile reduction potential: {speed_reduction} scenarios")
        return speed_groups
    
    def generate_reduction_report(self):
        """Generate comprehensive reduction analysis report"""
        print("="*80)
        print("SCENARIO REDUCTION ANALYSIS REPORT")
        print("="*80)
        
        print(f"Total scenarios analyzed: {len(self.scenarios)}")
        
        # Basic statistics
        if self.scenarios:
            durations = [s['duration'] for s in self.scenarios.values()]
            complexities = [s['traffic_complexity'] for s in self.scenarios.values()]
            
            print(f"\nScenario Statistics:")
            print(f"  Average duration: {np.mean(durations):.2f} seconds")
            print(f"  Duration range: {np.min(durations):.2f} - {np.max(durations):.2f} seconds")
            print(f"  Average traffic complexity: {np.mean(complexities):.1f} vehicles")
            print(f"  Complexity range: {np.min(complexities)} - {np.max(complexities)} vehicles")
        
        # Find different types of redundancies
        behavioral_duplicates = self.find_behavioral_duplicates()
        spatial_groups = self.find_spatial_similarities()
        speed_groups = self.find_speed_profile_similarities()
        
        # Calculate total reduction potential
        total_reduction = 0
        for scenarios in behavioral_duplicates.values():
            total_reduction += len(scenarios) - 1
        for group in spatial_groups:
            total_reduction += len(group) - 1
        for group in speed_groups:
            total_reduction += len(group) - 1
        
        print(f"\n=== OVERALL REDUCTION POTENTIAL ===")
        print(f"Maximum possible reduction: {total_reduction} scenarios")
        print(f"Remaining scenarios after reduction: {len(self.scenarios) - total_reduction}")
        print(f"Reduction percentage: {(total_reduction / len(self.scenarios) * 100):.1f}%")
        
        # Save detailed results
        self._save_results_to_file()
    
    def _save_results_to_file(self):
        """Save analysis results to JSON file"""
        results = {
            'total_scenarios': len(self.scenarios),
            'scenario_features': self.scenarios,
            'analysis_timestamp': str(carla.time.time()),
            'summary': {
                'total_scenarios': len(self.scenarios),
                'avg_duration': np.mean([s['duration'] for s in self.scenarios.values()]) if self.scenarios else 0,
                'avg_complexity': np.mean([s['traffic_complexity'] for s in self.scenarios.values()]) if self.scenarios else 0
            }
        }
        
        output_file = '/home/ads/ads_testing/scenario_reduction_analysis.json'
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\nDetailed results saved to: {output_file}")

def main():
    # Set environment variable
    os.environ['SCENARIO_RUNNER_ROOT'] = '/home/ads/ads_testing'
    
    # Initialize scenario reducer
    reducer = ScenarioReducer('/home/ads/ads_testing/log_files')
    
    # Analyze all scenarios
    reducer.analyze_all_scenarios()
    
    # Generate comprehensive report
    reducer.generate_reduction_report()

if __name__ == "__main__":
    main()
