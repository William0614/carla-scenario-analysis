#!/usr/bin/env python

import os
import sys
import carla
import math
import json
import glob
from collections import defaultdict

# Add the scenario runner to the path
sys.path.append('/home/ads/ads_testing/scenario_runner/scenario_runner-0.9.15')

from srunner.metrics.tools.metrics_log import MetricsLog

def quick_scenario_analysis(log_files_dir, max_files=10):
    """Quick analysis of scenario reduction potential"""
    
    client = carla.Client('localhost', 2000)
    scenarios = {}
    
    # Get first few log files for demonstration
    log_files = glob.glob(os.path.join(log_files_dir, "*.log"))[:max_files]
    print(f"Analyzing {len(log_files)} scenarios for reduction potential...\n")
    
    for i, log_file in enumerate(log_files):
        filename = os.path.basename(log_file)
        print(f"Processing {i+1}/{len(log_files)}: {filename}")
        
        try:
            recorder_file = f"log_files/{filename}"
            recorder_str = client.show_recorder_file_info(
                f"{os.getenv('SCENARIO_RUNNER_ROOT', './')}/{recorder_file}", True
            )
            log = MetricsLog(recorder_str)
            
            # Get ego vehicle
            ego_id = log.get_ego_vehicle_id()
            start_frame, end_frame = log.get_actor_alive_frames(ego_id)
            
            # Extract behavioral sequence
            behavioral_sequence = []
            prev_speed = 0
            
            # Sample every 20th frame for speed
            for frame in range(start_frame, min(end_frame, start_frame + 200), 20):
                try:
                    velocity = log.get_actor_velocity(ego_id, frame)
                    speed = math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
                    
                    try:
                        control = log.get_vehicle_control(ego_id, frame)
                        steer = control.steer if control else 0
                        brake = control.brake if control else 0
                        throttle = control.throttle if control else 0
                    except:
                        steer = brake = throttle = 0
                    
                    # Detect behavior
                    if brake > 0.3:
                        behavior = 'brake'
                    elif speed < 0.5:
                        behavior = 'stop'
                    elif speed > prev_speed + 1.0:
                        behavior = 'accelerate'
                    elif speed < prev_speed - 1.0:
                        behavior = 'decelerate'
                    elif steer > 0.15:
                        behavior = 'left_turn'
                    elif steer < -0.15:
                        behavior = 'right_turn'
                    else:
                        behavior = 'cruise'
                    
                    # Add to sequence if different from last
                    if not behavioral_sequence or behavioral_sequence[-1] != behavior:
                        behavioral_sequence.append(behavior)
                    
                    prev_speed = speed
                    
                except:
                    continue
            
            # Get traffic complexity
            vehicle_ids = log.get_actor_ids_with_type_id("vehicle.*")
            traffic_complexity = len([vid for vid in vehicle_ids if vid != ego_id])
            
            # Store scenario features
            scenarios[filename] = {
                'behavioral_sequence': behavioral_sequence,
                'duration': log.get_elapsed_time(end_frame - 1) if end_frame > start_frame else 0,
                'traffic_complexity': traffic_complexity,
                'total_frames': end_frame - start_frame
            }
            
        except Exception as e:
            print(f"  Error: {e}")
            continue
    
    print(f"\nSuccessfully analyzed {len(scenarios)} scenarios")
    return analyze_reduction_potential(scenarios)

def analyze_reduction_potential(scenarios):
    """Analyze reduction potential across different dimensions"""
    
    print("\n" + "="*80)
    print("SCENARIO REDUCTION ANALYSIS")
    print("="*80)
    
    # 1. Behavioral Sequence Analysis
    print("\n1. BEHAVIORAL SEQUENCE PATTERNS:")
    behavioral_groups = defaultdict(list)
    
    for scenario_name, features in scenarios.items():
        behavior_key = tuple(features['behavioral_sequence'])
        behavioral_groups[behavior_key].append(scenario_name)
    
    behavioral_reduction = 0
    duplicate_behaviors = []
    
    for sequence, scenario_list in behavioral_groups.items():
        if len(scenario_list) > 1:
            print(f"   Pattern: {' → '.join(sequence)}")
            print(f"   Scenarios ({len(scenario_list)}): {', '.join(scenario_list)}")
            print(f"   Reduction potential: {len(scenario_list) - 1} scenarios")
            behavioral_reduction += len(scenario_list) - 1
            duplicate_behaviors.append((sequence, scenario_list))
            print()
    
    # 2. Duration-based Grouping
    print("2. DURATION-BASED SIMILARITIES:")
    duration_groups = defaultdict(list)
    
    for scenario_name, features in scenarios.items():
        # Group by duration buckets (±2 seconds)
        duration_bucket = round(features['duration'] / 2) * 2
        duration_groups[duration_bucket].append(scenario_name)
    
    duration_reduction = 0
    for duration, scenario_list in duration_groups.items():
        if len(scenario_list) > 1:
            print(f"   Duration ~{duration}s ({len(scenario_list)} scenarios): {', '.join(scenario_list)}")
            duration_reduction += len(scenario_list) - 1
    
    # 3. Traffic Complexity Grouping
    print("\n3. TRAFFIC COMPLEXITY SIMILARITIES:")
    complexity_groups = defaultdict(list)
    
    for scenario_name, features in scenarios.items():
        complexity_groups[features['traffic_complexity']].append(scenario_name)
    
    complexity_reduction = 0
    for complexity, scenario_list in complexity_groups.items():
        if len(scenario_list) > 1:
            print(f"   {complexity} vehicles ({len(scenario_list)} scenarios): {', '.join(scenario_list)}")
            complexity_reduction += len(scenario_list) - 1
    
    # 4. Multi-dimensional Analysis
    print("\n4. MULTI-DIMENSIONAL ANALYSIS:")
    
    # Combine behavioral + complexity
    multi_groups = defaultdict(list)
    for scenario_name, features in scenarios.items():
        key = (tuple(features['behavioral_sequence']), features['traffic_complexity'])
        multi_groups[key].append(scenario_name)
    
    multi_reduction = 0
    for (behavior, complexity), scenario_list in multi_groups.items():
        if len(scenario_list) > 1:
            print(f"   Behavior: {' → '.join(behavior)}, Traffic: {complexity} vehicles")
            print(f"   Scenarios ({len(scenario_list)}): {', '.join(scenario_list)}")
            multi_reduction += len(scenario_list) - 1
            print()
    
    # Summary
    print("5. REDUCTION SUMMARY:")
    print(f"   Total scenarios analyzed: {len(scenarios)}")
    print(f"   Behavioral duplicates: {behavioral_reduction} scenarios")
    print(f"   Duration similarities: {duration_reduction} scenarios") 
    print(f"   Traffic complexity similarities: {complexity_reduction} scenarios")
    print(f"   Multi-dimensional duplicates: {multi_reduction} scenarios")
    print(f"   Maximum reduction potential: {max(behavioral_reduction, duration_reduction, complexity_reduction, multi_reduction)} scenarios")
    print(f"   Conservative reduction estimate: {multi_reduction} scenarios")
    
    # Additional reduction areas
    print("\n6. ADDITIONAL REDUCTION OPPORTUNITIES:")
    print("   • Geographic clustering (same country/city)")
    print("   • Road type similarities (highway, urban, rural)")
    print("   • Weather/environmental conditions")
    print("   • Time of day patterns")
    print("   • Safety-critical event patterns")
    print("   • Vehicle type distributions")
    print("   • Speed profile similarities")
    print("   • Spatial path similarities")
    
    return {
        'behavioral_reduction': behavioral_reduction,
        'duration_reduction': duration_reduction,
        'complexity_reduction': complexity_reduction,
        'multi_dimensional_reduction': multi_reduction,
        'total_scenarios': len(scenarios)
    }

if __name__ == "__main__":
    # Set environment variable
    os.environ['SCENARIO_RUNNER_ROOT'] = '/home/ads/ads_testing'
    
    # Run quick analysis
    results = quick_scenario_analysis('/home/ads/ads_testing/log_files', max_files=15)
    
