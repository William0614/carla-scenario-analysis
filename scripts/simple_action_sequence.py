#!/usr/bin/env python

import os
import sys
import carla
import math

# Add the scenario runner to the path
sys.path.append('/home/ads/ads_testing/scenario_runner/scenario_runner-0.9.15')

from srunner.metrics.tools.metrics_log import MetricsLog

def extract_simple_action_sequence(log_file):
    """Extract a simple, readable action sequence from log file"""
    
    client = carla.Client('localhost', 2000)
    
    try:
        # Load log data
        recorder_file = f"{os.getenv('SCENARIO_RUNNER_ROOT', './')}/{log_file}"
        recorder_str = client.show_recorder_file_info(recorder_file, True)
        log = MetricsLog(recorder_str)
        
        # Get ego vehicle
        ego_id = log.get_ego_vehicle_id()
        start_frame, end_frame = log.get_actor_alive_frames(ego_id)
        
        actions = []
        prev_speed = 0
        prev_action = None
        
        # Sample every 10th frame for simpler sequence
        for frame in range(start_frame, end_frame, 20):
            try:
                velocity = log.get_actor_velocity(ego_id, frame)
                speed = math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
                timestamp = log.get_elapsed_time(frame)
                
                # Get control inputs
                try:
                    control = log.get_vehicle_control(ego_id, frame)
                    steer = control.steer if control else 0
                    brake = control.brake if control else 0
                    throttle = control.throttle if control else 0
                except:
                    steer = brake = throttle = 0
                
                # Simple action detection
                action = None
                if speed < 0.5:
                    action = "STOP"
                elif brake > 0.3:
                    action = "BRAKE"
                elif speed > prev_speed + 1.0:
                    action = "ACCELERATE"
                elif abs(steer) > 0.15:
                    direction = "LEFT" if steer > 0 else "RIGHT"
                    action = f"TURN_{direction}"
                elif abs(steer) > 0.05:
                    direction = "LEFT" if steer > 0 else "RIGHT"
                    action = f"STEER_{direction}"
                elif speed > 5.0:
                    action = "CRUISE_FAST"
                elif speed > 1.0:
                    action = "CRUISE"
                else:
                    action = "IDLE"
                
                # Only add if different from previous action
                if action != prev_action:
                    actions.append({
                        'action': action,
                        'time': timestamp,
                        'speed': speed,
                        'steer': steer
                    })
                    prev_action = action
                
                prev_speed = speed
                
            except:
                continue
        
        return actions
        
    except Exception as e:
        print(f"Error: {e}")
        return []

def print_action_sequence(actions, log_file):
    """Print the action sequence in a clean format"""
    
    print(f"Action Sequence for: {log_file}")
    print("=" * 60)
    
    if not actions:
        print("No actions detected")
        return
    
    # Print detailed sequence
    print("Detailed sequence:")
    for i, action in enumerate(actions, 1):
        print(f"{i:2d}. {action['action']:<12} (t={action['time']:5.1f}s, speed={action['speed']:4.1f}m/s)")
    
    # Print simple sequence
    print(f"\nSimple sequence:")
    sequence = " → ".join([a['action'] for a in actions])
    print(sequence)
    
    # Print summary
    print(f"\nSummary:")
    print(f"  Total duration: {actions[-1]['time']:.1f} seconds")
    print(f"  Number of action changes: {len(actions)}")
    
    # Count action types
    action_counts = {}
    for action in actions:
        action_type = action['action']
        action_counts[action_type] = action_counts.get(action_type, 0) + 1
    
    print(f"  Action types: {list(action_counts.keys())}")

def main():
    # Set environment variable
    os.environ['SCENARIO_RUNNER_ROOT'] = '/home/ads/ads_testing'
    
    # You can change this to any log file
    log_file = input("Enter log file name (or press Enter for default): ").strip()
    if not log_file:
        log_file = "log_files/ARG_Carcarana-1_1_I-1-1.log"
    
    # Make sure it starts with log_files/ if not absolute path
    if not log_file.startswith('/') and not log_file.startswith('log_files/'):
        log_file = f"log_files/{log_file}"
    
    print(f"\nAnalyzing: {log_file}")
    
    # Extract and display actions
    actions = extract_simple_action_sequence(log_file)
    print_action_sequence(actions, log_file)

if __name__ == "__main__":
    main()
