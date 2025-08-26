#!/usr/bin/env python

import os
import sys
import carla
import math

# Add the scenario runner to the path
sys.path.append('/home/ads/ads_testing/scenario_runner/scenario_runner-0.9.15')

from srunner.metrics.tools.metrics_log import MetricsLog

class EgoActionSequenceExtractor:
    """Extract and display sequence of actions performed by ego vehicle"""
    
    def __init__(self):
        self.client = carla.Client('localhost', 2000)
        
    def extract_action_sequence(self, log_file, detailed=False):
        """Extract sequence of actions from log file"""
        
        try:
            # Load log data
            recorder_file = f"{os.getenv('SCENARIO_RUNNER_ROOT', './')}/{log_file}"
            recorder_str = self.client.show_recorder_file_info(recorder_file, True)
            log = MetricsLog(recorder_str)
            
            # Get ego vehicle
            ego_id = log.get_ego_vehicle_id()
            start_frame, end_frame = log.get_actor_alive_frames(ego_id)
            
            print(f"Analyzing ego vehicle (ID: {ego_id}) from frame {start_frame} to {end_frame}")
            print(f"Total duration: {log.get_elapsed_time(end_frame-1):.2f} seconds")
            print("="*80)
            
            # Initialize tracking variables
            actions = []
            prev_speed = 0
            prev_steer = 0
            prev_throttle = 0
            prev_brake = 0
            current_action = None
            action_start_time = 0
            action_start_frame = start_frame
            
            # Thresholds for action detection
            speed_threshold = 0.8  # m/s
            steer_threshold = 0.08  # steering input
            throttle_threshold = 0.1  # throttle input
            brake_threshold = 0.1  # brake input
            
            # Sample every few frames for smoother detection
            frame_step = 5
            
            for frame in range(start_frame, end_frame, frame_step):
                try:
                    # Get current state
                    timestamp = log.get_elapsed_time(frame)
                    velocity = log.get_actor_velocity(ego_id, frame)
                    speed = math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
                    
                    transform = log.get_actor_transform(ego_id, frame)
                    
                    # Get control inputs
                    try:
                        control = log.get_vehicle_control(ego_id, frame)
                        throttle = control.throttle if control else 0
                        brake = control.brake if control else 0
                        steer = control.steer if control else 0
                        hand_brake = control.hand_brake if control else False
                    except:
                        throttle = brake = steer = 0
                        hand_brake = False
                    
                    # Detect current action
                    detected_action = self._detect_action(
                        speed, prev_speed, steer, prev_steer, 
                        throttle, brake, hand_brake, detailed
                    )
                    
                    # Check if action changed
                    if detected_action != current_action:
                        # Save previous action if it existed
                        if current_action and frame > start_frame + frame_step:
                            action_duration = timestamp - action_start_time
                            actions.append({
                                'action': current_action,
                                'start_time': action_start_time,
                                'end_time': timestamp,
                                'duration': action_duration,
                                'start_frame': action_start_frame,
                                'end_frame': frame,
                                'speed_range': (prev_speed, speed) if current_action else (0, 0)
                            })
                        
                        # Start new action
                        current_action = detected_action
                        action_start_time = timestamp
                        action_start_frame = frame
                    
                    # Update previous values
                    prev_speed = speed
                    prev_steer = steer
                    prev_throttle = throttle
                    prev_brake = brake
                    
                except Exception as e:
                    if detailed:
                        print(f"Error at frame {frame}: {e}")
                    continue
            
            # Add final action
            if current_action:
                final_time = log.get_elapsed_time(end_frame-1)
                actions.append({
                    'action': current_action,
                    'start_time': action_start_time,
                    'end_time': final_time,
                    'duration': final_time - action_start_time,
                    'start_frame': action_start_frame,
                    'end_frame': end_frame,
                    'speed_range': (prev_speed, prev_speed)
                })
            
            return actions
            
        except Exception as e:
            print(f"Error processing log file: {e}")
            return []
    
    def _detect_action(self, speed, prev_speed, steer, prev_steer, throttle, brake, hand_brake, detailed=False):
        """Detect current driving action based on vehicle state"""
        
        speed_change = speed - prev_speed
        steer_magnitude = abs(steer)
        
        # Priority order for action detection
        
        # Emergency/immediate actions first
        if hand_brake:
            return "HAND_BRAKE"
        
        if brake > 0.5:
            return "HARD_BRAKE"
        
        if brake > 0.1:
            if speed < 0.5:
                return "BRAKE_TO_STOP"
            else:
                return "BRAKE"
        
        # Speed-based actions
        if speed < 0.3 and prev_speed < 0.3:
            return "STOPPED"
        
        if speed < 0.5 and speed_change < -0.3:
            return "STOPPING"
        
        # Turning actions (check steering)
        if steer_magnitude > 0.15:
            if speed > 2.0:
                direction = "LEFT" if steer > 0 else "RIGHT"
                return f"FAST_{direction}_TURN"
            elif speed > 0.5:
                direction = "LEFT" if steer > 0 else "RIGHT"
                return f"{direction}_TURN"
            else:
                direction = "LEFT" if steer > 0 else "RIGHT"
                return f"SLOW_{direction}_TURN"
        
        # Moderate steering
        if steer_magnitude > 0.05:
            direction = "LEFT" if steer > 0 else "RIGHT"
            return f"{direction}_STEER"
        
        # Speed change actions
        if speed_change > 0.5:
            return "ACCELERATE"
        
        if speed_change < -0.5:
            return "DECELERATE"
        
        # Steady state actions
        if throttle > 0.3:
            if speed > 5.0:
                return "HIGH_SPEED_CRUISE"
            elif speed > 2.0:
                return "CRUISE"
            else:
                return "SLOW_CRUISE"
        
        if speed > 0.5:
            return "COAST"
        
        return "IDLE"
    
    def display_action_sequence(self, actions, show_details=False):
        """Display the action sequence in a readable format"""
        
        if not actions:
            print("No actions detected.")
            return
        
        print(f"\nEGO VEHICLE ACTION SEQUENCE ({len(actions)} actions):")
        print("="*80)
        
        for i, action in enumerate(actions, 1):
            action_name = action['action']
            start_time = action['start_time']
            duration = action['duration']
            
            # Create arrow for sequence
            arrow = " → " if i < len(actions) else ""
            
            if show_details:
                speed_min, speed_max = action['speed_range']
                print(f"{i:2d}. {action_name:<15} "
                      f"({start_time:5.1f}s - {action['end_time']:5.1f}s, "
                      f"duration: {duration:4.1f}s, "
                      f"speed: {speed_min:4.1f}-{speed_max:4.1f} m/s){arrow}")
            else:
                print(f"{i:2d}. {action_name:<15} ({start_time:5.1f}s, {duration:4.1f}s){arrow}")
        
        # Summary statistics
        print("\n" + "="*80)
        print("ACTION SUMMARY:")
        
        # Count action types
        action_counts = {}
        total_duration = sum(a['duration'] for a in actions)
        
        for action in actions:
            action_type = action['action']
            if action_type not in action_counts:
                action_counts[action_type] = {'count': 0, 'total_time': 0}
            action_counts[action_type]['count'] += 1
            action_counts[action_type]['total_time'] += action['duration']
        
        print(f"Total duration: {total_duration:.1f} seconds")
        print(f"Number of different actions: {len(action_counts)}")
        print("\nAction breakdown:")
        
        for action_type, stats in sorted(action_counts.items()):
            percentage = (stats['total_time'] / total_duration) * 100
            print(f"  {action_type:<20}: {stats['count']:2d} times, "
                  f"{stats['total_time']:5.1f}s ({percentage:4.1f}%)")
    
    def create_simple_sequence(self, actions):
        """Create a simple sequence string"""
        if not actions:
            return "No actions detected"
        
        # Simplify action names
        simplified = []
        for action in actions:
            name = action['action']
            # Simplify names
            if 'TURN' in name:
                if 'LEFT' in name:
                    simplified.append('TURN_LEFT')
                else:
                    simplified.append('TURN_RIGHT')
            elif 'STEER' in name:
                if 'LEFT' in name:
                    simplified.append('STEER_LEFT')
                else:
                    simplified.append('STEER_RIGHT')
            elif 'BRAKE' in name:
                simplified.append('BRAKE')
            elif 'ACCELERATE' in name:
                simplified.append('ACCELERATE')
            elif 'CRUISE' in name:
                simplified.append('CRUISE')
            elif 'STOP' in name:
                simplified.append('STOP')
            else:
                simplified.append(name)
        
        return " → ".join(simplified)

def main():
    # Set environment variable
    os.environ['SCENARIO_RUNNER_ROOT'] = '/home/ads/ads_testing'
    
    # Initialize extractor
    extractor = EgoActionSequenceExtractor()
    
    # Analyze a specific log file
    log_file = "log_files/ARG_Carcarana-1_1_I-1-1.log"
    
    print(f"Extracting action sequence from: {log_file}")
    print("="*80)
    
    # Extract actions
    actions = extractor.extract_action_sequence(log_file, detailed=False)
    
    if actions:
        # Display detailed sequence
        extractor.display_action_sequence(actions, show_details=True)
        
        # Display simple sequence
        print(f"\n{'='*80}")
        print("SIMPLIFIED ACTION SEQUENCE:")
        print("="*80)
        simple_sequence = extractor.create_simple_sequence(actions)
        print(simple_sequence)
        
        # Save to file
        output_file = '/home/ads/ads_testing/ego_action_sequence.txt'
        with open(output_file, 'w') as f:
            f.write(f"Ego Vehicle Action Sequence Analysis\n")
            f.write(f"Log file: {log_file}\n")
            f.write(f"{'='*80}\n\n")
            
            f.write("Detailed sequence:\n")
            for i, action in enumerate(actions, 1):
                f.write(f"{i:2d}. {action['action']:<15} "
                       f"({action['start_time']:5.1f}s - {action['end_time']:5.1f}s, "
                       f"duration: {action['duration']:4.1f}s)\n")
            
            f.write(f"\nSimplified sequence:\n{simple_sequence}\n")
        
        print(f"\nDetailed analysis saved to: {output_file}")
    
    else:
        print("No actions could be extracted from the log file.")

if __name__ == "__main__":
    main()
