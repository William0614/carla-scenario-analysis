#!/usr/bin/env python

import os
import sys
import carla
import math
import matplotlib.pyplot as plt

# Add the scenario runner to the path
sys.path.append('/home/ads/ads_testing/scenario_runner/scenario_runner-0.9.15')

from srunner.metrics.tools.metrics_log import MetricsLog

def analyze_detailed_movement(log_file, actor_id=None):
    """Analyze detailed movement data for a vehicle using MetricsLog"""
    
    # Connect to CARLA and get log data
    client = carla.Client('localhost', 2000)
    recorder_file = "{}/{}".format(os.getenv('SCENARIO_RUNNER_ROOT', "./"), log_file)
    recorder_str = client.show_recorder_file_info(recorder_file, True)
    log = MetricsLog(recorder_str)
    
    # Get ego vehicle if no actor specified
    if actor_id is None:
        actor_id = log.get_ego_vehicle_id()
    
    print(f"Analyzing movement for Actor ID: {actor_id}")
    
    # Get the frames the actor was alive
    start_frame, end_frame = log.get_actor_alive_frames(actor_id)
    print(f"Actor alive from frame {start_frame} to {end_frame}")
    
    # Initialize data lists
    frames = []
    timestamps = []
    speeds = []
    throttles = []
    brakes = []
    steers = []
    accelerations = []
    
    print("Extracting movement data...")
    
    for frame in range(start_frame, min(end_frame, start_frame + 500)):  # Limit to 500 frames
        try:
            # Time data
            timestamp = log.get_elapsed_time(frame)
            
            # Velocity and speed
            velocity = log.get_actor_velocity(actor_id, frame)
            speed = math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
            
            # Acceleration
            acceleration = log.get_actor_acceleration(actor_id, frame)
            accel_magnitude = math.sqrt(acceleration.x**2 + acceleration.y**2)
            
            # Vehicle control data (steering, throttle, brake)
            try:
                control = log.get_vehicle_control(actor_id, frame)
                throttle = control.throttle if control else 0
                brake = control.brake if control else 0
                steer = control.steer if control else 0
            except:
                throttle = brake = steer = 0
            
            # Store data
            frames.append(frame)
            timestamps.append(timestamp)
            speeds.append(speed)
            throttles.append(throttle)
            brakes.append(brake)
            steers.append(steer)
            accelerations.append(accel_magnitude)
            
        except Exception as e:
            print(f"Error at frame {frame}: {e}")
            continue
    
    print(f"Extracted data for {len(frames)} frames")
    
    # Print movement statistics
    if speeds:
        print(f"\n=== MOVEMENT STATISTICS ===")
        print(f"Max speed: {max(speeds):.2f} m/s ({max(speeds)*3.6:.2f} km/h)")
        print(f"Average speed: {sum(speeds)/len(speeds):.2f} m/s")
        print(f"Max throttle: {max(throttles):.2f}")
        print(f"Max brake: {max(brakes):.2f}")
        print(f"Max steering: {max([abs(s) for s in steers]):.2f}")
        
        # Analyze turning behavior
        left_turns = [s for s in steers if s > 0.1]
        right_turns = [s for s in steers if s < -0.1]
        print(f"Left turn frames: {len(left_turns)} ({len(left_turns)/len(frames)*100:.1f}%)")
        print(f"Right turn frames: {len(right_turns)} ({len(right_turns)/len(frames)*100:.1f}%)")
        
        # Analyze acceleration/braking
        print(f"Max acceleration: {max(accelerations):.2f} m/s²")
        heavy_braking = [b for b in brakes if b > 0.5]
        print(f"Heavy braking events: {len(heavy_braking)}")
        
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle(f'Vehicle Movement Analysis - Actor {actor_id}')
        
        # Speed over time
        axes[0,0].plot(timestamps, speeds, 'b-', linewidth=1)
        axes[0,0].set_title('Speed over Time')
        axes[0,0].set_xlabel('Time (s)')
        axes[0,0].set_ylabel('Speed (m/s)')
        axes[0,0].grid(True, alpha=0.3)
        
        # Control inputs
        axes[0,1].plot(timestamps, throttles, 'g-', label='Throttle', alpha=0.8)
        axes[0,1].plot(timestamps, brakes, 'r-', label='Brake', alpha=0.8)
        axes[0,1].set_title('Throttle and Brake Inputs')
        axes[0,1].set_xlabel('Time (s)')
        axes[0,1].set_ylabel('Input Value (0-1)')
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)
        
        # Steering
        axes[1,0].plot(timestamps, steers, 'purple', linewidth=1)
        axes[1,0].set_title('Steering Input')
        axes[1,0].set_xlabel('Time (s)')
        axes[1,0].set_ylabel('Steer Value (-1 to 1)')
        axes[1,0].axhline(y=0, color='k', linestyle='--', alpha=0.5)
        axes[1,0].grid(True, alpha=0.3)
        
        # Acceleration
        axes[1,1].plot(timestamps, accelerations, 'orange', linewidth=1)
        axes[1,1].set_title('Acceleration Magnitude')
        axes[1,1].set_xlabel('Time (s)')
        axes[1,1].set_ylabel('Acceleration (m/s²)')
        axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_file = f'/home/ads/ads_testing/detailed_movement_analysis_actor_{actor_id}.png'
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"\nDetailed movement analysis plot saved to: {output_file}")
        plt.close()
    
    return {
        'frames': frames,
        'speeds': speeds,
        'throttles': throttles,
        'brakes': brakes,
        'steers': steers,
        'accelerations': accelerations
    }

if __name__ == "__main__":
    # Set the environment variable
    os.environ['SCENARIO_RUNNER_ROOT'] = '/home/ads/ads_testing'
    
    # Analyze the ego vehicle
    data = analyze_detailed_movement("log_files/ARG_Carcarana-1_1_I-1-1.log")
    
    print("\n=== Available MetricsLog Methods for Movement Analysis ===")
    print("• get_actor_transform() - Position and rotation")
    print("• get_actor_velocity() - Linear velocity")
    print("• get_actor_acceleration() - Linear acceleration")
    print("• get_actor_angular_velocity() - Rotational velocity")
    print("• get_vehicle_control() - Steering, throttle, brake")
    print("• get_vehicle_physics_control() - Physics parameters")
    print("• get_actor_collisions() - Collision events")
    print("• get_vehicle_lights() - Light states")
