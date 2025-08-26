#!/usr/bin/env python

import os
import sys
import carla

# Add the scenario runner to the path
sys.path.append('/home/ads/ads_testing/scenario_runner/scenario_runner-0.9.15')

from srunner.metrics.tools.metrics_log import MetricsLog

def inspect_log(log_file):
    """Inspect what's in the log file"""
    
    # Connect to CARLA
    client = carla.Client('localhost', 2000)
    
    # Get recorder info
    recorder_file = "{}/{}".format(os.getenv('SCENARIO_RUNNER_ROOT', "./"), log_file)
    print(f"Inspecting log file: {recorder_file}")
    
    recorder_str = client.show_recorder_file_info(recorder_file, True)
    
    # Create MetricsLog object
    log = MetricsLog(recorder_str)
    
    print("\n=== LOG INSPECTION ===")
    print(f"Number of actors: {len(log._actors)}")
    
    print("\n=== ACTORS WITH ROLE NAMES ===")
    role_names = set()
    for actor_id, actor in log._actors.items():
        if "role_name" in actor:
            role_name = actor["role_name"]
            role_names.add(role_name)
            print(f"Actor {actor_id}: role_name='{role_name}', type='{actor.get('type_id', 'unknown')}'")
    
    print(f"\n=== UNIQUE ROLE NAMES ===")
    for role_name in sorted(role_names):
        print(f"- {role_name}")
    
    print(f"\n=== ALL ACTOR TYPES ===")
    type_ids = set()
    for actor_id, actor in log._actors.items():
        type_id = actor.get('type_id', 'unknown')
        type_ids.add(type_id)
    
    for type_id in sorted(type_ids):
        print(f"- {type_id}")

if __name__ == "__main__":
    inspect_log("log_files/ARG_Carcarana-1_1_I-1-1.log")
