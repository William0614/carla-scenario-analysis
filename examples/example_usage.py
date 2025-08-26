#!/usr/bin/env python
"""
Example usage of CARLA Scenario Analysis Tools

This script demonstrates how to use the main analysis tools
with sample data and configurations.
"""

import os
import sys

def setup_environment():
    """Set up environment variables and paths"""
    # Set scenario runner root - adjust this path for your setup
    os.environ['SCENARIO_RUNNER_ROOT'] = '/home/ads/ads_testing'
    
    # Add scenario runner to Python path
    sys.path.append('/home/ads/ads_testing/scenario_runner/scenario_runner-0.9.15')
    
    print("Environment setup complete.")
    print(f"SCENARIO_RUNNER_ROOT: {os.environ.get('SCENARIO_RUNNER_ROOT')}")

def example_action_sequence_analysis():
    """Example: Extract action sequence from a log file"""
    print("\n" + "="*60)
    print("EXAMPLE: Action Sequence Analysis")
    print("="*60)
    
    # Import the action sequence extractor
    sys.path.append('./scripts')
    from simple_action_sequence import extract_simple_action_sequence, print_action_sequence
    
    # Example log file (adjust path as needed)
    log_file = "log_files/ARG_Carcarana-1_1_I-1-1.log"
    
    print(f"Analyzing: {log_file}")
    
    # Extract actions
    actions = extract_simple_action_sequence(log_file)
    
    if actions:
        print_action_sequence(actions, log_file)
    else:
        print("No actions could be extracted. Check that:")
        print("1. CARLA is running")
        print("2. Log file exists")
        print("3. SCENARIO_RUNNER_ROOT is set correctly")

def example_movement_analysis():
    """Example: Analyze vehicle movement patterns"""
    print("\n" + "="*60)
    print("EXAMPLE: Movement Analysis")
    print("="*60)
    
    # This would typically import and run detailed_movement_analysis
    print("To run movement analysis:")
    print("python scripts/detailed_movement_analysis.py")
    print("- Outputs movement statistics")
    print("- Creates visualization plots")
    print("- Analyzes speed, acceleration, steering patterns")

def example_scenario_reduction():
    """Example: Scenario reduction analysis"""
    print("\n" + "="*60)
    print("EXAMPLE: Scenario Reduction")
    print("="*60)
    
    print("To run scenario reduction analysis:")
    print("python scripts/quick_scenario_analysis.py")
    print("- Analyzes multiple log files")
    print("- Identifies behavioral duplicates")
    print("- Estimates reduction potential")
    print("- Groups similar scenarios")

def main():
    """Main example function"""
    print("CARLA Scenario Analysis Tools - Example Usage")
    print("=" * 60)
    
    # Setup environment
    setup_environment()
    
    # Run examples
    try:
        example_action_sequence_analysis()
    except Exception as e:
        print(f"Action sequence analysis failed: {e}")
        print("Make sure CARLA is running and log files are available")
    
    example_movement_analysis()
    example_scenario_reduction()
    
    print("\n" + "="*60)
    print("Example completed. Check individual scripts for more options.")
    print("="*60)

if __name__ == "__main__":
    main()
