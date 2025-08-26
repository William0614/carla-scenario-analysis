# CARLA Scenario Analysis Tools

A comprehensive toolkit for analyzing CARLA simulation log files to extract vehicle behaviors, perform scenario reduction, and understand driving patterns.

## Overview

This repository contains Python scripts for analyzing CARLA recorder log files to:
- Extract ego vehicle action sequences
- Analyze vehicle movement patterns
- Identify scenario redundancies for test suite reduction
- Visualize driving behaviors and trajectories

## Features

### 🚗 Action Sequence Analysis
- Extract detailed sequences of driving actions (accelerate, turn, brake, etc.)
- Timeline-based action analysis with precise timing
- Behavioral pattern recognition

### 📊 Movement Analysis
- Vehicle speed, acceleration, and control input analysis
- Spatial path analysis and trajectory visualization
- Angular velocity and steering pattern analysis

### 🔄 Scenario Reduction
- Identify redundant scenarios based on behavioral patterns
- Multi-dimensional similarity analysis
- Quantify reduction potential across test suites

### 📋 Log File Inspection
- Display recorder file information and vehicle attributes
- Extract vehicle metadata and simulation parameters

## Scripts

### Core Analysis Tools

#### `ego_action_sequence.py`
Extracts detailed action sequences performed by the ego vehicle.

**Features:**
- 65+ detailed action types (ACCELERATE, TURN_LEFT, BRAKE, etc.)
- Precise timing and duration for each action
- Statistical breakdown and action frequency analysis
- Saves results to file

**Usage:**
```bash
python scripts/ego_action_sequence.py
```

#### `simple_action_sequence.py`
Interactive tool for extracting simplified action sequences from any log file.

**Features:**
- Clean, readable action sequences
- Interactive log file selection
- Summary statistics
- 19 main action types

**Usage:**
```bash
python scripts/simple_action_sequence.py
# Enter log file name when prompted
```

#### `detailed_movement_analysis.py`
Comprehensive movement analysis with visualization.

**Features:**
- Speed, acceleration, and control input analysis
- Statistical summaries (max speed, turning behavior, etc.)
- Visualization plots saved as PNG files
- Driving behavior classification

**Usage:**
```bash
python scripts/detailed_movement_analysis.py
```

### Scenario Reduction Tools

#### `scenario_reducer.py`
Advanced scenario reduction analysis across multiple dimensions.

**Features:**
- Behavioral sequence pattern matching
- Spatial path similarity analysis
- Traffic complexity grouping
- Multi-dimensional reduction potential calculation

**Usage:**
```bash
python scripts/scenario_reducer.py
```

#### `quick_scenario_analysis.py`
Fast analysis tool for identifying reduction opportunities.

**Features:**
- Processes multiple log files quickly
- Behavioral, duration, and complexity clustering
- Reduction potential estimation
- Batch processing capabilities

**Usage:**
```bash
python scripts/quick_scenario_analysis.py
```

### Utility Tools

#### `inspect_log.py`
Detailed log file inspection with actor and scenario information.

**Features:**
- Actor role names and IDs
- Vehicle types and attributes
- Simulation timeline and duration

#### `show_log_info.py`
Simple tool to display basic recorder file information.

**Usage:**
```bash
python scripts/show_log_info.py
```

## Requirements

### Dependencies
- Python 3.7+
- CARLA Python API (0.9.15 or compatible)
- matplotlib
- numpy
- scenario_runner (for MetricsLog functionality)

### CARLA Setup
1. CARLA simulator must be running on localhost:2000
2. Scenario runner must be installed and accessible
3. Set `SCENARIO_RUNNER_ROOT` environment variable:
   ```bash
   export SCENARIO_RUNNER_ROOT=/path/to/your/carla/data
   ```

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/William0614/carla-scenario-analysis.git
   cd carla-scenario-analysis
   ```

2. Install CARLA and scenario_runner (follow official documentation)

3. Set up environment variables:
   ```bash
   export SCENARIO_RUNNER_ROOT=/path/to/your/carla/data
   ```

4. Install Python dependencies:
   ```bash
   pip install matplotlib numpy
   ```

## Usage Examples

### Analyze Action Sequence
```bash
# Extract detailed action sequence from a log file
python scripts/ego_action_sequence.py

# Get simple action sequence interactively
python scripts/simple_action_sequence.py
# Enter: log_files/your_scenario.log
```

### Movement Analysis
```bash
# Analyze vehicle movement patterns
python scripts/detailed_movement_analysis.py
# Outputs: movement_analysis_plot.png
```

### Scenario Reduction
```bash
# Find redundant scenarios across your test suite
python scripts/scenario_reducer.py
# Outputs: scenario_reduction_analysis.json

# Quick reduction analysis
python scripts/quick_scenario_analysis.py
```

### Log Inspection
```bash
# Inspect log file contents
python scripts/inspect_log.py

# Show basic recorder info
python scripts/show_log_info.py
```

## Output Examples

### Action Sequence Output
```
EGO VEHICLE ACTION SEQUENCE (19 actions):
1. STOP         (t=  0.0s, speed= 0.0m/s)
2. ACCELERATE   (t=  0.6s, speed= 1.3m/s)
3. CRUISE       (t=  1.2s, speed= 4.9m/s)
4. TURN_RIGHT   (t=  3.2s, speed= 8.2m/s)
5. CRUISE_FAST  (t=  6.6s, speed= 8.2m/s)
...

Simple sequence:
STOP → ACCELERATE → CRUISE → TURN_RIGHT → CRUISE_FAST → TURN_LEFT
```

### Movement Statistics
```
=== MOVEMENT STATISTICS ===
Max speed: 8.45 m/s (30.42 km/h)
Average speed: 6.30 m/s
Max throttle: 0.75
Right turn frames: 190 (38.0%)
Heavy braking events: 0
```

### Scenario Reduction Results
```
=== BEHAVIORAL SEQUENCE DUPLICATES ===
Behavior: accelerate → right_turn → cruise
  Scenarios (3): scenario_A.log, scenario_B.log, scenario_C.log
  Reduction potential: 2 scenarios

Total reduction potential: 45 scenarios (26% of test suite)
```

## Research Applications

This toolkit is designed for:
- **Autonomous Vehicle Testing**: Optimize test scenarios by identifying redundancies
- **Behavioral Analysis**: Understand driving patterns and vehicle interactions
- **Test Suite Reduction**: Maintain coverage while reducing computational costs
- **Scenario Validation**: Verify that test scenarios cover diverse driving behaviors

## Scenario Reduction Methodology

The tools implement multi-dimensional scenario reduction:

1. **Behavioral Sequences**: Identify scenarios with identical action patterns
2. **Spatial Similarity**: Group scenarios with similar paths and geometries
3. **Traffic Complexity**: Cluster by number and types of vehicles
4. **Temporal Patterns**: Group by duration and timing characteristics
5. **Multi-dimensional Analysis**: Combine criteria for maximum reduction

## Contributing

Contributions are welcome! Please feel free to submit pull requests, report bugs, or suggest new features.

## License

This project is open source. Please check individual script headers for specific license information.

## Acknowledgments

- Built on top of the CARLA simulator and scenario_runner framework
- Utilizes the CARLA Python API for log file analysis
- Inspired by autonomous vehicle testing and validation research

---

For questions or support, please open an issue in this repository.
