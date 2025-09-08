# CARLA Scenario Analysis Tools

A comprehensive research toolkit for analyzing CARLA simulation scenarios using advanced similarity metrics to identify redundancies and optimize testing databases.

## 🎯 Project Overview

This repository implements a multi-phase research program for CARLA scenario similarity analysis, focusing on:
- **Phase 1**: Distance-based similarity metrics with normalization methods ✅ **COMPLETED**
- **Phase 2**: Set-based similarity metrics (Jaccard, Dice, Overlap) 🔄 **PLANNED**
- **Phase 3**: Sequence-based metrics (DTW, Edit Distance, LCS) 🔄 **PLANNED**
- **Phase 4**: Machine learning and ensemble approaches 🔄 **PLANNED**

## 🏆 Phase 1 Results Summary

**Best Performing Combination: Z-Score + Minkowski (p=0.5)**
- **F1-Score**: 0.644
- **Accuracy**: 81.5%
- **Dataset**: 174 CARLA scenarios (15,051 pairs)
- **Redundancy Rate**: 29% similarity in ground truth

### Key Findings
- Z-Score normalization provides optimal balanced performance
- Minkowski distance with p=0.5 outperforms traditional metrics
- 37-dimensional feature representation effectively captures scenario characteristics
- Threshold of 0.8 optimal for practical applications

## 📁 Repository Structure

```
carla-scenario-analysis/
├── phase1_results/           # Phase 1 complete analysis and results
│   ├── METHODOLOGY_AND_RESULTS_REPORT.md    # Comprehensive methodology
│   ├── PHASE1_COMPLETE_REPORT.md            # Executive summary
│   ├── phase1_distance_metrics_research.py   # Main experiment script
│   ├── phase1_distance_metrics_results.json  # Complete results data
│   ├── visualize_phase1_results.py          # Analysis and visualization
│   └── *.png                                # Performance visualizations
├── scripts/                  # Legacy analysis tools
├── docs/                    # Documentation
└── examples/               # Usage examples
```

## 🔬 Research Methodology

### Phase 1: Distance-Based Metrics
- **Metrics Tested**: Euclidean, Manhattan, Cosine, Minkowski (p=3, p=0.5)
- **Normalization**: None, Min-Max, Z-Score, Robust scaling
- **Feature Engineering**: 37-dimensional vectors (temporal, spatial, behavioral)
- **Evaluation**: F1-score, Accuracy, Precision, Recall, Correlation analysis

### Feature Categories
1. **Temporal Features** (13 dims): Duration, velocity, acceleration patterns
2. **Spatial Features** (15 dims): Positions, distances, trajectory characteristics  
3. **Behavioral Features** (9 dims): Actions, interactions, route completion

## 📊 Quick Start - Phase 1 Analysis

### Prerequisites
```bash
pip install numpy scipy scikit-learn matplotlib seaborn pandas

```

### Run Phase 1 Analysis
```bash
# Navigate to phase1_results directory
cd phase1_results/

# Run the complete distance metrics analysis
python phase1_distance_metrics_research.py --full

# Generate comprehensive visualizations and summary
python visualize_phase1_results.py
```

### View Results
- **Methodology Report**: `METHODOLOGY_AND_RESULTS_REPORT.md`
- **Executive Summary**: `PHASE1_COMPLETE_REPORT.md`
- **Raw Results**: `phase1_distance_metrics_results.json`
- **Visualizations**: `phase1_heatmap_*.png`, `phase1_*_comparison.png`

## 📈 Performance Benchmarks

| Metric Combination | F1-Score | Accuracy | Precision | Recall |
|-------------------|----------|----------|-----------|--------|
| Z-Score + Minkowski (p=0.5) | **0.644** | **81.5%** | 72.9% | 57.7% |
| Z-Score + Manhattan | 0.618 | 71.3% | 50.3% | 80.1% |
| Z-Score + Cosine | 0.596 | 73.1% | 52.8% | 68.5% |
| Z-Score + Euclidean | 0.590 | 62.1% | 43.0% | 93.9% |

## 🔬 Research Applications

### Autonomous Vehicle Testing
- **Scenario Database Optimization**: Remove 29% redundant scenarios
- **Testing Efficiency**: Focus resources on unique scenarios
- **Coverage Analysis**: Ensure comprehensive test coverage
- **Quality Assurance**: Validate scenario generation systems

### Academic Research
- **Similarity Analysis**: Novel distance metrics for high-dimensional data
- **Normalization Studies**: Empirical validation of preprocessing methods
- **Feature Engineering**: Domain-specific scenario characterization
- **Benchmarking**: Standardized evaluation framework

## 🚀 Next Research Phases

### Phase 2: Set-Based Metrics (In Development)
- Jaccard Index for behavioral overlap analysis
- Dice Coefficient for scenario intersection
- Overlap Coefficient for asymmetric similarities

### Phase 3: Sequence-Based Metrics (Planned)
- Dynamic Time Warping for temporal pattern matching
- Edit Distance for action sequence comparison
- Longest Common Subsequence for behavioral patterns

### Phase 4: Advanced Methods (Planned)
- Machine learning-based similarity learning
- Ensemble methods combining multiple metrics
- Deep learning embeddings for scenario representation

## 📋 Legacy Tools

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
```json
{
  "phase1_best_result": {
    "metric": "z_score_minkowski_p0.5",
    "f1_score": 0.644,
    "accuracy": 0.815,
    "threshold": 0.8,
    "redundancy_detected": "29% of scenario pairs"
  }
}
```

## 🎓 Research Applications

### Autonomous Vehicle Industry
- **Test Optimization**: Reduce testing time by 29% through redundancy elimination
- **Quality Assurance**: Automated validation of scenario databases
- **Coverage Analysis**: Ensure comprehensive testing with minimal redundancy
- **Synthetic Data Validation**: Verify generated scenarios against real-world patterns

### Academic Research
- **Similarity Analysis**: Novel distance metrics for high-dimensional scenario data
- **Normalization Studies**: Empirical validation of preprocessing techniques
- **Feature Engineering**: Domain-specific characterization of driving scenarios
- **Benchmarking**: Standardized evaluation framework for scenario analysis

## 🔬 Scientific Contributions

### Methodological Innovations
1. **Sub-linear Distance Metrics**: Demonstrated superiority of Minkowski (p=0.5)
2. **Normalization Analysis**: Comprehensive evaluation of preprocessing methods
3. **Feature Engineering**: 37-dimensional scenario characterization framework
4. **Evaluation Framework**: Multi-metric assessment for practical applications

### Empirical Findings
- Z-Score normalization consistently outperforms other methods
- Sub-linear distance metrics capture non-linear scenario relationships
- 37-dimensional feature space provides adequate discriminative power
- Threshold optimization crucial for practical deployment

## 📊 Reproducibility

### Data Requirements
- **Minimum Dataset**: 50+ scenarios for reliable results
- **Recommended**: 100+ scenarios for statistical significance
- **Current Study**: 174 scenarios from diverse geographic locations
- **Feature Format**: 37-dimensional numerical vectors

### Computational Requirements
- **Memory**: ~100MB for 174 scenarios
- **Processing Time**: ~30 minutes for full Phase 1 analysis
- **Dependencies**: Standard scientific Python stack
- **Platform**: Linux/Windows/macOS compatible

## 🤝 Contributing

We welcome contributions in several areas:

### Code Contributions
- Phase 2/3/4 metric implementations
- Performance optimizations
- Additional visualization tools
- Bug fixes and improvements

### Research Contributions
- Novel similarity metrics
- Feature engineering improvements
- Evaluation methodology enhancements
- Domain-specific applications

### Data Contributions
- Additional CARLA scenario datasets
- Ground truth annotations
- Cross-validation datasets
- Real-world scenario comparisons

## 📜 Citation

If you use this work in your research, please cite:

```bibtex
@misc{carla_scenario_analysis_2025,
  title={CARLA Scenario Similarity Analysis: Distance-Based Metrics for Autonomous Vehicle Testing},
  author={[Your Name]},
  year={2025},
  publisher={GitHub},
  url={https://github.com/William0614/carla-scenario-analysis}
}
```

## 📄 License

This project is released under the MIT License. See individual files for specific licensing information.

## 🙏 Acknowledgments

- **CARLA Team**: For the excellent autonomous driving simulator
- **Scenario Runner**: For the testing framework and metrics infrastructure
- **Scientific Python Community**: NumPy, SciPy, Scikit-learn, Matplotlib, Seaborn
- **Research Community**: For methodological foundations in similarity analysis

---

**🔬 Research Status**: Phase 1 Complete | Phase 2 In Development  
**📧 Contact**: Open an issue for questions, suggestions, or collaboration opportunities  
**🌐 Repository**: https://github.com/William0614/carla-scenario-analysis
