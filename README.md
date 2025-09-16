# CARLA Scenario Analysis Tools

A comprehensive research toolkit for analyzing CARLA simulation scenarios using advanced similarity metrics to identify redundancies and optimize testing databases.

## 🎯 Project Overview

This repository implements a multi-phase research program for CARLA scenario similarity analysis, focusing on:
- **Phase 1**: Distance-based similarity metrics with normalization methods ✅ **COMPLETED**
- **Phase 2**: Set-based similarity metrics (Jaccard coefficient only) ✅ **COMPLETED**
- **Phase 3**: Sequence-based metrics (N-gram Jaccard, DTW, Edit Distance, LCS) ✅ **COMPLETED**
- **Phase 4**: Machine learning and ensemble approaches 🔄 **PLANNED**

## 🏆 Results Summary

### Phase 3: Sequence-Based Similarity ✅ **NEW**
**Best Performance: N-gram Jaccard Similarity (threshold=0.3)**
- **F1-Score**: 0.702 🏆 **BEST OVERALL**
- **Accuracy**: 81.2%
- **Approach**: Action sequence analysis with 2-gram behavioral patterns
- **Sequences**: 169 scenarios, avg 21.8 actions each
- **Action Types**: 10 driving behaviors (CRUISE_FAST, BRAKE, STEER_*, etc.)
- **Key Innovation**: Behavioral transition patterns outperform individual action analysis

### Phase 2: Set-Based Jaccard Similarity ✅
**Best Performance: Jaccard Threshold = 0.5**
- **F1-Score**: 0.598
- **Accuracy**: 61.4%
- **Approach**: Jaccard coefficient only (no Dice or other set metrics)
- **Feature Sets**: 15 categorical features, avg 6.17 per scenario
- **Similarity Rate**: 29.0% (consistent with Phase 1)
- **Key Features**: traffic_scenario (99.4%), speed_peaks (89.7%), turning_scenario (81.0%)

### Phase 1: Distance-Based Metrics ✅
**Best Performance: Z-Score + Minkowski (p=0.5)**
- **F1-Score**: 0.644
- **Accuracy**: 81.5%
- **Dataset**: 174 CARLA scenarios (15,051 pairs)
- **Redundancy Rate**: 29% similarity in ground truth

### Cross-Phase Performance Comparison
| Phase | Approach | F1-Score | Accuracy | Key Strength |
|-------|----------|----------|----------|--------------|
| **3** | **Sequence-based** | **🏆 0.702** | **81.2%** | **Behavioral pattern recognition** |
| 1 | Distance-based | 0.644 | 81.5% | High precision, nuanced similarities |
| 2 | Set-based | 0.598 | 61.4% | Interpretable features, high recall |

## 📁 Repository Structure

```
carla-scenario-analysis/
├── phase1_results/           # Phase 1 distance-based analysis
│   ├── METHODOLOGY_AND_RESULTS_REPORT.md    # Comprehensive methodology
│   ├── PHASE1_COMPLETE_REPORT.md            # Executive summary
│   ├── phase1_distance_metrics_research.py   # Main experiment script
│   ├── phase1_distance_metrics_results.json  # Complete results data
│   ├── visualize_phase1_results.py          # Analysis and visualization
│   └── *.png                                # Performance visualizations
├── phase2_results/           # Phase 2 set-based analysis
│   ├── phase2_executive_summary.md          # Phase 2 executive summary
│   ├── phase2_detailed_report.md            # Comprehensive technical report
│   ├── phase2_changelog.md                  # Updated project changelog
│   ├── phase2_set_based_jaccard_research_fixed.py  # Main Phase 2 script
│   ├── phase2_jaccard_results_*.json        # Results data
│   └── *.png                                # Phase 2 visualizations
├── phase3_results/           # Phase 3 sequence-based analysis ✨ NEW
│   ├── PHASE3_COMPLETE_REPORT.md            # Comprehensive Phase 3 report
│   ├── phase3_executive_summary.md          # Executive summary
│   ├── phase3_sequence_based_research.py    # Main Phase 3 script
│   ├── phase3_sequence_results_*.json       # Results data
│   └── phase3_sequence_analysis_*.png       # Performance visualizations
├── scripts/                  # Legacy analysis tools
├── docs/                    # Documentation
└── examples/               # Usage examples
```

## 🔬 Research Methodology

### Phase 3: Sequence-Based Metrics ✨ **NEW**
- **Action Extraction**: Frame-by-frame vehicle control analysis from CARLA logs
- **Action Types**: 10 driving behaviors (STOP, CRUISE_FAST, BRAKE, STEER_*, TURN_*, etc.)
- **Sequence Processing**: Consecutive duplicate removal, avg length 21.8 actions
- **Metrics Tested**: Edit Distance, LCS, DTW, N-gram Jaccard, Global Alignment
- **Key Innovation**: N-gram (2-gram) behavioral pattern analysis
- **Success Factor**: Action transitions capture driving behavior better than individual actions

### Phase 2: Set-Based Metrics
- **Primary Metric**: Jaccard coefficient only (|A ∩ B| / |A ∪ B|)
- **Feature Conversion**: 37D continuous → 15 categorical feature sets
- **Threshold Optimization**: 5 similarity thresholds tested (0.1-0.5)
- **Key Innovation**: Domain-specific thresholds for automotive scenarios
- **Evaluation**: Same ground truth as Phase 1 for direct comparison

### Phase 1: Distance-Based Metrics
- **Metrics Tested**: Euclidean, Manhattan, Cosine, Minkowski (p=3, p=0.5)
- **Normalization**: None, Min-Max, Z-Score, Robust scaling
- **Feature Engineering**: 37-dimensional vectors (temporal, spatial, behavioral, speed, traffic)
- **Evaluation**: F1-score, Accuracy, Precision, Recall, Correlation analysis

### Complete 37-Dimensional Feature Vector

#### Temporal Features (6 dimensions)
1. Duration (seconds)
2. Frame count
3. Speed changes count
4. Speed change ratio
5. Speed variability (std)
6. Significant accelerations count

#### Behavioral Features (10 dimensions)
7. Stop events count
8. Acceleration events count
9. Deceleration events count
10. Turn maneuvers count
11. Cruise behavior count
12. Behavior transitions count
13. Unique behaviors count
14. Average steering magnitude
15. Maximum steering magnitude
16. Total behavior events count

#### Spatial Features (8 dimensions)
17. Total path length
18. Total displacement
19. Path tortuosity ratio
20. Bounding box width
21. Bounding box height
22. Bounding box area
23. Direction changes count
24. Curvature density

#### Speed Features (10 dimensions)
25. Mean speed
26. Speed standard deviation
27. Minimum speed
28. Maximum speed
29. Median speed
30. Speed interquartile range
31. Mean acceleration
32. Acceleration standard deviation
33. High speed events count
34. Hard acceleration/deceleration count

#### Traffic Features (3 dimensions)
35. Traffic vehicle count
36. Capped traffic complexity
37. Traffic presence indicator

### Feature Categories Summary
- **Temporal Features** (6 dims): Duration, velocity patterns, acceleration events
- **Behavioral Features** (10 dims): Actions, maneuvers, steering characteristics
- **Spatial Features** (8 dims): Path geometry, displacement, curvature metrics
- **Speed Features** (10 dims): Speed statistics, acceleration patterns, threshold events
- **Traffic Features** (3 dims): Vehicle interactions, traffic density indicators

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
