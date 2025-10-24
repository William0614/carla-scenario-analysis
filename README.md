# CARLA Scenario Similarity Analysis Framework

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![CARLA](https://img.shields.io/badge/CARLA-0.9.15-orange.svg)](https://carla.org/)


---

## 🚀 Overview

A comprehensive toolkit for analyzing similarity between CARLA simulation scenarios using multi-dimensional feature extraction and various similarity metrics. This framework consolidates research from three phases of similarity analysis with both basic and advanced ground truth validation.

### 🔬 Key Capabilities

- **31-dimensional feature extraction** from CARLA scenario logs (temporal, motion, behavioral, spatial, context)
- **Accurate timing measurements** using CARLA's get_elapsed_time() API instead of FPS assumptions
- **Dual ground truth validation**: Basic (filename-based) + Multi-dimensional (behavioral)
- **3 categories of similarity metrics**: Distance-based, Sequence-based, Set-based  
- **Comprehensive evaluation framework** for metric validation
- **Clean CLI interface** for easy usage

---

## 📁 Project Structure

```
carla-scenario-analysis/
├── carla_similarity/              # Main framework module
│   ├── __init__.py               # Module interface
│   ├── feature_extraction.py     # 32-dimensional feature extractor
│   ├── similarity_metrics.py     # All similarity metric implementations
│   ├── ground_truth.py          # Basic & multi-dimensional ground truth
│   ├── evaluation.py            # Comprehensive evaluation framework
│   └── main.py                  # CLI interface
├── examples/                     # Usage examples
│   └── example_usage.py         # Framework usage demonstrations
├── 📄 README.md                  # This file
├── 📄 FEATURE_EXTRACTION_METHODOLOGY.md  # Detailed 32D feature engineering guide
├── 📄 requirements.txt           # Python dependencies
└── 📄 [Research Documentation]   # Additional methodology and results files
```

---

## 🔧 Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/William0614/carla-scenario-analysis.git
   cd carla-scenario-analysis
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Install CARLA Python API** (version 0.9.15):
   ```bash
   pip install carla==0.9.15
   ```

---

## 🚀 Quick Start

### CLI Usage

```bash
# Extract 32-dimensional features from log files
python carla_similarity/main.py extract-features /path/to/log/files

# Generate basic ground truth (filename-based)
python carla_similarity/main.py basic-gt /path/to/log/files

# Generate multi-dimensional ground truth (behavioral)
python carla_similarity/main.py multi-gt /path/to/log/files

# Evaluate all similarity metrics
python carla_similarity/main.py evaluate /path/to/log/files

# Run complete analysis pipeline
python carla_similarity/main.py full-analysis /path/to/log/files
```

### Programmatic Usage

```python
from carla_similarity import FeatureExtractor, SimilarityEvaluator
from carla_similarity import BasicGroundTruth, MultiDimensionalGroundTruth

# Extract features
extractor = FeatureExtractor()
features = extractor.extract_features_from_logs('log_directory/')

# Generate ground truth
basic_gt = BasicGroundTruth()
multi_gt = MultiDimensionalGroundTruth()

basic_pairs = basic_gt.generate_ground_truth(log_files)
multi_pairs = multi_gt.generate_ground_truth(features)

# Evaluate similarity metrics
evaluator = SimilarityEvaluator()
results = evaluator.evaluate_all_metrics(features, multi_pairs)
```

---

## 📊 Feature Extraction (32 Dimensions)

### Temporal Features (4D)
- **Duration**: Total scenario time length
- **Frame Count**: Recording resolution indicator  
- **Event Frequency**: Dynamic events per second (temporal rate)
- **Temporal Density**: Events per frame (recording density)

### Motion Features (8D)  
- **Speed Statistics**: Mean, standard deviation, min, max, and range
- **Acceleration Analysis**: Mean and standard deviation of acceleration values
- **Dynamic Events**: Unified threshold (2.5 m/s²) for significant speed/acceleration changes

### Behavioral Features (8D)  
- **Stop Events**: Traffic lights, intersections, congestion
- **Acceleration/Deceleration Events**: Merging, braking patterns
- **Turn Maneuvers**: Lane changes, navigation complexity
- **Cruise Behavior**: Highway driving, steady flow periods
- **Behavior Transitions**: Scenario complexity measure
- **Steering Patterns**: Average and maximum steering intensity

### Spatial Features (8D)
- **Path Length**: Total distance traveled
- **Displacement**: Net progress (start to end distance)
- **Tortuosity Ratio**: Route straightness vs. complexity
- **Bounding Box**: Width, height, and area of spatial coverage
- **Direction Changes**: Route complexity measure
- **Curvature Density**: Normalized spatial complexity

### Context Features (4D)
- **Traffic Count**: Number of other vehicles in scenario
- **Traffic Density**: Vehicles per spatial area (normalized measure)
- **Traffic Presence**: Binary isolated vs. multi-vehicle indicator  
- **Scenario Complexity**: Combined spatial and traffic complexity score

> **📖 Detailed Methodology**: For complete extraction algorithms, validation methods, and engineering rationale, see [`FEATURE_EXTRACTION_METHODOLOGY.md`](FEATURE_EXTRACTION_METHODOLOGY.md).

---

## 🏆 Research Results

### Key Findings
- **Best Distance Metric**: Cosine Similarity (F1: 0.567)
- **Best Sequence Metric**: LCS - Longest Common Subsequence (F1: 0.671)  
- **Best Set Metric**: Jaccard with N-grams (F1: 0.702)

### Ground Truth Impact
- **Multi-dimensional GT**: More realistic similarity assessment
- **Basic GT**: Simpler filename-based matching

---

## � API Reference

### FeatureExtractor
Extracts 32-dimensional feature vectors from CARLA log files.

### SimilarityMetrics
- **DistanceBasedMetrics**: Euclidean, Manhattan, Cosine, Chebyshev
- **SequenceBasedMetrics**: LCS, Edit Distance, DTW, N-gram Jaccard  
- **SetBasedMetrics**: Jaccard, Overlap coefficients

### GroundTruth
- **BasicGroundTruth**: Filename pattern-based similarity
- **MultiDimensionalGroundTruth**: 4-aspect behavioral analysis

### Evaluation
Comprehensive framework for validating similarity metric performance.

---

## 🔬 Research Background

- **Phase 1**: Distance-based similarity metrics with normalization
- **Phase 2**: Set-based similarity analysis
- **Phase 3**: Sequence-based metrics optimization
- **Integration**: Multi-dimensional ground truth validation

> **📚 Complete Research Archive**: All detailed research work, phase-specific implementations, and experimental results are preserved in the `research-archive` branch for reference and reproducibility.

---

## 📈 Performance Metrics

The framework evaluates similarity metrics using:
- **Precision**: Accuracy of positive similarity predictions
- **Recall**: Coverage of true similar pairs
- **F1-Score**: Harmonic mean of precision and recall
- **Category Analysis**: Performance by scenario type

---

## 🛠️ Development

### Testing
```bash
# Run framework tests
python -m pytest carla_similarity/tests/

# Validate with sample data
python examples/example_usage.py
```

### Contributing
1. Follow PEP 8 style guidelines
2. Add comprehensive docstrings
3. Include unit tests for new features
4. Update documentation as needed

---

## 📋 Requirements

- **Python**: 3.8+
- **CARLA**: 0.9.15 Python API
- **Scenario Runner**: MetricsLog API
- **NumPy**: Numerical computations
- **Matplotlib**: Visualization
- **SciPy**: Scientific computing (optional)

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---