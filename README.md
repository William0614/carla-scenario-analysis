# CARLA Scenario Similarity Analysis Framework

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![CARLA](https://img.shields.io/badge/CARLA-0.9.15-orange.svg)](https://carla.org/)

**Production-Ready Framework for CARLA Scenario Redundancy Detection and Multi-Dimensional Similarity Analysis**

## 🎯 Project Status: COMPLETE & PRODUCTION READY ✅

**Version**: 1.2.0  
**Last Updated**: October 14, 2025  
**Status**: Clean, Refactored, Production Ready  

---

## 🚀 Overview

A comprehensive toolkit for analyzing similarity between CARLA simulation scenarios using multi-dimensional feature extraction and various similarity metrics. This framework consolidates research from three phases of similarity analysis with both basic and advanced ground truth validation.

### 🔬 Key Capabilities

- **37-dimensional feature extraction** from CARLA scenario logs
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
│   ├── feature_extraction.py     # 37-dimensional feature extractor
│   ├── similarity_metrics.py     # All similarity metric implementations
│   ├── ground_truth.py          # Basic & multi-dimensional ground truth
│   ├── evaluation.py            # Comprehensive evaluation framework
│   └── main.py                  # CLI interface
├── examples/                     # Usage examples
│   └── example_usage.py         # Framework usage demonstrations
├── 📄 README.md                  # This file
├── 📄 FEATURE_EXTRACTION_METHODOLOGY.md  # Detailed 37D feature engineering guide
├── 📄 requirements.txt           # Python dependencies
└── 📄 [Research Documentation]   # Additional methodology and results files
```

---

## 🔧 Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
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
# Extract 37-dimensional features from log files
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

## 📊 Feature Extraction (37 Dimensions)

Our framework extracts a comprehensive 37-dimensional feature vector from each CARLA scenario log, carefully engineered to capture all essential aspects of driving scenarios for robust similarity analysis.

### Temporal Features (6D)
- **Duration**: Total scenario time length
- **Frame Count**: Recording resolution indicator  
- **Speed Changes Count**: Dynamic behavior frequency
- **Speed Change Ratio**: Normalized dynamism measure
- **Speed Variability**: Consistency vs. variance in driving
- **Significant Accelerations**: Emergency/aggressive maneuvers

### Behavioral Features (10D)  
- **Stop Events**: Traffic lights, intersections, congestion
- **Acceleration/Deceleration Events**: Merging, braking patterns
- **Turn Maneuvers**: Lane changes, navigation complexity
- **Cruise Behavior**: Highway driving, steady flow periods
- **Behavior Transitions**: Scenario complexity measure
- **Unique Behaviors**: Scenario diversity indicator
- **Steering Patterns**: Average and maximum steering intensity
- **Total Behavior Events**: Overall activity level

### Spatial Features (8D)
- **Path Length**: Total distance traveled
- **Displacement**: Net progress (start to end distance)
- **Tortuosity Ratio**: Route straightness vs. complexity
- **Bounding Box**: Width, height, and area of spatial coverage
- **Direction Changes**: Route complexity measure
- **Curvature Density**: Normalized spatial complexity

### Speed Features (10D)
- **Statistical Measures**: Min, max, mean, median, standard deviation
- **Distribution Analysis**: Interquartile range for robustness
- **Percentile Analysis**: 10th, 25th, 75th, 90th percentiles
- **Traffic Condition Indicators**: From congestion to free-flow patterns

### Traffic Features (3D)
- **Vehicle Count**: Traffic density measure
- **Traffic Complexity**: Interaction difficulty score (capped)
- **Traffic Presence**: Binary isolated vs. multi-vehicle indicator

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
- **Performance varies** significantly between ground truth methodologies

---

## � API Reference

### FeatureExtractor
Extracts 37-dimensional feature vectors from CARLA log files.

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

This framework represents the culmination of a multi-phase research project:

- **Phase 1**: Distance-based similarity metrics with normalization
- **Phase 2**: Set-based similarity analysis (Jaccard focus)  
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
- **NumPy**: Numerical computations
- **Matplotlib**: Visualization
- **SciPy**: Scientific computing (optional)

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 🤝 Acknowledgments

- CARLA Simulator team for the autonomous driving platform
- Research community for similarity analysis methodologies
- Contributors to the multi-dimensional ground truth framework

---

## 📞 Contact

For questions, issues, or contributions, please open a GitHub issue or contact the development team.

---

**🎯 Ready for Production**: This framework provides a complete, validated solution for CARLA scenario similarity analysis with clean architecture and comprehensive documentation.
