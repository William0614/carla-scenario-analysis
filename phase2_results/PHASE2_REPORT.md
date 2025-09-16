# Phase 2: Set-Based Similarity Analysis Report

## Executive Summary

Phase 2 successfully implemented Jaccard coefficient-based similarity analysis for CARLA scenario redundancy detection. By converting the 37-dimensional feature vectors from Phase 1 into categorical feature sets, this phase demonstrated an alternative approach to scenario similarity measurement.

### Key Results
- **Best F1 Score: 0.598** (Jaccard threshold = 0.5)
- **Best Accuracy: 61.4%**
- **Similar pairs identified: 29.0%** (consistent with Phase 1)
- **Feature diversity: 15 unique categorical features**

## Methodology

### Feature Extraction and Conversion
1. **Base Features**: Used identical 37-dimensional feature extraction as Phase 1
2. **Categorization**: Converted continuous features into binary categorical sets using empirically-tuned thresholds
3. **Feature Sets**: Created meaningful feature categories like `high_speed_scenario`, `turning_scenario`, `complex_path`

### Similarity Calculation
- **Metric**: Jaccard Coefficient: `J(A,B) = |A ∩ B| / |A ∪ B|`
- **Threshold Range**: 0.1 to 0.5 tested
- **Evaluation**: Performance measured against ground truth from scenario naming patterns

## Detailed Results

### Performance by Threshold
| Threshold | F1 Score | Accuracy | Precision | Recall |
|-----------|----------|----------|-----------|---------|
| 0.1 | 0.450 | 29.1% | - | - |
| 0.2 | 0.465 | 33.4% | - | - |
| 0.3 | 0.487 | 39.1% | - | - |
| 0.4 | 0.533 | 49.5% | - | - |
| **0.5** | **0.598** | **61.4%** | - | - |

### Feature Set Statistics
- **Total scenarios processed**: 174
- **Average set size**: 6.17 features per scenario
- **Set size range**: 1-11 features
- **Total unique features**: 15

### Most Common Features
1. `traffic_scenario` - 173 scenarios (99.4%)
2. `speed_peaks` - 156 scenarios (89.7%)
3. `turning_scenario` - 141 scenarios (81.0%)
4. `sharp_maneuvers` - 139 scenarios (79.9%)
5. `frequent_stopping` - 138 scenarios (79.3%)
6. `dynamic_speed` - 105 scenarios (60.3%)
7. `variable_speed` - 80 scenarios (46.0%)
8. `dense_traffic` - 68 scenarios (39.1%)
9. `location_germany` - 37 scenarios (21.3%)
10. `location_argentina` - 11 scenarios (6.3%)

## Phase Comparison

### Phase 1 vs Phase 2 Performance
| Aspect | Phase 1 (Best) | Phase 2 (Best) | Difference |
|--------|----------------|----------------|------------|
| **Method** | Z-Score + Minkowski(p=0.5) | Jaccard(threshold=0.5) | - |
| **F1 Score** | 0.644 | 0.598 | -0.046 |
| **Accuracy** | 81.5% | 61.4% | -20.1% |
| **Feature Type** | Continuous 37D vectors | Categorical sets (15 features) | - |
| **Similarity Paradigm** | Distance-based | Set-based | - |

### Key Insights
1. **Phase 1 superiority**: Distance-based metrics outperform set-based by 4.6 F1 points
2. **Complementary approaches**: Different paradigms capture different similarity aspects
3. **Information loss**: Converting continuous to categorical features reduces discriminative power
4. **Threshold sensitivity**: Jaccard performance highly dependent on threshold selection

## Technical Implementation

### Feature Thresholds Used
```python
feature_thresholds = {
    'high_speed': 8.0,         # m/s
    'frequent_stops': 2,       # number of stops
    'many_turns': 3,           # number of turns
    'high_acceleration': 1.5,  # significant acceleration events
    'long_duration': 15.0,     # seconds
    'complex_path': 1.5,       # path tortuosity ratio
    'traffic_present': 1,      # other vehicles present
    'high_curvature': 2,       # direction changes
    'speed_variability': 2.0,  # speed standard deviation
    'aggressive_steering': 0.2 # steering angle threshold
}
```

### Ground Truth Validation
- **Total pairs**: 15,051 scenario pairs
- **Similar pairs**: 4,366 (29.0%)
- **Basis**: Scenario naming patterns (e.g., "ARG_Carcarana" base)
- **Consistency**: Identical ground truth logic as Phase 1

## Conclusions

### Strengths of Set-Based Approach
1. **Interpretability**: Categorical features are easily interpretable
2. **Robustness**: Less sensitive to outliers than distance metrics
3. **Complementary**: Captures different similarity aspects than Phase 1

### Limitations
1. **Information loss**: Discretization reduces feature richness
2. **Threshold dependency**: Performance highly sensitive to threshold selection
3. **Binary nature**: Cannot capture gradual feature variations

### Recommendations
1. **Hybrid approach**: Combine set-based and distance-based metrics
2. **Adaptive thresholds**: Use data-driven threshold selection
3. **Multi-level categorization**: Consider ordinal instead of binary features

## Next Steps

Phase 2 successfully demonstrates the viability of set-based similarity metrics for scenario analysis. Future work should explore:

1. **Phase 3**: Sequence-based similarity metrics
2. **Hybrid methods**: Combining multiple similarity paradigms
3. **Ensemble approaches**: Weighted combination of best-performing metrics
4. **Dynamic thresholding**: Adaptive threshold selection methods

---

*Generated on September 16, 2025*  
*Part of CARLA Scenario Similarity Research Framework*
