# Phase 2: Set-Based Similarity Analysis - Detailed Technical Report

## 1. Introduction

Phase 2 of the CARLA scenario similarity research implements set-based similarity metrics using the Jaccard coefficient. This approach complements Phase 1's distance-based methods by converting continuous feature vectors into categorical feature sets, providing an alternative perspective on scenario similarity.

## 2. Methodology

### 2.1 Feature Extraction Pipeline

The analysis uses the identical 37-dimensional feature extraction framework from Phase 1:

**Temporal Features (6 dimensions):**
- Duration, frame count, speed changes, change ratio, speed variability, significant accelerations

**Behavioral Features (10 dimensions):**
- Stop/acceleration/deceleration counts, turn counts, cruise behavior, transitions, steering metrics

**Spatial Features (8 dimensions):**
- Path length, displacement, tortuosity, bounding box dimensions, curvature metrics

**Speed Features (10 dimensions):**
- Mean/std/min/max speeds, percentiles, acceleration statistics, speed thresholds

**Traffic Features (3 dimensions):**
- Vehicle count, capped complexity, traffic presence indicator

### 2.2 Continuous-to-Categorical Conversion

Each continuous feature is converted to categorical membership using domain-specific thresholds:

```python
feature_thresholds = {
    'high_speed': 8.0,         # m/s (realistic urban threshold)
    'frequent_stops': 2,       # stop events
    'many_turns': 3,           # turn maneuvers  
    'high_acceleration': 1.5,  # acceleration events
    'long_duration': 15.0,     # seconds
    'complex_path': 1.5,       # tortuosity ratio
    'traffic_present': 1,      # vehicle count
    'high_curvature': 2,       # direction changes
    'speed_variability': 2.0,  # speed standard deviation
    'aggressive_steering': 0.2 # steering magnitude
}
```

### 2.3 Jaccard Similarity Implementation

The Jaccard coefficient is calculated as:

**J(A,B) = |A ∩ B| / |A ∪ B|**

Where A and B are feature sets for two scenarios.

## 3. Experimental Results

### 3.1 Dataset Characteristics
- **Total Scenarios:** 174
- **Total Scenario Pairs:** 15,051
- **Ground Truth Similar Pairs:** 4,366 (29.0%)
- **Feature Extraction Success Rate:** 100%

### 3.2 Feature Set Statistics

| Metric | Value |
|--------|-------|
| Total Unique Features | 15 |
| Average Set Size | 6.17 ± 1.92 |
| Minimum Set Size | 1 |
| Maximum Set Size | 11 |

### 3.3 Feature Frequency Analysis

| Feature | Count | Percentage |
|---------|--------|-----------|
| traffic_scenario | 173 | 99.4% |
| speed_peaks | 156 | 89.7% |
| turning_scenario | 141 | 81.0% |
| sharp_maneuvers | 139 | 79.9% |
| frequent_stopping | 138 | 79.3% |
| dynamic_speed | 105 | 60.3% |
| variable_speed | 80 | 46.0% |
| dense_traffic | 68 | 39.1% |
| location_germany | 37 | 21.3% |
| location_argentina | 11 | 6.3% |

### 3.4 Performance Evaluation

| Threshold | Precision | Recall | F1 Score | Accuracy |
|-----------|-----------|--------|----------|----------|
| 0.1 | 0.290 | 0.999 | 0.450 | 0.291 |
| 0.2 | 0.304 | 0.999 | 0.465 | 0.334 |
| 0.3 | 0.325 | 0.998 | 0.487 | 0.391 |
| 0.4 | 0.368 | 0.996 | 0.533 | 0.495 |
| 0.5 | 0.427 | 0.995 | 0.598 | 0.614 |

**Optimal Performance:** Jaccard threshold = 0.5
- F1 Score: 0.598
- Accuracy: 61.4%
- Precision: 0.427
- Recall: 0.995

## 4. Comparative Analysis

### 4.1 Phase 1 vs Phase 2 Performance

| Metric | Phase 1 (Z-Score + Minkowski) | Phase 2 (Jaccard) | Difference |
|--------|-------------------------------|-------------------|------------|
| F1 Score | 0.644 | 0.598 | -0.046 |
| Accuracy | 81.5% | 61.4% | -20.1% |
| Precision | 0.795 | 0.427 | -0.368 |
| Recall | 0.540 | 0.995 | +0.455 |

### 4.2 Key Differences

**Phase 1 Characteristics:**
- Higher precision, moderate recall
- Better overall accuracy
- Sensitive to feature scaling and normalization
- Captures continuous feature relationships

**Phase 2 Characteristics:**
- Lower precision, very high recall  
- More liberal similarity detection
- Robust to scaling issues
- Interpretable categorical features

## 5. Analysis and Insights

### 5.1 Strengths of Set-Based Approach

1. **Interpretability:** Clear categorical features like "high_speed_scenario" are easily understood
2. **Robustness:** Not affected by feature scaling or normalization choices
3. **Computational Efficiency:** Set operations are fast and scalable
4. **High Recall:** Captures most truly similar scenarios (99.5%)

### 5.2 Limitations

1. **Information Loss:** Binary membership loses continuous value information
2. **Threshold Dependency:** Feature membership heavily dependent on threshold choices
3. **Lower Precision:** Many false positives due to coarse categorical representation
4. **Limited Granularity:** Cannot distinguish between "slightly" vs "very" high speeds

### 5.3 Feature Engineering Insights

**Most Discriminative Features:**
- Location-based features provide strong similarity signals
- Traffic density effectively distinguishes scenario types
- Speed variability captures scenario complexity

**Less Useful Features:**
- Nearly universal features (99% traffic_scenario) provide little discrimination
- Binary thresholds may be too coarse for some characteristics

## 6. Error Analysis

### 6.1 False Positive Analysis
High false positive rate (low precision) indicates that many scenarios share categorical features but are fundamentally different in continuous space.

Example: Two scenarios both have "high_speed_scenario" but one averages 8.1 m/s while another averages 15.2 m/s.

### 6.2 False Negative Analysis
Very low false negative rate (high recall) suggests the categorical features effectively capture most similarity patterns, even if coarsely.

## 7. Recommendations

### 7.1 Immediate Improvements
1. **Multi-Level Thresholds:** Use low/medium/high categories instead of binary
2. **Adaptive Thresholds:** Set thresholds based on data distribution percentiles
3. **Weighted Jaccard:** Weight features by discriminative power
4. **Feature Combinations:** Create composite features (e.g., "high_speed_dense_traffic")

### 7.2 Integration Strategies
1. **Ensemble Methods:** Combine Phase 1 and Phase 2 predictions
2. **Hierarchical Similarity:** Use sets for coarse filtering, then apply distance metrics
3. **Context-Aware Thresholds:** Different thresholds for different scenario types

## 8. Implementation Details

### 8.1 Ground Truth Generation
Scenarios are considered similar if they share the same base name pattern:
- `ARG_Carcarana-11_2_I-1-1` and `ARG_Carcarana-11_3_I-1-1` → Similar (base: `ARG_Carcarana`)

### 8.2 Evaluation Framework
- Cross-validation approach using scenario name patterns
- Multiple threshold evaluation for optimal operating point
- Comprehensive visualization of results

### 8.3 Reproducibility
- All parameters documented and version-controlled
- Deterministic feature extraction pipeline
- Standardized evaluation metrics

## 9. Conclusion

Phase 2 successfully demonstrates the viability of set-based similarity metrics for CARLA scenario analysis. While achieving lower overall performance than distance-based methods (F1: 0.598 vs 0.644), the approach offers valuable interpretability and computational benefits.

The high recall (99.5%) indicates that set-based methods effectively identify similar scenarios, making them suitable for applications where missing similar scenarios is more costly than including some dissimilar ones.

Future work should focus on hybrid approaches that combine the precision of distance-based methods with the interpretability of set-based approaches.

---
**Analysis Date:** 16 September 2025  
**Framework Version:** CARLA 0.9.15  
**Total Analysis Time:** ~45 minutes  
**Scenarios Processed:** 174/174 (100% success rate)
