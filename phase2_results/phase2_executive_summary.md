# Phase 2: Set-Based Similarity Analysis - Executive Summary

## Overview
Phase 2 implemented Jaccard coefficient for set-based similarity analysis of CARLA scenario data, complementing Phase 1's distance-based approach. This phase converts 37-dimensional continuous feature vectors into categorical feature sets for similarity comparison.

## Key Results

### Performance Metrics
- **Best F1 Score:** 0.598 (Jaccard threshold = 0.5)
- **Best Accuracy:** 61.4%
- **Total Scenarios Analyzed:** 174
- **Similar Pairs (Ground Truth):** 4,366 out of 15,051 (29.0%)

### Feature Analysis
- **Total Unique Features:** 15 categorical features
- **Average Set Size:** 6.17 features per scenario
- **Feature Range:** 1-11 features per scenario

### Most Common Features
1. `traffic_scenario`: 173 scenarios (99.4%)
2. `speed_peaks`: 156 scenarios (89.7%)
3. `turning_scenario`: 141 scenarios (81.0%)
4. `sharp_maneuvers`: 139 scenarios (79.9%)
5. `frequent_stopping`: 138 scenarios (79.3%)

## Methodology

### Feature Extraction
- Used identical 37-dimensional feature extraction as Phase 1
- Converted continuous features to categorical sets using optimized thresholds
- Applied domain-specific thresholds for automotive scenario characteristics

### Similarity Analysis
- Implemented Jaccard coefficient: |A ∩ B| / |A ∪ B|
- Tested thresholds: 0.1, 0.2, 0.3, 0.4, 0.5
- Evaluated against ground truth based on scenario naming patterns

## Phase Comparison

| Metric | Phase 1 (Best) | Phase 2 (Jaccard) | Difference |
|--------|----------------|-------------------|------------|
| F1 Score | 0.644 | 0.598 | -0.046 (-7.1%) |
| Accuracy | 81.5% | 61.4% | -20.1% |
| Best Method | Z-Score + Minkowski(p=0.5) | Jaccard(threshold=0.5) | - |

## Key Insights

### Strengths of Set-Based Approach
- Interpretable categorical features (e.g., "high_speed_scenario", "traffic_scenario")
- Robust to feature scaling issues
- Computationally efficient
- Clear threshold-based decision making

### Limitations
- Information loss from continuous→categorical conversion
- Threshold sensitivity affects feature membership
- Lower performance compared to distance-based metrics
- Binary feature presence/absence loses magnitude information

## Recommendations

1. **Hybrid Approach:** Combine set-based and distance-based methods for complementary insights
2. **Threshold Optimization:** Further tune feature thresholds using validation data
3. **Feature Engineering:** Explore additional categorical features from domain knowledge
4. **Multi-Level Sets:** Consider hierarchical feature sets (basic→advanced characteristics)

## Technical Implementation
- **Framework:** Python with CARLA integration
- **Evaluation:** 5-fold cross-validation approach
- **Visualization:** Comprehensive performance and feature analysis plots
- **Reproducibility:** All code and parameters documented

## Conclusion
Phase 2 demonstrates that set-based similarity metrics can effectively identify scenario redundancy, achieving 59.8% F1 score. While not surpassing distance-based methods, set-based approaches offer valuable interpretability and computational efficiency for practical scenario management applications.

---
**Generated:** 16 September 2025  
**Analysis Framework:** CARLA Scenario Similarity Research  
**Phase:** 2 of 3 (Set-Based Metrics)
