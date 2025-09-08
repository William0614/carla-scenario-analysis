# Phase 1 Distance Metrics Results Analysis

## Experiment Overview
- **Date**: September 8, 2025
- **Dataset**: 174 CARLA scenarios (full dataset)
- **Total Pairs**: 15,051 scenario pairs
- **Ground Truth Similar Pairs**: 4,366 (29.0% of total pairs)
- **Feature Dimensionality**: 37 features per scenario

## Metrics Tested
1. **Euclidean Distance**
2. **Manhattan Distance (cityblock)**
3. **Cosine Similarity**
4. **Minkowski Distance (p=3)**
5. **Minkowski Distance (p=0.5)**

## Normalization Methods
1. **None** - Raw features
2. **Min-Max** - Scale to [0,1] range
3. **Z-Score** - Standard normalization (mean=0, std=1)
4. **Robust** - Median-based scaling

## Top Performing Combinations

### Best F1-Score Results:
1. **Z-Score + Minkowski (p=0.5)** - F1: 0.644, Accuracy: 81.5%
   - Threshold: 0.8
   - Precision: 72.9%, Recall: 57.7%
   - Pearson Correlation: 0.549

2. **Z-Score + Manhattan** - F1: 0.618, Accuracy: 71.3%
   - Threshold: 0.7
   - Precision: 50.3%, Recall: 80.1%
   - Pearson Correlation: 0.536

3. **Z-Score + Cosine** - F1: 0.596, Accuracy: 73.1%
   - Threshold: 0.6
   - Precision: 52.8%, Recall: 68.5%
   - Pearson Correlation: 0.438

### Best Accuracy Results:
1. **Z-Score + Minkowski (p=0.5)** - Accuracy: 81.5%
2. **Robust + Minkowski (p=3)** - Accuracy: 79.4%
3. **None + Minkowski (p=0.5)** - Accuracy: 79.4%

### Best Correlation Results:
1. **None + Minkowski (p=0.5)** - Pearson: 0.706, Spearman: 0.560
2. **None + Manhattan** - Pearson: 0.550, Spearman: 0.511
3. **Z-Score + Minkowski (p=0.5)** - Pearson: 0.549, Spearman: 0.527

## Key Findings

### 1. Normalization Impact
- **Z-Score normalization** consistently provides the best balanced performance
- **No normalization** works well for correlation but poor for classification
- **Min-Max** tends to over-predict similarities (high recall, low precision)
- **Robust normalization** provides conservative predictions

### 2. Distance Metric Performance
- **Minkowski with p=0.5** emerges as the best overall metric
  - Captures non-linear relationships better than standard metrics
  - Performs well across different normalization methods
- **Manhattan distance** shows strong consistent performance
- **Cosine similarity** moderate performance, good for normalized data
- **Euclidean distance** standard baseline performance
- **Minkowski with p=3** very conservative, high precision but low recall

### 3. Threshold Sensitivity
- Best thresholds vary significantly by metric and normalization:
  - Z-Score combinations: 0.6-0.8
  - Min-Max combinations: 0.8-0.9
  - No normalization: 0.5
  - Robust normalization: 0.5-0.9

### 4. Trade-offs Observed
- **High Precision**: Minkowski p=3, No normalization metrics
- **High Recall**: Min-Max normalized metrics
- **Balanced Performance**: Z-Score + Minkowski p=0.5

## Recommendations for Next Phases

### Phase 2: Set-Based Metrics
Based on Phase 1 results, implement:
1. **Jaccard Index** with Z-Score normalization
2. **Dice Coefficient** with multiple normalization methods
3. **Overlap Coefficient** for asymmetric similarities

### Phase 3: Sequence-Based Metrics
Focus on:
1. **Dynamic Time Warping (DTW)** for temporal patterns
2. **Edit Distance** for action sequences
3. **Longest Common Subsequence** for behavioral patterns

### Feature Engineering Insights
- Current 37-feature representation shows good discriminative power
- Consider feature selection based on correlation analysis
- Investigate feature interactions and non-linear transformations

## Statistical Significance
- Large sample size (15,051 pairs) provides statistical reliability
- Ground truth distribution (29% similar) represents realistic scenario diversity
- Correlation coefficients indicate meaningful relationships between metrics and ground truth

## Next Steps
1. Implement Phase 2 set-based similarity metrics
2. Validate results with cross-validation
3. Investigate ensemble methods combining top-performing metrics
4. Analyze feature importance and dimensionality reduction
5. Scale to larger datasets and different scenario types
