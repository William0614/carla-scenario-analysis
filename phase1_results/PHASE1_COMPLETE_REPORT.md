# CARLA Scenario Similarity Analysis - Phase 1 Complete Results

**Date**: September 8, 2025  
**Dataset**: 174 CARLA scenarios (Full Dataset)  
**Experiment Type**: Distance-Based Similarity Metrics with Normalization Methods  

## 🎯 Executive Summary

Phase 1 of the CARLA scenario similarity analysis has been completed successfully, testing 5 distance-based metrics with 4 normalization methods across 174 scenarios (15,051 scenario pairs). The research identified **Z-Score normalization with Minkowski distance (p=0.5)** as the best-performing combination.

### Key Results:
- **Best F1-Score**: 0.644 (Z-Score + Minkowski p=0.5)
- **Best Accuracy**: 81.5% (Z-Score + Minkowski p=0.5)  
- **Best Correlation**: 0.706 (No Normalization + Minkowski p=0.5)
- **Optimal Threshold**: 0.8 for Z-Score combinations

## 📊 Performance Analysis

### Top 5 Performing Combinations:
1. **Z-Score + Minkowski (p=0.5)** - F1: 0.644, Accuracy: 81.5%
2. **Z-Score + Manhattan** - F1: 0.618, Accuracy: 71.3%
3. **Z-Score + Cosine** - F1: 0.596, Accuracy: 73.1%
4. **Z-Score + Euclidean** - F1: 0.590, Accuracy: 62.1%
5. **Min-Max + Minkowski (p=0.5)** - F1: 0.589, Accuracy: 66.5%

### Normalization Method Performance:
- **Z-Score**: Best overall performance (mean F1: 0.587, max accuracy: 81.5%)
- **Min-Max**: Good balanced performance (mean F1: 0.548, max accuracy: 76.5%)
- **Robust**: Conservative approach (mean F1: 0.494, high precision)
- **None**: Best for correlation analysis but poor classification performance

### Distance Metric Insights:
- **Minkowski (p=0.5)**: Superior performance across normalization methods
- **Manhattan**: Consistent strong performance, good baseline
- **Cosine**: Moderate performance, good for normalized features
- **Euclidean**: Standard baseline, average performance
- **Minkowski (p=3)**: Conservative, high precision but low recall

## 🔬 Technical Details

### Dataset Characteristics:
- **Total Scenarios**: 174
- **Feature Dimensions**: 37 features per scenario
- **Total Pairs**: 15,051
- **Similar Pairs (Ground Truth)**: 4,366 (29.0%)
- **Feature Types**: Movement patterns, vehicle behaviors, spatial relationships

### Evaluation Metrics:
- **Accuracy**: Overall classification performance
- **F1-Score**: Harmonic mean of precision and recall
- **Precision**: True positive rate among predicted positives
- **Recall**: True positive rate among actual positives
- **Pearson/Spearman Correlation**: Linear and rank-order relationships

### Ground Truth Validation:
- Ground truth similarity matrix shows realistic distribution
- Mean similarity: 0.370 (std: 0.403)
- Range: 0.1 to 1.0, indicating good scenario diversity

## 📈 Key Findings

### 1. Normalization Impact
- **Z-Score normalization** provides the most balanced and reliable performance
- **Min-Max scaling** tends to over-predict similarities (high recall, lower precision)
- **No normalization** works well for correlation analysis but poor for classification
- **Robust scaling** provides conservative predictions with high precision

### 2. Distance Metric Effectiveness
- **Minkowski with p=0.5** captures non-linear relationships effectively
- Lower p-values in Minkowski distance better suited for high-dimensional scenario data
- **Manhattan distance** provides consistent, interpretable results
- **Cosine similarity** effective when feature magnitudes vary significantly

### 3. Threshold Optimization
- Optimal thresholds vary significantly by metric and normalization
- Z-Score combinations: 0.6-0.8 range
- Min-Max combinations: 0.8-0.9 range
- Raw data: 0.5 typically optimal

### 4. Feature Space Analysis
- 37-dimensional feature space provides good discriminative power
- Feature statistics show realistic ranges and distributions
- Standard deviations indicate meaningful variance across scenarios

## 🎨 Visualization Outputs

The following visualization files have been generated:
- `phase1_heatmap_f1_score.png`: Performance heatmap for F1-scores
- `phase1_heatmap_accuracy.png`: Accuracy performance across all combinations
- `phase1_heatmap_pearson_correlation.png`: Correlation coefficient analysis
- `phase1_top_performers.png`: Detailed analysis of top 10 performers
- `phase1_normalization_comparison.png`: Normalization method comparison
- `phase1_metric_comparison.png`: Distance metric effectiveness analysis

## 🚀 Research Implications

### For CARLA Scenario Analysis:
1. **Z-Score + Minkowski (p=0.5)** recommended for scenario redundancy detection
2. Threshold of 0.8 provides good balance of precision and recall
3. Feature engineering captures meaningful scenario characteristics
4. Current approach scales well to large scenario datasets

### For Autonomous Vehicle Testing:
1. 29% similarity rate suggests good scenario diversity in current dataset
2. Identified redundant scenarios can be removed to optimize testing efficiency
3. Similarity metrics can guide synthetic scenario generation
4. Quality assurance for scenario databases

## 📋 Next Steps: Phase 2 Implementation

Based on Phase 1 results, the following Phase 2 activities are recommended:

### Immediate Actions:
1. **Implement Set-Based Metrics**: Jaccard, Dice, Overlap coefficients
2. **Validate with Cross-Validation**: K-fold validation on current results
3. **Feature Selection**: Analyze feature importance and reduce dimensionality
4. **Ensemble Methods**: Combine top-performing metrics

### Advanced Research:
1. **Sequence-Based Metrics**: DTW, Edit Distance, LCS for temporal patterns
2. **Machine Learning Approaches**: Learned similarity metrics
3. **Probabilistic Methods**: Statistical distance measures
4. **Domain-Specific Features**: CARLA-specific scenario characteristics

## 📊 Data Files Generated

- `phase1_distance_metrics_results.json`: Complete experimental results
- `phase1_analysis_summary.md`: Detailed methodology and findings
- `visualize_phase1_results.py`: Visualization and analysis script
- All visualization PNG files for presentation and documentation

## 🏆 Conclusion

Phase 1 successfully established a robust baseline for CARLA scenario similarity analysis. The identification of **Z-Score normalization with Minkowski distance (p=0.5)** as the optimal approach provides a strong foundation for scenario redundancy detection and database optimization.

The research demonstrates that:
- Distance-based metrics can effectively capture scenario similarities
- Proper normalization is crucial for classification performance
- Non-standard distance metrics (Minkowski p<1) may outperform traditional approaches
- The current 37-feature representation provides meaningful scenario characterization

This foundation enables confident progression to Phase 2 set-based metrics and subsequent advanced similarity analysis techniques.

---

*This analysis was conducted as part of a comprehensive research program to optimize CARLA scenario databases for autonomous vehicle testing and validation.*
