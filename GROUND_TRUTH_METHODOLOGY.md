# Multi-Dimensional Ground Truth Methodology for CARLA Scenario Similarity

## Problem Statement

The original ground truth method in this repository relies solely on filename patterns to determine scenario similarity. This approach has significant limitations:

### Current Method Weaknesses
1. **Filename-only approach**: Scenarios like `ARG_Carcarana-1_1_I-1-1.log` vs `ARG_Carcarana-1_3_I-1-1.log` are considered similar based only on shared location
2. **No behavioral analysis**: Completely different driving behaviors are ignored
3. **No spatial consideration**: Different road layouts, path geometries, and traffic patterns are overlooked
4. **Arbitrary weighting**: Scoring weights (0.4, 0.3, 0.2, 0.1) lack empirical validation
5. **Binary classification**: Complex similarity relationships are reduced to simple similar/not-similar labels

## Proposed Multi-Dimensional Ground Truth

We propose a composite similarity measure that evaluates four key dimensions:

### 1. Behavioral Similarity (Weight: 40%)
**Rationale**: Most critical for autonomous vehicle testing - similar behaviors indicate similar testing value.

**Metrics**:
- Action sequence similarity using edit distance and LCS
- Speed profile correlation (Pearson correlation of speed time series)
- Control pattern similarity (throttle, brake, steering patterns)
- Behavioral transition patterns (how actions flow together)

**Calculation**:
```
behavioral_sim = 0.4 * action_similarity + 0.3 * speed_correlation + 0.3 * control_similarity
```

### 2. Spatial Similarity (Weight: 30%) 
**Rationale**: Similar spatial patterns indicate similar road geometry and navigation challenges.

**Metrics**:
- Path geometry similarity using Dynamic Time Warping (DTW)
- Bounding box overlap and area comparison
- Curvature pattern correlation
- Total displacement and path length ratios

**Calculation**:
```
spatial_sim = 0.4 * path_dtw_similarity + 0.3 * bbox_overlap + 0.3 * geometry_correlation
```

### 3. Traffic/Environmental Similarity (Weight: 20%)
**Rationale**: Traffic context significantly affects scenario complexity and testing relevance.

**Metrics**:
- Vehicle count similarity
- Traffic density patterns
- Interaction complexity (based on traffic vehicle behaviors)
- Environmental conditions if available

**Calculation**:
```
traffic_sim = 0.6 * vehicle_count_similarity + 0.4 * density_pattern_similarity
```

### 4. Contextual Similarity (Weight: 10%)
**Rationale**: Geographic and scenario family context provides useful but secondary information.

**Metrics**:
- Geographic region (country/city matching)
- Scenario family classification (intersection, highway, urban)
- Time of day and weather conditions if available

**Calculation**:
```
contextual_sim = 0.5 * geographic_similarity + 0.5 * scenario_family_similarity
```

## Combined Similarity Score

**Final Formula**:
```
total_similarity = 0.4 * behavioral_sim + 0.3 * spatial_sim + 0.2 * traffic_sim + 0.1 * contextual_sim
```

**Threshold for "Similar" Classification**: 0.65
- This threshold was chosen to be more conservative than the original 0.6 threshold
- Requires meaningful similarity across multiple dimensions rather than just filename matching

## Implementation Considerations

### Weight Justification
- **Behavioral (40%)**: Primary concern for autonomous vehicle testing validity
- **Spatial (30%)**: Critical for navigation and path planning challenges  
- **Traffic (20%)**: Important for interaction complexity but secondary to core behavior
- **Contextual (10%)**: Useful for organization but least important for testing equivalence

### Similarity Metric Choices
- **Edit Distance/LCS**: Proven effective for sequence comparison in Phase 3 results
- **DTW**: Handles temporal alignment issues in spatial path comparison
- **Pearson Correlation**: Standard measure for continuous signal similarity
- **Jaccard Index**: Effective for set-based comparisons (traffic patterns)

### Threshold Considerations
- **Per-dimension thresholds**: Each dimension uses normalized [0,1] scale
- **Combined threshold (0.65)**: Requires substantial similarity across multiple dimensions
- **Sensitivity analysis**: Threshold can be adjusted based on validation results

## Validation Strategy

### Phase 1: Automated Validation
1. **Consistency Check**: Verify symmetric similarity (sim(A,B) = sim(B,A))
2. **Triangle Inequality**: Check transitivity properties where applicable
3. **Extreme Case Analysis**: Verify identical scenarios score 1.0, completely different score ~0.0

### Phase 2: Expert Validation
1. **Sample Annotation**: Domain experts manually rate 100-200 scenario pairs
2. **Inter-rater Agreement**: Multiple experts rate same pairs, measure agreement
3. **Correlation Analysis**: Compare expert ratings with automated scores
4. **Threshold Calibration**: Adjust threshold based on expert agreement

### Phase 3: Cross-Validation
1. **Split Validation**: Use different scenario subsets for training/testing thresholds
2. **Sensitivity Analysis**: Test robustness to weight and threshold variations
3. **Comparison Studies**: Compare against original filename-based method

## Implementation Notes

### Computational Efficiency
- **Batch Processing**: Process scenarios in batches for memory efficiency
- **Caching**: Store computed features to avoid recomputation
- **Parallel Processing**: Utilize multiprocessing for pairwise comparisons
- **Progressive Filtering**: Use fast metrics first, detailed analysis for promising pairs

### Scalability Considerations
- **Memory Usage**: Large datasets may require streaming or chunked processing
- **Time Complexity**: O(n²) pairwise comparison limits scale - consider approximate methods for very large datasets
- **Storage**: Precomputed similarity matrices for repeated analysis

### Error Handling
- **Missing Data**: Graceful handling of incomplete log files
- **Corrupted Logs**: Validation and error reporting for malformed data
- **Feature Extraction Failures**: Robust fallback mechanisms

## Expected Impact

### Improved Ground Truth Quality
- **Reduced False Positives**: Scenarios with same location but different behaviors no longer automatically similar
- **Increased True Positives**: Scenarios with similar behaviors across locations properly identified
- **Better Discrimination**: More nuanced similarity scores enable better threshold tuning

### Enhanced Research Validity
- **Metric Evaluation**: More reliable assessment of similarity algorithm performance
- **Generalization**: Results more likely to transfer to real-world scenario analysis
- **Interpretability**: Multi-dimensional scores provide insight into similarity sources

### Practical Applications
- **Test Suite Optimization**: Better identification of redundant scenarios for removal
- **Scenario Generation**: Improved validation of synthetic scenario uniqueness
- **Quality Assurance**: Automated detection of scenario database inconsistencies

## Future Extensions

### Additional Dimensions
- **Semantic Similarity**: Road sign types, lane configurations, intersection types
- **Weather/Lighting**: Environmental condition impacts on driving behavior
- **Mission Context**: Goal-oriented similarity (parking vs navigation vs emergency response)

### Advanced Techniques
- **Machine Learning**: Train similarity models from expert annotations
- **Deep Embeddings**: Learn scenario representations via neural networks
- **Multi-Scale Analysis**: Hierarchical similarity from coarse to fine-grained features

### Integration Opportunities
- **Real-time Analysis**: Incorporate into scenario generation pipelines
- **Dashboard Integration**: Visual similarity exploration and analysis tools
- **API Development**: Standalone similarity service for external tools

## References and Methodology Sources

1. **Phase 3 Results**: N-gram Jaccard method achieved best performance (F1: 0.702)
2. **DTW Literature**: Dynamic Time Warping for temporal sequence alignment
3. **Autonomous Vehicle Testing**: Domain-specific requirements for scenario similarity
4. **Information Retrieval**: Multi-dimensional similarity combination methods
5. **Expert Systems**: Human-in-the-loop validation approaches

---

**Document Version**: 1.0  
**Last Updated**: October 6, 2025  
**Authors**: Research Team  
**Status**: Implementation Ready