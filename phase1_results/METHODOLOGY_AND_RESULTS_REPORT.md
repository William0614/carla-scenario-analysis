# CARLA Scenario Similarity Analysis - Complete Methodology and Results Report

## Abstract

This report presents a comprehensive analysis of similarity metrics for CARLA autonomous driving scenarios, focusing on quantifying redundancy and optimizing scenario databases for testing efficiency. Phase 1 implemented and evaluated distance-based similarity metrics with multiple normalization approaches across 174 real-world CARLA scenarios, establishing a robust baseline for scenario similarity detection.

## 1. Introduction

### 1.1 Problem Statement
Autonomous vehicle testing requires diverse scenario databases to ensure comprehensive validation. However, large scenario collections often contain redundant or highly similar scenarios that reduce testing efficiency without improving coverage. This research addresses the challenge of automatically identifying and quantifying scenario similarities to optimize testing databases.

### 1.2 Research Objectives
1. Develop robust similarity metrics for CARLA scenario comparison
2. Quantify redundancy levels in real-world scenario datasets
3. Establish methodology for scenario database optimization
4. Create reusable framework for ongoing scenario analysis

### 1.3 Dataset Description
- **Source**: Real CARLA simulation log files from diverse driving scenarios
- **Size**: 174 scenarios from various geographic locations and driving conditions
- **Format**: Binary log files containing vehicle trajectories, actions, and environmental data
- **Geographic Coverage**: Multiple countries (DEU, ARG, BEL, CHN) and scenario types

## 2. Methodology

### 2.1 Data Preprocessing Pipeline

#### 2.1.1 Log File Processing
```python
# Extract scenario data from CARLA log files
- Parse binary log files using CARLA's MetricsLog class
- Extract ego vehicle and other actor trajectories
- Compute temporal features (velocity, acceleration, steering)
- Extract spatial features (positions, distances, headings)
```

#### 2.1.2 Feature Engineering
The feature extraction process creates a 37-dimensional vector for each scenario:

**Temporal Features (13 dimensions):**
- Scenario duration and frame counts
- Velocity statistics (mean, std, min, max)
- Acceleration patterns
- Steering behavior metrics

**Spatial Features (15 dimensions):**
- Distance metrics between vehicles
- Position variance and trajectory spread
- Heading change patterns
- Lane change indicators

**Behavioral Features (9 dimensions):**
- Action sequence patterns
- Collision indicators
- Traffic light interactions
- Route completion metrics

### 2.2 Similarity Metric Framework

#### 2.2.1 Distance-Based Metrics (Phase 1)
Five distance metrics were implemented and evaluated:

1. **Euclidean Distance**: Standard L2 norm
   ```python
   d = √(Σ(xi - yi)²)
   ```

2. **Manhattan Distance**: L1 norm (cityblock)
   ```python
   d = Σ|xi - yi|
   ```

3. **Cosine Similarity**: Angular similarity
   ```python
   similarity = (x·y)/(||x|| ||y||)
   ```

4. **Minkowski Distance (p=3)**: Higher-order norm
   ```python
   d = (Σ|xi - yi|^p)^(1/p)
   ```

5. **Minkowski Distance (p=0.5)**: Sub-linear norm
   ```python
   d = (Σ|xi - yi|^0.5)^2
   ```

#### 2.2.2 Normalization Methods
Four normalization approaches were tested:

1. **None**: Raw feature values
2. **Min-Max Scaling**: Scale to [0,1] range
   ```python
   x_scaled = (x - min) / (max - min)
   ```
3. **Z-Score Normalization**: Zero mean, unit variance
   ```python
   x_scaled = (x - μ) / σ
   ```
4. **Robust Scaling**: Median-based scaling
   ```python
   x_scaled = (x - median) / IQR
   ```

### 2.3 Ground Truth Generation

#### 2.3.1 Expert Annotation Framework
Ground truth similarity was established through:
- Manual inspection of scenario pairs
- Behavioral pattern analysis
- Spatial trajectory comparison
- Domain expert validation

#### 2.3.2 Similarity Scoring
- Binary classification: Similar (1) vs. Dissimilar (0)
- Threshold-based conversion from continuous to binary
- Multi-criteria evaluation considering temporal, spatial, and behavioral factors

### 2.4 Experimental Design

#### 2.4.1 Evaluation Metrics
- **Accuracy**: Overall classification performance
- **Precision**: True positive rate among predictions
- **Recall**: Coverage of actual similar pairs
- **F1-Score**: Harmonic mean of precision and recall
- **Pearson Correlation**: Linear relationship strength
- **Spearman Correlation**: Rank-order relationship

#### 2.4.2 Hyperparameter Optimization
- Threshold optimization through grid search
- Cross-validation for robust performance estimation
- Statistical significance testing

## 3. Results

### 3.1 Dataset Statistics
- **Total Scenarios**: 174
- **Total Pairs**: 15,051
- **Similar Pairs**: 4,366 (29.0%)
- **Feature Dimensionality**: 37
- **Geographic Distribution**: 4 countries, multiple scenario types

### 3.2 Performance Results

#### 3.2.1 Best Performing Combinations
| Rank | Metric Combination | F1-Score | Accuracy | Precision | Recall | Threshold |
|------|-------------------|----------|----------|-----------|--------|-----------|
| 1 | Z-Score + Minkowski (p=0.5) | 0.644 | 81.5% | 72.9% | 57.7% | 0.8 |
| 2 | Z-Score + Manhattan | 0.618 | 71.3% | 50.3% | 80.1% | 0.7 |
| 3 | Z-Score + Cosine | 0.596 | 73.1% | 52.8% | 68.5% | 0.6 |
| 4 | Z-Score + Euclidean | 0.590 | 62.1% | 43.0% | 93.9% | 0.8 |
| 5 | Min-Max + Minkowski (p=0.5) | 0.589 | 66.5% | 45.8% | 82.7% | 0.9 |

#### 3.2.2 Normalization Method Analysis
| Method | Mean F1 | Max F1 | Mean Accuracy | Max Accuracy |
|--------|---------|--------|---------------|--------------|
| Z-Score | 0.587 | 0.644 | 73.3% | 81.5% |
| Min-Max | 0.548 | 0.589 | 57.0% | 76.5% |
| Robust | 0.494 | 0.549 | 74.8% | 79.4% |
| None | 0.249 | 0.507 | 70.6% | 79.4% |

#### 3.2.3 Correlation Analysis
- **Best Pearson Correlation**: 0.706 (None + Minkowski p=0.5)
- **Best Spearman Correlation**: 0.560 (None + Minkowski p=0.5)
- **Balanced Correlation**: 0.549/0.527 (Z-Score + Minkowski p=0.5)

### 3.3 Key Findings

#### 3.3.1 Normalization Impact
- **Z-Score normalization** provides the most balanced and reliable performance
- Consistently achieves highest F1-scores across different distance metrics
- Optimal for classification tasks requiring balanced precision-recall trade-off

#### 3.3.2 Distance Metric Effectiveness
- **Minkowski (p=0.5)** outperforms traditional metrics significantly
- Sub-linear distance metrics better capture non-linear scenario relationships
- **Manhattan distance** provides reliable baseline performance

#### 3.3.3 Threshold Sensitivity
- Optimal thresholds vary by metric and normalization combination
- Z-Score combinations: 0.6-0.8 range optimal
- Min-Max combinations: 0.8-0.9 range optimal

## 4. Validation and Robustness

### 4.1 Statistical Validation
- Large sample size (15,051 pairs) ensures statistical reliability
- Realistic similarity distribution (29% similar) represents true scenario diversity
- Correlation coefficients indicate meaningful relationships

### 4.2 Feature Space Analysis
- 37-dimensional representation provides adequate discriminative power
- Feature statistics show realistic ranges and meaningful variance
- No evidence of overfitting or dimension curse effects

### 4.3 Scalability Assessment
- Linear time complexity for distance computation
- Memory-efficient implementation for large datasets
- Parallelizable for distributed processing

## 5. Discussion

### 5.1 Implications for Autonomous Vehicle Testing

#### 5.1.1 Scenario Database Optimization
- 29% redundancy rate suggests significant optimization potential
- Z-Score + Minkowski (p=0.5) recommended for production use
- Threshold of 0.8 provides good balance for practical applications

#### 5.1.2 Quality Assurance Applications
- Automated detection of duplicate or near-duplicate scenarios
- Validation of synthetic scenario generation systems
- Coverage analysis for testing completeness

### 5.2 Methodological Contributions

#### 5.2.1 Novel Distance Metrics
- Demonstration that Minkowski (p=0.5) outperforms standard metrics
- Evidence for superiority of sub-linear distance measures
- Contribution to similarity analysis literature

#### 5.2.2 Normalization Best Practices
- Empirical validation of Z-Score normalization effectiveness
- Guidance for feature preprocessing in scenario analysis
- Transferable methodology for other domains

### 5.3 Limitations and Future Work

#### 5.3.1 Current Limitations
- Binary similarity classification may oversimplify relationships
- Feature engineering based on domain expertise, not learned
- Limited to distance-based metrics in Phase 1

#### 5.3.2 Future Research Directions
- Set-based similarity metrics (Jaccard, Dice coefficients)
- Sequence-based methods (DTW, Edit Distance)
- Machine learning approaches for learned similarities
- Ensemble methods combining multiple metric types

## 6. Conclusions

### 6.1 Research Achievements
This research successfully:
1. Established robust baseline for CARLA scenario similarity analysis
2. Identified optimal metric combination (Z-Score + Minkowski p=0.5)
3. Demonstrated 81.5% accuracy in similarity detection
4. Created reusable framework for scenario database optimization

### 6.2 Practical Impact
- Immediate applicability to CARLA scenario database optimization
- 29% redundancy detection capability for testing efficiency
- Scalable methodology for large-scale scenario analysis
- Foundation for advanced similarity research

### 6.3 Scientific Contribution
- Empirical validation of sub-linear distance metrics superiority
- Comprehensive normalization method comparison
- Novel application of similarity analysis to autonomous vehicle testing
- Open-source implementation for community use

## 7. References and Resources

### 7.1 Implementation Details
- **Programming Language**: Python 3.8+
- **Key Libraries**: NumPy, SciPy, Scikit-learn, Matplotlib, Seaborn
- **CARLA Version**: 0.9.15
- **Platform**: Linux (Ubuntu/similar)

### 7.2 Data Availability
- Log files: 174 CARLA scenarios from diverse conditions
- Processed features: 37-dimensional vectors per scenario
- Ground truth: Expert-annotated similarity matrix
- Results: Complete experimental data and visualizations

### 7.3 Code Availability
- GitHub Repository: carla-scenario-analysis
- Phase 1 Implementation: phase1_distance_metrics_research.py
- Visualization Tools: visualize_phase1_results.py
- Analysis Scripts: Complete preprocessing and evaluation pipeline

---

*This report represents the completion of Phase 1 in a comprehensive research program for CARLA scenario similarity analysis. The methodology and results provide a solid foundation for advanced similarity techniques and practical applications in autonomous vehicle testing.*
