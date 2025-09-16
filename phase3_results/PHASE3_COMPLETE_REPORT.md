# Phase 3: Sequence-Based Similarity Analysis - Complete Report

## Executive Summary

Phase 3 successfully implemented and evaluated sequence-based similarity metrics for CARLA scenario analysis. Using simple action sequences derived from vehicle control data, we analyzed 169 scenarios and found that **N-gram Jaccard similarity** performed best with an **F1 score of 0.702** and **accuracy of 81.2%**.

## Methodology

### 1. Action Sequence Extraction

We extracted driving action sequences from CARLA log files using the following approach:

#### Action Classification Thresholds:
- **Speed thresholds**: Stop (<0.5 m/s), Cruise (1-8 m/s), Fast (>8 m/s)
- **Steering thresholds**: Light steering (>0.05), Heavy steering/turning (>0.15)
- **Acceleration**: Change >1.0 m/s² from previous frame
- **Braking**: Brake input >0.3

#### Action Types Identified:
1. **STOP** - Vehicle stationary
2. **IDLE** - Very low speed movement
3. **CRUISE** - Normal speed driving
4. **CRUISE_FAST** - High-speed driving
5. **ACCELERATE** - Significant speed increase
6. **BRAKE** - Active braking
7. **STEER_LEFT/RIGHT** - Light steering adjustments
8. **TURN_LEFT/RIGHT** - Heavy steering/turning maneuvers

### 2. Sequence Processing

- **Sampling**: Every 5th frame to reduce noise while maintaining behavioral patterns
- **Compression**: Consecutive duplicate actions removed to focus on behavioral changes
- **Length**: Average sequence length of 21.8 actions (range: 3-75)

### 3. Similarity Metrics Evaluated

#### 3.1 Edit Distance (Levenshtein)
- **Purpose**: Measures minimum edits to transform one sequence to another
- **Best threshold**: 0.4
- **Performance**: F1=0.587, Accuracy=76.3%

#### 3.2 Longest Common Subsequence (LCS)
- **Purpose**: Finds longest common subsequence between sequences
- **Best threshold**: 0.5  
- **Performance**: F1=0.589, Accuracy=78.3%

#### 3.3 Dynamic Time Warping (DTW)
- **Purpose**: Aligns sequences with different temporal patterns
- **Best threshold**: 0.5
- **Performance**: F1=0.630, Accuracy=69.9%

#### 3.4 N-gram Jaccard Similarity (Winner) 🏆
- **Purpose**: Compares 2-grams (action pairs) using Jaccard coefficient
- **Best threshold**: 0.3
- **Performance**: **F1=0.702, Accuracy=81.2%**
- **Why it works**: Captures behavioral patterns in action transitions

#### 3.5 Global Sequence Alignment
- **Purpose**: Optimal alignment with gap penalties
- **Best threshold**: 0.5
- **Performance**: F1=0.584, Accuracy=71.4%

## Results Analysis

### Performance Ranking:
1. **N-gram Jaccard**: F1=0.702, Acc=81.2% (threshold=0.3)
2. **DTW Similarity**: F1=0.630, Acc=69.9% (threshold=0.5)  
3. **LCS Similarity**: F1=0.589, Acc=78.3% (threshold=0.5)
4. **Edit Distance**: F1=0.587, Acc=76.3% (threshold=0.4)
5. **Alignment Score**: F1=0.584, Acc=71.4% (threshold=0.5)

### Key Insights:

#### 1. N-gram Jaccard Success Factors:
- **Action transitions matter**: Pairs like "BRAKE→STOP" or "ACCELERATE→CRUISE_FAST" capture driving patterns
- **Robust to length differences**: Jaccard similarity naturally handles varying sequence lengths
- **Pattern recognition**: Identifies similar behavioral patterns regardless of sequence timing

#### 2. Sequence Characteristics:
- **169/174 scenarios** successfully processed (97.1% success rate)
- **14,196 scenario pairs** analyzed
- **29.7% ground truth similarity** based on scenario naming patterns
- **Action distribution**:
  - CRUISE_FAST: 29.1% (most common)
  - BRAKE: 24.6% 
  - STEER_LEFT: 10.7%
  - ACCELERATE: 7.8%
  - STEER_RIGHT: 7.0%

#### 3. Comparison with Previous Phases:
- **Phase 1 (Distance-based)**: Best F1=0.644 (Z-Score + Minkowski p=0.5)
- **Phase 2 (Set-based)**: Best F1=0.598 (Jaccard threshold=0.5)
- **Phase 3 (Sequence-based)**: **Best F1=0.702** (N-gram Jaccard) ⭐

## Technical Implementation

### Data Processing Pipeline:
1. **Log File Reading**: Used CARLA client to parse binary recorder files
2. **Action Extraction**: Frame-by-frame vehicle control analysis
3. **Sequence Building**: Action classification and consecutive duplicate removal
4. **Similarity Calculation**: Pairwise comparison using multiple metrics
5. **Evaluation**: F1 score and accuracy against ground truth

### Code Structure:
- **ActionSequenceAnalyzer**: Main analysis class
- **Similarity metrics**: Individual implementation of each metric
- **Ground truth**: Pattern-based similarity from scenario names
- **Visualization**: Performance comparison plots

## Limitations and Future Work

### Current Limitations:
1. **Simple action model**: Basic classification may miss subtle behavioral differences
2. **Ground truth assumptions**: Naming-based similarity may not capture all true similarities
3. **Temporal information**: Some metrics lose precise timing information
4. **Context independence**: Actions analyzed without considering traffic context

### Future Improvements:
1. **Hierarchical actions**: Multi-level action classification (macro + micro behaviors)
2. **Context-aware sequences**: Include traffic light states, nearby vehicles
3. **Learned representations**: Use neural networks for action embedding
4. **Temporal clustering**: Group similar temporal patterns
5. **Multi-modal analysis**: Combine sequence with spatial/statistical features

## Conclusion

Phase 3 successfully demonstrated that sequence-based similarity analysis can effectively identify similar driving scenarios. The **N-gram Jaccard similarity metric achieved the best performance** with F1=0.702 and accuracy=81.2%, outperforming both distance-based (Phase 1) and set-based (Phase 2) approaches.

The success of N-gram Jaccard suggests that **behavioral pattern recognition through action transitions** is more effective than individual action analysis or statistical feature comparison for scenario similarity assessment.

## Files Generated

- `phase3_sequence_based_research.py` - Main analysis script
- `phase3_sequence_results_20250916_132850.json` - Detailed results data
- `phase3_sequence_analysis_20250916_132850.png` - Performance visualization
- `PHASE3_COMPLETE_REPORT.md` - This comprehensive report

---
*Analysis completed: September 16, 2025*  
*Total scenarios processed: 169/174 (97.1% success rate)*  
*Best performing metric: N-gram Jaccard (F1=0.702, Accuracy=81.2%)*
