# 37-Dimensional Feature Extraction Methodology

**Comprehensive Guide to CARLA Scenario Feature Engineering**

## 📊 Overview

This document provides a detailed explanation of the 37-dimensional feature vector used in the CARLA Scenario Similarity Analysis Framework. These features were carefully selected and engineered to capture the essential characteristics of autonomous driving scenarios for similarity analysis and redundancy detection.

## 🎯 Feature Selection Philosophy

The 37 dimensions were chosen based on the following principles:

1. **Completeness**: Cover all major aspects of driving scenarios
2. **Discriminability**: Provide sufficient granularity to distinguish between scenarios
3. **Robustness**: Remain stable across different simulation conditions
4. **Interpretability**: Allow domain experts to understand similarity judgments
5. **Computational Efficiency**: Enable fast similarity calculations for large datasets

## 🔬 Feature Categories and Detailed Explanation

### 1. Temporal Features (6 Dimensions)

These features capture the time-based characteristics of scenarios, essential for understanding scenario duration and timing patterns.

#### **1.1 Duration (seconds)**
- **Extraction Method**: `total_time = max_timestamp - min_timestamp`
- **Why Important**: Scenarios with similar durations often represent similar complexity levels
- **Range**: Typically 20-300 seconds for CARLA scenarios
- **Similarity Rationale**: Similar duration scenarios likely have comparable event density

#### **1.2 Frame Count**
- **Extraction Method**: `frame_count = len(recorded_frames)`
- **Why Important**: Indicates recording resolution and scenario granularity
- **Range**: Varies based on recording frequency (typically 100-3000 frames)
- **Similarity Rationale**: Similar frame counts suggest similar recording conditions

#### **1.3 Speed Changes Count**
- **Extraction Method**: Count velocity magnitude changes exceeding threshold (0.5 m/s)
- **Why Important**: Measures scenario dynamism and complexity
- **Range**: 0-200+ changes depending on scenario complexity
- **Similarity Rationale**: Scenarios with similar speed change patterns indicate comparable driving complexity

#### **1.4 Speed Change Ratio**
- **Extraction Method**: `speed_changes / duration`
- **Why Important**: Normalizes dynamism by scenario length
- **Range**: 0.1-5.0 changes per second
- **Similarity Rationale**: Indicates consistent event density across different scenario durations

#### **1.5 Speed Variability (Standard Deviation)**
- **Extraction Method**: `np.std(speed_values)`
- **Why Important**: Captures consistency vs. variability in vehicle behavior
- **Range**: 0.5-8.0 m/s standard deviation
- **Similarity Rationale**: Similar variability indicates similar driving style consistency

#### **1.6 Significant Accelerations Count**
- **Extraction Method**: Count acceleration changes exceeding ±2.0 m/s²
- **Why Important**: Identifies emergency or aggressive maneuvers
- **Range**: 0-50 significant accelerations
- **Similarity Rationale**: Similar acceleration patterns indicate comparable driving aggressiveness

### 2. Behavioral Features (10 Dimensions)

These features characterize the driving behaviors exhibited during scenarios, crucial for understanding driving patterns and maneuver types.

#### **2.1 Stop Events Count**
- **Extraction Method**: Count frames where speed < 0.1 m/s for >1 second
- **Why Important**: Identifies traffic light stops, intersections, traffic jams
- **Range**: 0-20 stops per scenario
- **Similarity Rationale**: Similar stop patterns indicate comparable traffic situations

#### **2.2 Acceleration Events Count**  
- **Extraction Method**: Count sustained acceleration >1.5 m/s² for >0.5s
- **Why Important**: Indicates merging, overtaking, or startup behaviors
- **Range**: 0-30 acceleration events
- **Similarity Rationale**: Similar acceleration patterns suggest comparable driving contexts

#### **2.3 Deceleration Events Count**
- **Extraction Method**: Count sustained deceleration <-1.5 m/s² for >0.5s  
- **Why Important**: Captures braking behaviors, traffic responses
- **Range**: 0-25 deceleration events
- **Similarity Rationale**: Similar braking patterns indicate comparable traffic density

#### **2.4 Turn Maneuvers Count**
- **Extraction Method**: Count steering angle changes exceeding ±15° sustained >1s
- **Why Important**: Identifies lane changes, turns, navigation complexity
- **Range**: 0-50 turn maneuvers
- **Similarity Rationale**: Similar turn counts suggest comparable route complexity

#### **2.5 Cruise Behavior Count**
- **Extraction Method**: Count sustained constant speed (±0.5 m/s) periods >3s
- **Why Important**: Indicates highway driving or steady traffic flow
- **Range**: 0-15 cruise periods  
- **Similarity Rationale**: Similar cruise patterns suggest comparable traffic flow conditions

#### **2.6 Behavior Transitions Count**
- **Extraction Method**: Count changes between behavioral states (stop→cruise, cruise→turn, etc.)
- **Why Important**: Measures scenario complexity and driving pattern diversity
- **Range**: 5-100 transitions
- **Similarity Rationale**: Similar transition counts indicate comparable scenario complexity

#### **2.7 Unique Behaviors Count**
- **Extraction Method**: Count distinct behavior types observed (max 10 types)
- **Why Important**: Indicates scenario diversity and completeness
- **Range**: 1-10 unique behaviors
- **Similarity Rationale**: Similar behavior diversity suggests comparable scenario richness

#### **2.8 Average Steering Magnitude**
- **Extraction Method**: `np.mean(np.abs(steering_angles))`
- **Why Important**: Characterizes typical steering intensity
- **Range**: 0.1-0.8 radians average
- **Similarity Rationale**: Similar steering patterns indicate comparable route geometry

#### **2.9 Maximum Steering Magnitude**
- **Extraction Method**: `np.max(np.abs(steering_angles))`
- **Why Important**: Identifies sharp turns or emergency maneuvers
- **Range**: 0.2-1.2 radians maximum
- **Similarity Rationale**: Similar maximum steering suggests comparable maneuver difficulty

#### **2.10 Total Behavior Events Count**
- **Extraction Method**: Sum of all detected behavior events
- **Why Important**: Overall measure of scenario activity level
- **Range**: 10-200 total events
- **Similarity Rationale**: Similar event totals indicate comparable overall scenario activity

### 3. Spatial Features (8 Dimensions)

These features capture the geometric and spatial characteristics of vehicle trajectories, essential for understanding route complexity and spatial patterns.

#### **3.1 Total Path Length**
- **Extraction Method**: Sum of Euclidean distances between consecutive positions
- **Why Important**: Indicates route length and travel distance
- **Range**: 100-5000 meters depending on scenario
- **Similarity Rationale**: Similar path lengths suggest comparable journey scope

#### **3.2 Total Displacement**
- **Extraction Method**: Euclidean distance between start and end positions
- **Why Important**: Measures net progress vs. path complexity
- **Range**: 50-3000 meters
- **Similarity Rationale**: Similar displacements indicate comparable journey outcomes

#### **3.3 Path Tortuosity Ratio**
- **Extraction Method**: `total_path_length / total_displacement`
- **Why Important**: Quantifies route straightness vs. complexity (1.0 = straight line)
- **Range**: 1.0-5.0+ (higher = more winding)
- **Similarity Rationale**: Similar tortuosity indicates comparable route geometry complexity

#### **3.4 Bounding Box Width**
- **Extraction Method**: `max(x_positions) - min(x_positions)`
- **Why Important**: Measures lateral space utilization
- **Range**: 10-1000 meters
- **Similarity Rationale**: Similar spatial extents suggest comparable spatial complexity

#### **3.5 Bounding Box Height**
- **Extraction Method**: `max(y_positions) - min(y_positions)`
- **Why Important**: Measures longitudinal space utilization
- **Range**: 10-1000 meters  
- **Similarity Rationale**: Similar spatial extents suggest comparable spatial coverage

#### **3.6 Bounding Box Area**
- **Extraction Method**: `bounding_box_width * bounding_box_height`
- **Why Important**: Overall spatial footprint of the scenario
- **Range**: 100-1,000,000 square meters
- **Similarity Rationale**: Similar areas indicate comparable spatial scope

#### **3.7 Direction Changes Count**
- **Extraction Method**: Count heading angle changes exceeding ±10° between consecutive frames
- **Why Important**: Measures route complexity and navigation difficulty
- **Range**: 10-500 direction changes
- **Similarity Rationale**: Similar direction change patterns indicate comparable route complexity

#### **3.8 Curvature Density**
- **Extraction Method**: `direction_changes_count / total_path_length`
- **Why Important**: Normalizes route complexity by distance traveled
- **Range**: 0.001-0.1 changes per meter
- **Similarity Rationale**: Similar curvature density indicates comparable route geometry complexity

### 4. Speed Features (10 Dimensions)

These features provide detailed statistical analysis of speed patterns, crucial for understanding traffic conditions and driving dynamics.

#### **4.1 Mean Speed**
- **Extraction Method**: `np.mean(speed_values)`
- **Why Important**: Central tendency of speed distribution
- **Range**: 1-25 m/s (typical urban/highway speeds)
- **Similarity Rationale**: Similar average speeds suggest comparable traffic conditions

#### **4.2 Speed Standard Deviation**
- **Extraction Method**: `np.std(speed_values)`
- **Why Important**: Variability in speed patterns
- **Range**: 0.5-8.0 m/s
- **Similarity Rationale**: Similar speed variability indicates comparable traffic flow consistency

#### **4.3 Minimum Speed**
- **Extraction Method**: `np.min(speed_values)`
- **Why Important**: Identifies stops or slowest conditions
- **Range**: 0-5 m/s
- **Similarity Rationale**: Similar minimum speeds suggest comparable traffic congestion

#### **4.4 Maximum Speed**
- **Extraction Method**: `np.max(speed_values)`
- **Why Important**: Peak speed capabilities and road type indicators
- **Range**: 5-35 m/s  
- **Similarity Rationale**: Similar maximum speeds indicate comparable road types (urban vs highway)

#### **4.5 Median Speed**
- **Extraction Method**: `np.median(speed_values)`
- **Why Important**: Robust central tendency less affected by outliers
- **Range**: 1-20 m/s
- **Similarity Rationale**: Similar median speeds indicate comparable typical driving conditions

#### **4.6 Speed Interquartile Range (IQR)**
- **Extraction Method**: `np.percentile(speeds, 75) - np.percentile(speeds, 25)`
- **Why Important**: Robust measure of speed distribution spread
- **Range**: 1-10 m/s
- **Similarity Rationale**: Similar IQR indicates comparable speed distribution patterns

#### **4.7 Mean Acceleration**
- **Extraction Method**: `np.mean(acceleration_values)`
- **Why Important**: Characterizes typical acceleration/deceleration patterns
- **Range**: -2.0 to +2.0 m/s² typical range
- **Similarity Rationale**: Similar acceleration patterns indicate comparable driving aggressiveness

#### **4.8 Acceleration Standard Deviation**
- **Extraction Method**: `np.std(acceleration_values)`
- **Why Important**: Measures variability in acceleration patterns
- **Range**: 0.5-4.0 m/s² standard deviation
- **Similarity Rationale**: Similar acceleration variability indicates comparable driving smoothness

#### **4.9 High Speed Events Count**
- **Extraction Method**: `len([s for s in speeds if s > 10])` (count of speeds > 10 m/s)
- **Why Important**: Identifies highway driving or high-speed segments
- **Range**: 0-1000+ events depending on scenario type
- **Similarity Rationale**: Similar high-speed event counts indicate comparable road types (highway vs urban)

#### **4.10 Hard Acceleration/Deceleration Count**
- **Extraction Method**: `len([a for a in accelerations if abs(a) > 3])` (count of |acceleration| > 3 m/s²)
- **Why Important**: Identifies emergency braking, aggressive acceleration, or abrupt maneuvers
- **Range**: 0-50+ events depending on scenario complexity
- **Similarity Rationale**: Similar hard acceleration counts indicate comparable driving aggressiveness or emergency situations

### 5. Traffic Features (3 Dimensions)

These features characterize the traffic environment and vehicle interactions, essential for understanding scenario context and complexity.

#### **5.1 Traffic Vehicle Count**
- **Extraction Method**: Count unique non-ego vehicles detected in scenario
- **Why Important**: Measures traffic density and interaction complexity
- **Range**: 0-50 vehicles
- **Similarity Rationale**: Similar vehicle counts suggest comparable traffic density levels

#### **5.2 Capped Traffic Complexity**
- **Extraction Method**: Complex interaction analysis capped at maximum threshold
- **Why Important**: Prevents outlier scenarios from dominating similarity calculations
- **Range**: 0-100 complexity score
- **Similarity Rationale**: Similar complexity scores indicate comparable interaction difficulty

#### **5.3 Traffic Presence Indicator**
- **Extraction Method**: Binary indicator (1 if traffic_count > 0, else 0)
- **Why Important**: Simple distinction between isolated vs. multi-vehicle scenarios
- **Range**: 0 (isolated) or 1 (traffic present)
- **Similarity Rationale**: Fundamental categorization for scenario types

## 🧮 Feature Engineering Principles

### Normalization Strategy
- **Temporal features**: Normalized by scenario duration where appropriate
- **Spatial features**: Normalized by path length or area where appropriate  
- **Speed features**: Use robust statistics (percentiles) to handle outliers
- **Count features**: Raw counts preserved to maintain interpretability

### Robustness Measures
- **Outlier handling**: Percentile-based features resist extreme values
- **Missing data**: Graceful degradation with default values for corrupted logs
- **Scale invariance**: Ratios and normalized metrics reduce absolute scale dependency

### Domain Knowledge Integration
- **Thresholds**: Speed and acceleration thresholds based on real driving patterns
- **Time windows**: Behavior detection windows matched to human reaction times
- **Categories**: Feature groupings aligned with traffic engineering principles

## 📈 Feature Validation

### Statistical Properties
- **Distribution analysis**: Each feature exhibits reasonable distribution across dataset
- **Correlation analysis**: Features show expected correlations (e.g., speed features)
- **Discriminative power**: Each feature contributes to scenario differentiation

### Domain Expert Validation
- **Interpretability**: Features align with traffic engineering concepts
- **Completeness**: Cover major aspects of driving scenario characterization
- **Practicality**: Enable actionable insights for scenario optimization

## 🔍 Usage in Similarity Analysis

### Distance-Based Metrics
- Features normalized and weighted equally in Euclidean/Cosine similarity
- Demonstrated effectiveness: Cosine similarity F1-score of 0.567

### Sequence-Based Metrics  
- Temporal and behavioral features inform action sequence generation
- Support LCS and DTW algorithms for behavioral pattern matching

### Set-Based Metrics
- Categorical transformations enable Jaccard coefficient calculations
- Feature thresholding creates meaningful categorical boundaries

## 📊 Performance Impact

### Computational Efficiency
- **Extraction Time**: ~0.1 seconds per scenario log file
- **Memory Usage**: ~37 floats (148 bytes) per scenario
- **Scalability**: Linear scaling with dataset size

### Similarity Quality
- **Best Overall Performance**: F1-score of 0.702 with N-gram Jaccard
- **Robust Discrimination**: Consistent performance across scenario types
- **Interpretable Results**: Feature analysis enables understanding of similarity judgments

## 🚀 Future Extensions

### Potential Additional Features
- **Weather conditions**: Rain, fog, lighting conditions
- **Road geometry**: Intersection types, lane configurations
- **Traffic lights**: Signal timing and compliance patterns
- **Pedestrian interactions**: Pedestrian density and crossing events

### Advanced Processing
- **Deep learning embeddings**: Learned representations from raw sensor data
- **Temporal sequences**: Time-series feature extraction methods
- **Multi-scale analysis**: Features at different temporal resolutions

---

**Note**: This feature extraction methodology represents a balance between computational efficiency, interpretability, and discriminative power. The 37-dimensional feature vector provides a comprehensive yet manageable representation of CARLA driving scenarios suitable for large-scale similarity analysis and scenario optimization applications.