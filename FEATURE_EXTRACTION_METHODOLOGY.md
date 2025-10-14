# 🧬 Feature Extraction Methodology

This document provides details on the 32-dimensional feature extraction process used in our CARLA scenario similarity analysis framework.

## 📋 Overview

Feature extraction methodology captures essential characteristics of driving scenarios through five feature categories:

- **Temporal Features (4 dimensions)**: Time-based patterns and event rates
- **Motion Features (8 dimensions)**: Speed, acceleration, and dynamics with unified thresholds
- **Behavioral Features (8 dimensions)**: Driving patterns and maneuvers  
- **Spatial Features (8 dimensions)**: Geographic and geometric characteristics
- **Context Features (4 dimensions)**: Traffic and environmental context

## 🎯 Feature Selection Rationale

1. **Completeness**: Cover all major aspects of driving scenarios
2. **Discriminability**: Provide sufficient granularity to distinguish between scenarios
3. **Robustness**: Remain stable across different simulation conditions
4. **Interpretability**: Allow domain experts to understand similarity judgments
5. **Computational Efficiency**: Enable fast similarity calculations for large datasets

## 🔬 Feature Categories and Detailed Explanation

### 1. Temporal Features (4 Dimensions)

These features capture pure time-based characteristics of driving scenarios, focusing on temporal patterns and event rates.

#### **1.1 Duration (Seconds)**
- **Extraction Method**: `duration = frame_count * 0.05` (assuming 20 FPS)
- **Why Important**: Fundamental scenario characteristic affecting all other metrics
- **Range**: 5-300 seconds depending on scenario type
- **Similarity Rationale**: Similar durations indicate comparable scenario scope and complexity

#### **1.2 Frame Count**
- **Extraction Method**: `frame_count = len(recorded_frames)`
- **Why Important**: Indicates recording resolution and scenario granularity
- **Range**: Varies based on recording frequency (typically 100-3000 frames)
- **Similarity Rationale**: Similar frame counts suggest similar recording conditions

#### **1.3 Event Frequency (Events per Second)**
- **Extraction Method**: `(speed_changes + significant_accelerations) / duration`
- **Why Important**: Normalizes scenario dynamism by time, unified event detection
- **Range**: 0.1-5.0 events per second
- **Similarity Rationale**: Similar event rates indicate comparable temporal dynamics

#### **1.4 Temporal Density (Events per Frame)**
- **Extraction Method**: `total_dynamic_events / frame_count`
- **Why Important**: Measures event concentration relative to recording resolution
- **Range**: 0.01-0.5 events per frame
- **Similarity Rationale**: Similar density indicates comparable recording detail and scenario richness

### 2. Motion Features (8 Dimensions)

These features characterize speed, acceleration, and vehicle dynamics using unified thresholds and statistical measures without redundancy.

#### **2.1 Mean Speed**
- **Extraction Method**: `np.mean(speed_values)`
- **Why Important**: Primary indicator of scenario velocity profile
- **Range**: 1-30 m/s depending on scenario type (urban vs highway)
- **Similarity Rationale**: Similar mean speeds suggest comparable traffic conditions

#### **2.2 Speed Standard Deviation**
- **Extraction Method**: `np.std(speed_values)`
- **Why Important**: Measures speed consistency vs. variability
- **Range**: 0.5-8.0 m/s standard deviation
- **Similarity Rationale**: Similar variability indicates similar driving style consistency

#### **2.3 Minimum Speed**
- **Extraction Method**: `np.min(speed_values)`
- **Why Important**: Identifies lowest speed points (stops, congestion)
- **Range**: 0-10 m/s
- **Similarity Rationale**: Similar minimums indicate comparable traffic constraints

#### **2.4 Maximum Speed**
- **Extraction Method**: `np.max(speed_values)`
- **Why Important**: Identifies peak speeds and speed limits
- **Range**: 5-50 m/s depending on road type
- **Similarity Rationale**: Similar maximums suggest comparable road types

#### **2.5 Speed Range**
- **Extraction Method**: `max_speed - min_speed`
- **Why Important**: More interpretable than IQR, captures full speed variation
- **Range**: 2-40 m/s
- **Similarity Rationale**: Similar ranges indicate comparable speed dynamics

#### **2.6 Mean Acceleration**
- **Extraction Method**: `np.mean(acceleration_values)`
- **Why Important**: Indicates average acceleration tendency
- **Range**: -1 to +1 m/s² (near zero for normal driving)
- **Similarity Rationale**: Similar acceleration patterns suggest comparable driving aggressiveness

#### **2.7 Acceleration Standard Deviation**
- **Extraction Method**: `np.std(acceleration_values)`
- **Why Important**: Measures acceleration smoothness vs. jerkiness
- **Range**: 0.5-3.0 m/s²
- **Similarity Rationale**: Similar acceleration variability indicates comparable driving smoothness

#### **2.8 Dynamic Events Count**
- **Extraction Method**: Count accelerations exceeding ±2.5 m/s² (unified threshold)
- **Why Important**: Identifies significant dynamic maneuvers with consistent threshold
- **Range**: 0-50 dynamic events
- **Similarity Rationale**: Similar event counts indicate comparable maneuver intensity

### 3. Behavioral Features (8 Dimensions)

These features characterize the driving behaviors exhibited during scenarios, focusing on maneuver patterns.

#### **3.1 Stop Events Count**
- **Extraction Method**: Count frames where speed < 0.1 m/s for >1 second
- **Why Important**: Identifies traffic light stops, intersections, traffic jams
- **Range**: 0-20 stops per scenario
- **Similarity Rationale**: Similar stop patterns indicate comparable traffic situations

#### **2.2 Acceleration Events Count**  
- **Extraction Method**: Count sustained acceleration >1.5 m/s² for >0.5s
- **Why Important**: Indicates merging, overtaking, or startup behaviors
- **Range**: 0-30 acceleration events
- **Similarity Rationale**: Similar acceleration patterns suggest comparable driving contexts

#### **3.2 Acceleration Events Count**  
- **Extraction Method**: Count sustained acceleration >1.5 m/s² for >0.5s
- **Why Important**: Indicates merging, overtaking, or startup behaviors
- **Range**: 0-30 acceleration events
- **Similarity Rationale**: Similar acceleration patterns suggest comparable driving contexts

#### **3.3 Deceleration Events Count**
- **Extraction Method**: Count sustained deceleration <-1.5 m/s² for >0.5s  
- **Why Important**: Captures braking behaviors, traffic responses
- **Range**: 0-25 deceleration events
- **Similarity Rationale**: Similar braking patterns indicate comparable traffic density

#### **3.4 Turn Maneuvers Count**
- **Extraction Method**: Count steering angle changes exceeding ±15° sustained >1s
- **Why Important**: Identifies lane changes, turns, navigation complexity
- **Range**: 0-50 turn maneuvers
- **Similarity Rationale**: Similar turn counts suggest comparable route complexity

#### **3.5 Cruise Behavior Count**
- **Extraction Method**: Count sustained constant speed (±0.5 m/s) periods >3s
- **Why Important**: Indicates highway driving or steady traffic flow
- **Range**: 0-15 cruise periods  
- **Similarity Rationale**: Similar cruise patterns suggest comparable traffic flow conditions

#### **3.6 Behavior Transitions Count**
- **Extraction Method**: Count changes between behavioral states (stop→cruise, cruise→turn, etc.)
- **Why Important**: Measures scenario complexity and driving pattern diversity
- **Range**: 5-100 transitions
- **Similarity Rationale**: Similar transition counts indicate comparable scenario complexity

#### **3.7 Average Steering Magnitude**
- **Extraction Method**: `np.mean(np.abs(steering_angles))`
- **Why Important**: Characterizes typical steering intensity
- **Range**: 0.1-0.8 radians average
- **Similarity Rationale**: Similar steering patterns indicate comparable route geometry

#### **3.8 Maximum Steering Magnitude**
- **Extraction Method**: `np.max(np.abs(steering_angles))`
- **Why Important**: Identifies sharp turns or emergency maneuvers
- **Range**: 0.2-1.2 radians maximum
- **Similarity Rationale**: Similar maximum steering suggests comparable maneuver difficulty

### 4. Spatial Features (8 Dimensions)

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

### 5. Context Features (4 Dimensions)

These features characterize the environmental context and scenario complexity, providing essential background information without redundancy.

#### **5.1 Traffic Count**
- **Extraction Method**: Count unique non-ego vehicles detected in scenario
- **Why Important**: Direct measure of traffic density and interaction complexity
- **Range**: 0-50 vehicles
- **Similarity Rationale**: Similar vehicle counts suggest comparable traffic density levels

#### **5.2 Traffic Density**
- **Extraction Method**: `traffic_count / spatial_bounding_box_area`
- **Why Important**: Normalized density measure independent of spatial scale
- **Range**: 0-1.0 vehicles per square meter
- **Similarity Rationale**: Similar density indicates comparable crowding independent of area

#### **5.3 Traffic Presence Indicator**
- **Extraction Method**: Binary indicator (1 if traffic_count > 0, else 0)
- **Why Important**: Simple distinction between isolated vs. multi-vehicle scenarios
- **Range**: 0 (isolated) or 1 (traffic present)
- **Similarity Rationale**: Fundamental categorization for scenario types

#### **5.4 Scenario Complexity Score**
- **Extraction Method**: Combined score from traffic count and spatial complexity
- **Why Important**: Holistic measure of overall scenario difficulty
- **Range**: 0-10 complexity units
- **Similarity Rationale**: Similar complexity scores indicate comparable overall scenario difficulty

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
- **Extraction Time**: ~0.5 seconds per scenario log file
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