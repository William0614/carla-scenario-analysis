# CARLA Scenario Similarity Analysis Results

This directory contains the similarity analysis results for CARLA scenario log files.

## Files

### 1. `gower_similarity_matrix.csv`
- **Description**: Pairwise Gower similarity matrix for all scenarios
- **Range**: 0 to 1 (1 = most similar)
- **Method**: Gower distance calculation combining numerical and categorical features
- **Features used**:
  - **Numerical**: duration, total_distance, mean_speed, max_speed, max_acceleration, max_deceleration, mean_abs_jerk, max_jerk, mean_abs_curvature, max_curvature, num_other_vehicles, traffic_density, accel_events_count, decel_events_count, stop_events_count, turn_events_count
  - **Categorical**: map_name, collision_occurred, traffic_presence

### 2. `sequence_similarity_matrix.csv`
- **Description**: Pairwise sequence (narrative) similarity matrix for all scenarios
- **Range**: 0 to 1 (1 = most similar, 0 = completely different)
- **Method**: Normalized edit distance on symbolic action sequences
- **Symbolic actions**: CR (Cruise), AC (Accelerate), DC (Decelerate), ST (Stop), LT (Left Turn), RT (Right Turn)

## Usage

### Python
```python
import pandas as pd

# Load Gower similarity matrix
gower_sim = pd.read_csv('gower_similarity_matrix.csv', index_col=0)

# Load sequence similarity matrix
seq_sim = pd.read_csv('sequence_similarity_matrix.csv', index_col=0)

# Find most similar scenarios to a target scenario
target = 'ZAM_Tjunction-1_86_I-1-1.log'
similar_gower = gower_sim[target].sort_values(ascending=False).head(10)
similar_sequence = seq_sim[target].sort_values(ascending=False).head(10)

print("Top 10 most similar scenarios (Gower):")
print(similar_gower)

print("\nTop 10 most similar scenarios (Sequence):")
print(similar_sequence)
```

## Interpretation

- **High Gower similarity**: Scenarios have similar overall characteristics (speed, distance, duration, etc.)
- **High sequence similarity**: Scenarios follow similar action patterns over time
- **Combined analysis**: Use both metrics together to find scenarios that are similar in both features and temporal behavior

## Notes

- Each matrix is symmetric (similarity from A to B equals similarity from B to A)
- Diagonal values are 1.0 (each scenario is identical to itself)
- Missing values indicate scenarios where features could not be extracted
