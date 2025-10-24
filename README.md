# CARLA Scenario Similarity Analysis

Framework for analyzing and comparing CARLA driving scenarios from log files, specifically for datasets like SCTrans. The methodology is grounded in a multi-modal approach, representing each scenario in two distinct ways to capture different aspects of similarity:

1. **Holistic Similarity**: A fixed-length feature vector that provides a scenario's overall characteristics. This is analyzed using Gower's Distance, which is ideal for mixed numerical and categorical data.

2. **Narrative Similarity**: A symbolic sequence of actions that represents the driving behavior. This is analyzed using Levenshtein (Edit) Distance to compare how similarly two scenarios unfold over time.

---

## How It Works

### Stage 1: Feature Extraction (`feature_extractor.py`)

This is the primary data processing step. The script iterates through a directory of CARLA `.log` files and performs the following for each file:

1. **Connects to CARLA Server**: Uses a running CARLA server's powerful log parsing capabilities
2. **Parses the Log**: Uses the `srunner.metrics.tools.MetricsLog` class to robustly parse the log file and automatically identify the ego vehicle using a safe fallback mechanism (`_get_ego_vehicle_id_safe`)
3. **Builds a DataFrame**: Constructs a clean, time-indexed `pandas.DataFrame` for the ego vehicle, containing all its dynamic states (position, velocity, acceleration, yaw) for every frame
4. **Extracts Features**: Calculates the two feature representations: the holistic vector and the symbolic action sequence
5. **Saves Output**: The extracted features for each log are saved as a `.json` file in the `extracted_features/` directory

### Stage 2: Similarity Analysis

Once the features are extracted, two separate processor scripts analyze the generated `.json` files:

#### Holistic Similarity (`gower_similarity.py`)
- Loads the `holistic_features` vector from all `.json` files into a single DataFrame
- Computes a complete pairwise similarity matrix using Gower's Distance
- Result: Value between 0 (completely different) and 1 (identical) for every pair of scenarios
- Saves results to `similarity_results/gower_similarity_matrix.csv`

#### Narrative Similarity (`sequence_similarity.py`)
- Loads the `action_sequence` from all `.json` files
- Computes a complete pairwise similarity matrix using normalized Levenshtein (Edit) Distance
- Result: Similarity score between 0 (completely different) and 1 (identical) for every pair of scenarios
- Saves results to `similarity_results/sequence_similarity_matrix.csv`

---

## Feature Set

The framework extracts two distinct sets of features for each scenario.

### 1. Holistic Feature Vector

This is a 19-dimensional vector containing a mix of continuous and categorical/count data that summarizes the entire scenario.

| Category | Feature Name | Unit | Description |
|----------|-------------|------|-------------|
| **Temporal** | `duration` | seconds | The total duration of the ego vehicle's involvement in the scenario |
| **Kinematics** | `total_distance` | meters | Total distance traveled by the ego vehicle |
| | `mean_speed` | m/s | Average speed of the ego vehicle over the scenario |
| | `max_speed` | m/s | Maximum speed achieved by the ego vehicle |
| | `max_acceleration` | m/s² | Maximum longitudinal acceleration |
| | `max_deceleration` | m/s² | Maximum longitudinal deceleration (as a positive value) |
| **Behavioral** | `mean_abs_jerk` | m/s³ | Mean of the absolute magnitude of the jerk vector, indicating overall driving smoothness |
| | `max_jerk` | m/s³ | Maximum magnitude of the jerk vector, capturing the most abrupt maneuver |
| | `mean_abs_curvature` | 1/m | Mean of the absolute path curvature, indicating the average turning intensity |
| | `max_curvature` | 1/m | Maximum path curvature, identifying the sharpest turn |
| **Contextual** | `map_name` | string | The name of the CARLA map where the scenario took place |
| | `num_other_vehicles` | count | The number of unique non-ego vehicles present during the scenario |
| | `traffic_density` | vehicles/m² | Number of other vehicles divided by the area of the ego-vehicle's spatial bounding box |
| **Event Counts** | `collision_occurred` | 0 or 1 | A binary flag indicating if the ego vehicle was involved in a collision |
| | `traffic_presence` | 0 or 1 | A binary flag indicating if any other vehicles were present |
| | `stop_events_count` | count | The number of times the vehicle transitioned into a 'Stop' state (speed < 0.5 m/s) |
| | `accel_events_count` | count | The number of times the vehicle initiated a significant acceleration event (> 1.5 m/s²) |
| | `decel_events_count` | count | The number of times the vehicle initiated a significant deceleration event (< -1.5 m/s²) |
| | `turn_events_count` | count | The number of times the vehicle initiated a significant turning maneuver (curvature > 0.005) |

### 2. Symbolic Action Sequence

This feature represents the narrative of the scenario as a compressed sequence of driving states.

| Symbol | State | Description |
|--------|-------|-------------|
| `ST` | Stop | Vehicle speed is below 0.5 m/s |
| `AC` | Accelerate | Longitudinal acceleration is greater than 1.5 m/s² |
| `DC` | Decelerate | Longitudinal acceleration is less than -1.5 m/s² |
| `CR` | Cruise | The vehicle is moving but not in any other defined state |
| `LT` | Left Turn | The vehicle is executing a significant left turn (based on curvature and yaw rate) |
| `RT` | Right Turn | The vehicle is executing a significant right turn (based on curvature and yaw rate) |

**Example Sequence:** `['CR', 'AC', 'CR', 'LT', 'CR', 'DC', 'ST']`

---

## How to Run

### Prerequisites

1. A running CARLA server (the feature extractor needs to connect to it to parse the log files)
2. Python 3.7+
3. A local installation of ScenarioRunner (version 0.9.15 is used in this codebase)

### Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/William0614/carla-scenario-analysis.git
   cd carla-scenario-analysis
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure scenario_runner Path:**
   
   This is a critical step. Open `feature_extractor.py` and update the following line to point to your local scenario_runner installation directory:
   
   ```python
   # feature_extractor.py
   # ...
   # Example: sys.path.append('/home/user/scenario_runner')
   sys.path.append('/home/ads/CARLA_0.9.15/scenario_runner/scenario_runner-0.9.15')
   # ...
   ```

4. **Prepare Data:**
   
   Place your CARLA `.log` files (e.g., from the SCTrans dataset) into a directory. For this guide, we assume they are in a folder named `log_files/`.

### Execution

#### Start CARLA Server (in background without GUI)

```bash
cd ~/CARLA_0.9.15
./CarlaUE4.sh -RenderOffScreen &
```

#### Run Full Pipeline (Extraction + Analysis)

From the project root directory, run the following command. This will first process all `.log` files in the specified directory and save the features to `extracted_features/`. It will then automatically run both similarity analyses on the generated files.

```bash
python carla_similarity/main.py --log_dir log_files
```

#### Run Analysis Only

If you have already extracted the features and just want to re-run the similarity analysis, use the `--skip_extraction` flag. This is much faster.

```bash
python carla_similarity/main.py --log_dir log_files --skip_extraction
```

#### Analyze Results

Use the provided analysis script to explore the similarity matrices:

```bash
python analyze_similarity.py
```

### Command Line Options

```
python carla_similarity/main.py --log_dir <path> [options]

Required:
  --log_dir PATH          Directory containing .log files

Optional:
  --features_dir PATH     Directory for extracted features (default: extracted_features)
  --output_dir PATH       Directory for similarity matrices (default: similarity_results)
  --skip_extraction       Skip feature extraction, use existing features
```

---

## Output

The script generates the following output:

### Feature Files (`extracted_features/`)
- One JSON file per log file containing extracted features
- Each file contains both holistic features and symbolic action sequences

### Similarity Matrices (`similarity_results/`)
- `gower_similarity_matrix.csv`: Holistic feature similarity (range: 0-1)
- `sequence_similarity_matrix.csv`: Action sequence similarity (range: 0-1)
- `README.md`: Documentation about the results

Both similarity matrices are also printed to the console during execution.

---

## Example Analysis

```python
import pandas as pd

# Load similarity matrices
gower = pd.read_csv('similarity_results/gower_similarity_matrix.csv', index_col=0)
sequence = pd.read_csv('similarity_results/sequence_similarity_matrix.csv', index_col=0)

# Find top 10 most similar scenarios to a target
target = 'ZAM_Tjunction-1_86_I-1-1.log'
similar = gower[target].sort_values(ascending=False).head(11)
print(similar)

# Find scenarios with high similarity in both metrics
high_gower = gower[target] > 0.95
high_seq = sequence[target] > 0.5
both_high = high_gower & high_seq
print(gower[target][both_high])
```

---

## Project Structure

```
carla-scenario-analysis/
├── carla_similarity/
│   ├── __init__.py
│   ├── feature_extractor.py      # Stage 1: Feature extraction
│   ├── gower_similarity.py       # Stage 2: Gower similarity
│   ├── sequence_similarity.py    # Stage 2: Sequence similarity
│   └── main.py                   # Pipeline orchestrator
├── log_files/                    # Input: CARLA log files
├── extracted_features/           # Output: Extracted features (JSON)
├── similarity_results/           # Output: Similarity matrices (CSV)
├── analyze_similarity.py         # Helper script for analysis
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

---

## License

This project is licensed under the MIT License.

---

## Acknowledgments

- CARLA Simulator
- ScenarioRunner
- SCTrans Dataset
 