# feature_extractor.py

import os
import sys
import carla
import math
import numpy as np
import pandas as pd
import re
import json
import glob
from scipy.signal import savgol_filter

# --- IMPORTANT ---
# Add the path to your local ScenarioRunner installation.
# This is required to import the MetricsLog class.
# Example: sys.path.append('/home/user/scenario_runner')
sys.path.append('/home/ads/CARLA_0.9.15/scenario_runner/scenario_runner-0.9.15')
try:
    from srunner.metrics.tools.metrics_log import MetricsLog
except ImportError:
    print("Error: Could not import MetricsLog.")
    print("Please ensure the path to your scenario_runner installation is in sys.path.")
    sys.exit(1)

class SCTransFeatureExtractor:
    """
    Extracts a multi-modal feature representation from existing CARLA scenario logs.
    """

    def __init__(self, carla_host='localhost', carla_port=2000):
        """Initializes the extractor and connects to the CARLA client."""
        print("Connecting to CARLA server to enable log parsing...")
        self.client = carla.Client(carla_host, carla_port)
        self.client.set_timeout(30.0)
        print("Connection successful.")

    def _get_ego_vehicle_id_safe(self, log: MetricsLog):
        """
        Safely retrieves the ego vehicle ID with fallback logic.
        First tries 'hero', then 'ego', then falls back to first vehicle found.
        """
        # Try standard role names
        for role_name in ['hero', 'ego', 'ego_vehicle']:
            ego_ids = log.get_actor_ids_with_role_name(role_name)
            if ego_ids:
                print(f"Found ego vehicle with role_name '{role_name}': ID {ego_ids[0]}")
                return ego_ids[0]
        
        # Fallback: get the first vehicle in the log
        vehicle_ids = log.get_actor_ids_with_type_id("vehicle.*")
        if vehicle_ids:
            print(f"Warning: No standard ego role found. Using first vehicle: ID {vehicle_ids[0]}")
            return vehicle_ids[0]
        
        return None

    def extract_features(self, log_path: str):
        """
        Main method to extract all feature representations from a single log file.
        """
        print(f"\nProcessing log file: {os.path.basename(log_path)}...")
        try:
            recorder_str = self.client.show_recorder_file_info(log_path, show_all=True)
            if not recorder_str:
                print(f"Warning: Log file '{log_path}' is empty or could not be read.")
                return None

            log = MetricsLog(recorder_str)
            
            # Try to get ego vehicle ID with fallback logic
            ego_id = self._get_ego_vehicle_id_safe(log)
            if ego_id is None:
                print(f"Error: Could not find ego vehicle in {log_path}.")
                return None
                
            start_frame, end_frame = log.get_actor_alive_frames(ego_id)

            if ego_id is None or start_frame is None:
                print(f"Error: Could not determine ego vehicle or frame data from {log_path}.")
                return None

            ego_df = self._build_actor_dataframe(log, ego_id, start_frame, end_frame)
            map_name = self._get_map_name_from_log_header(recorder_str)

            holistic_features = self._extract_holistic_features(ego_df, log, map_name, start_frame, end_frame, ego_id)
            action_sequence = self._generate_symbolic_sequence(ego_df)

            print("Feature extraction successful.")
            return {
                "holistic_features": holistic_features,
                "action_sequence": action_sequence
            }

        except Exception as e:
            print(f"An error occurred while processing {log_path}: {e}")
            return None

    def _build_actor_dataframe(self, log: MetricsLog, actor_id: int, start_frame: int, end_frame: int) -> pd.DataFrame:
        """Constructs a time-indexed pandas DataFrame for a single actor."""
        records = []
        for frame in range(start_frame, end_frame + 1):
            try:
                transform = log.get_actor_transform(actor_id, frame)
                velocity = log.get_actor_velocity(actor_id, frame)
                acceleration = log.get_actor_acceleration(actor_id, frame)
                timestamp = log.get_elapsed_time(frame)

                records.append({
                    'time': timestamp,
                    'pos_x': transform.location.x, 'pos_y': transform.location.y,
                    'yaw': transform.rotation.yaw,
                    'vel_x': velocity.x, 'vel_y': velocity.y,
                    'acc_x': acceleration.x, 'acc_y': acceleration.y,
                })
            except Exception:
                continue
        
        df = pd.DataFrame(records)
        return df.sort_values(by='time').set_index('time') if not df.empty else df

    def _get_map_name_from_log_header(self, log_string: str) -> str:
        """Extracts the map name from the header of the log string."""
        match = re.search(r"Map: (\w+)", log_string)
        return match.group(1) if match else "Unknown"

    def _extract_holistic_features(self, ego_df: pd.DataFrame, log: MetricsLog, map_name: str, start_frame: int, end_frame: int, ego_id: int) -> dict:
        """Extracts the unified vector of continuous and categorical features."""
        if ego_df.empty: return {}

        ego_df['speed'] = np.sqrt(ego_df['vel_x']**2 + ego_df['vel_y']**2)
        ego_df['long_accel'] = np.gradient(ego_df['speed'], ego_df.index)
        accel_mag = np.sqrt(ego_df['acc_x']**2 + ego_df['acc_y']**2)
        
        # Calculate jerk with proper handling
        if len(accel_mag) >= 5:
            smooth_accel = savgol_filter(accel_mag, window_length=min(5, len(accel_mag)), polyorder=2)
            jerk_mag = np.gradient(smooth_accel, ego_df.index)
        else:
            # For short sequences, use simple gradient
            jerk_mag = np.gradient(accel_mag, ego_df.index)
        
        curvature = self._calculate_curvature(ego_df['pos_x'], ego_df['pos_y'])
        # Clip extreme curvature values to reasonable range (max 10 rad/m)
        curvature = np.clip(curvature, -10, 10)
        ego_df['curvature'] = curvature
        # Use unwrap without period parameter for compatibility with older NumPy versions
        ego_df['yaw_rate'] = np.gradient(np.unwrap(np.radians(ego_df['yaw'])), ego_df.index)

        features = {}
        features['map_name'] = map_name
        
        # A. Temporal Features
        try:
            start_time = log.get_elapsed_time(start_frame)
            end_time = log.get_elapsed_time(end_frame)
            duration = end_time - start_time
        except (IndexError, TypeError, AttributeError):
            # Fallback: use the time index from the dataframe
            if not ego_df.empty and len(ego_df.index) > 0:
                duration = ego_df.index[-1] - ego_df.index[0]
            else:
                duration = 0.0
        features['duration'] = duration

        # B. Spatial and Kinematic Features
        positions = ego_df[['pos_x', 'pos_y']].to_numpy()
        features['total_distance'] = np.sum(np.sqrt(np.sum(np.diff(positions, axis=0)**2, axis=1)))
        features['mean_speed'] = ego_df['speed'].mean()
        features['max_speed'] = ego_df['speed'].max()
        features['max_acceleration'] = ego_df['long_accel'].max()
        features['max_deceleration'] = abs(ego_df['long_accel'].min())

        # C. Ego Behavioral Features
        features['mean_abs_jerk'] = np.mean(np.abs(jerk_mag))
        features['max_jerk'] = np.max(np.abs(jerk_mag))
        features['mean_abs_curvature'] = np.mean(np.abs(curvature))
        features['max_curvature'] = np.max(np.abs(curvature))

        # D. Traffic Context Features
        vehicle_ids = log.get_actor_ids_with_type_id("vehicle.*")
        features['num_other_vehicles'] = len([vid for vid in vehicle_ids if vid != ego_id])
        xs, ys = ego_df['pos_x'], ego_df['pos_y']
        bbox_area = (xs.max() - xs.min()) * (ys.max() - ys.min())
        features['traffic_density'] = features['num_other_vehicles'] / bbox_area if bbox_area > 1 else 0

        # E. Event-Based Features
        features['collision_occurred'] = 1 if log.get_actor_collisions(ego_id) else 0
        features['traffic_presence'] = 1 if features['num_other_vehicles'] > 0 else 0
        
        stop_events = (ego_df['speed'] < 0.5).astype(int).diff().eq(1).sum()
        accel_events = (ego_df['long_accel'] > 1.5).astype(int).diff().eq(1).sum()
        decel_events = (ego_df['long_accel'] < -1.5).astype(int).diff().eq(1).sum()
        turn_events = (ego_df['curvature'] > 0.005).astype(int).diff().eq(1).sum()

        features['stop_events_count'] = stop_events
        features['accel_events_count'] = accel_events
        features['decel_events_count'] = decel_events
        features['turn_events_count'] = turn_events
        
        return features

    def _generate_symbolic_sequence(self, ego_df: pd.DataFrame) -> list:
        """Generates a compressed symbolic sequence of actions, including turns."""
        states = []
        for _, row in ego_df.iterrows():
            if row['speed'] < 0.5: state = 'ST'
            elif row['long_accel'] > 1.5: state = 'AC'
            elif row['long_accel'] < -1.5: state = 'DC'
            elif row['curvature'] > 0.005:
                if row['yaw_rate'] > 1.0: state = 'LT'
                elif row['yaw_rate'] < -1.0: state = 'RT'
                else: state = 'CR'
            else: state = 'CR'
            
            if not states or states[-1]!= state:
                states.append(state)
        return states

    def _calculate_curvature(self, x: pd.Series, y: pd.Series) -> np.ndarray:
        """Calculates the curvature of a 2D path."""
        dx_dt = np.gradient(x, x.index)
        dy_dt = np.gradient(y, y.index)
        d2x_dt2 = np.gradient(dx_dt, x.index)
        d2y_dt2 = np.gradient(dy_dt, y.index)
        numerator = np.abs(dx_dt * d2y_dt2 - dy_dt * d2x_dt2)
        denominator = (dx_dt**2 + dy_dt**2)**1.5
        return np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator!=0)