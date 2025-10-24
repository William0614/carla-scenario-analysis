# gower_processor.py

import pandas as pd
import json
import os
import glob
import gower

def calculate_gower_similarity_matrix(features_dir: str):
    """
    Loads all feature files, computes, and returns the Gower similarity matrix.
    """
    all_features = {}
    json_files = glob.glob(os.path.join(features_dir, "*.json"))

    for file_path in json_files:
        filename = os.path.basename(file_path).replace('.json', '.log')
        with open(file_path, 'r') as f:
            data = json.load(f)
            all_features[filename] = data['holistic_features']

    if len(all_features) < 2:
        print("Not enough feature files to perform a Gower similarity comparison.")
        return None

    unified_df = pd.DataFrame.from_dict(all_features, orient='index')
    unified_df = unified_df.reindex(sorted(unified_df.columns), axis=1)

    # Identify truly categorical columns (only map_name and binary flags)
    categorical_cols = [
        col for col in unified_df.columns 
        if unified_df[col].dtype == 'object'  # map_name
        or col in ['collision_occurred', 'traffic_presence']  # binary flags only
    ]
    
    # Remove features with zero variance (they don't help distinguish scenarios)
    zero_var_cols = [col for col in unified_df.columns 
                     if unified_df[col].dtype in ['float64', 'int64'] 
                     and unified_df[col].std() == 0]
    
    if zero_var_cols:
        print(f"Removing zero-variance features: {zero_var_cols}")
        unified_df = unified_df.drop(columns=zero_var_cols)
    
    # Convert categorical columns to category dtype for proper handling
    for col in categorical_cols:
        if col in unified_df.columns:  # Make sure it wasn't dropped
            unified_df[col] = unified_df[col].astype('category')
    
    # Create boolean array for categorical features (required by gower package)
    cat_features = [col in categorical_cols for col in unified_df.columns]
    
    print(f"\n--- Calculating Gower Distance Matrix for {len(unified_df)} scenarios ---")
    print(f"Categorical features identified for Gower: {[col for col in unified_df.columns if col in categorical_cols]}")
    print(f"Numerical features: {[col for col in unified_df.columns if col not in categorical_cols]}")

    gower_dissimilarity = gower.gower_matrix(unified_df, cat_features=cat_features)
    gower_similarity = 1 - gower_dissimilarity
    
    return pd.DataFrame(gower_similarity, index=unified_df.index, columns=unified_df.index)