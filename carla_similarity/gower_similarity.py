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

    categorical_cols = [
        col for col in unified_df.columns 
        if unified_df[col].dtype == 'object' 
        or 'occurred' in col 
        or 'presence' in col
        or '_count' in col
    ]
    categorical_flags = {col: True for col in categorical_cols}
    
    print(f"\n--- Calculating Gower Distance Matrix for {len(unified_df)} scenarios ---")
    print(f"Categorical features identified for Gower: {categorical_cols}")

    gower_dissimilarity = gower.gower_matrix(unified_df, cat_features=categorical_flags)
    gower_similarity = 1 - gower_dissimilarity
    
    return pd.DataFrame(gower_similarity, index=unified_df.index, columns=unified_df.index)