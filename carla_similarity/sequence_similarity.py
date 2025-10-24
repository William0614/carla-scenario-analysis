# sequence_processor.py

import json
import os
import glob
import Levenshtein
import pandas as pd
import numpy as np

def calculate_sequence_similarity_matrix(features_dir: str):
    """
    Loads all action sequences, computes, and returns the edit distance similarity matrix.
    """
    all_sequences = {}
    json_files = glob.glob(os.path.join(features_dir, "*.json"))

    for file_path in json_files:
        filename = os.path.basename(file_path).replace('.json', '.log')
        with open(file_path, 'r') as f:
            data = json.load(f)
            all_sequences[filename] = data['action_sequence']

    if len(all_sequences) < 2:
        print("Not enough feature files to perform a sequence similarity comparison.")
        return None

    print(f"\n--- Calculating Narrative (Edit Distance) Similarity for {len(all_sequences)} scenarios ---")

    scenario_names = list(all_sequences.keys())
    num_scenarios = len(scenario_names)
    similarity_matrix = np.zeros((num_scenarios, num_scenarios))

    for i in range(num_scenarios):
        for j in range(num_scenarios):
            if i == j:
                similarity_matrix[i, j] = 1.0
                continue
            
            seq1 = all_sequences[scenario_names[i]]
            seq2 = all_sequences[scenario_names[j]]
            
            edit_distance = Levenshtein.distance("".join(seq1), "".join(seq2))
            max_len = max(len(seq1), len(seq2))
            
            score = 1.0 - (edit_distance / max_len) if max_len > 0 else 1.0
            similarity_matrix[i, j] = score

    return pd.DataFrame(similarity_matrix, index=scenario_names, columns=scenario_names)