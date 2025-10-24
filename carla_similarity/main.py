# main.py

import os
import glob
import json
import numpy as np
import argparse

from feature_extractor import SCTransFeatureExtractor
from gower_similarity import calculate_gower_similarity_matrix
from sequence_similarity import calculate_sequence_similarity_matrix

def run_feature_extraction(log_dir, output_dir):
    """
    Processes all log files in a directory and saves the extracted features.
    """
    print("--- STAGE 1: Feature Extraction ---")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    extractor = SCTransFeatureExtractor()
    log_files = glob.glob(os.path.join(log_dir, "*.log"))
    
    if not log_files:
        print(f"Error: No.log files found in '{log_dir}'. Please check the path.")
        return

    for log_file in log_files:
        # Convert to absolute path for CARLA server
        abs_log_file = os.path.abspath(log_file)
        features = extractor.extract_features(abs_log_file)
        
        if features:
            filename = os.path.basename(log_file)
            output_path = os.path.join(output_dir, filename.replace('.log', '.json'))
            with open(output_path, 'w') as f:
                def convert(o):
                    if isinstance(o, np.generic): return o.item()
                    raise TypeError
                json.dump(features, f, indent=4, default=convert)
            print(f"Saved features to {output_path}")

def run_similarity_analysis(features_dir, output_dir="similarity_results"):
    """
    Runs both Gower and Sequence similarity analyses, prints and saves the results.
    """
    print("\n--- STAGE 2: Similarity Analysis ---")
    
    # Create output directory for similarity results
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Gower Similarity
    gower_similarity_df = calculate_gower_similarity_matrix(features_dir)
    if gower_similarity_df is not None:
        print("\nPairwise Gower Similarity Matrix (1 = most similar):")
        print(gower_similarity_df.round(4))
        
        # Save Gower similarity matrix
        gower_output_csv = os.path.join(output_dir, "gower_similarity_matrix.csv")
        gower_similarity_df.to_csv(gower_output_csv)
        print(f"\nGower similarity matrix saved to: {gower_output_csv}")

    # Sequence Similarity
    sequence_similarity_df = calculate_sequence_similarity_matrix(features_dir)
    if sequence_similarity_df is not None:
        print("\nPairwise Sequence Similarity Matrix (1 = most similar):")
        print(sequence_similarity_df.round(4))
        
        # Save Sequence similarity matrix
        sequence_output_csv = os.path.join(output_dir, "sequence_similarity_matrix.csv")
        sequence_similarity_df.to_csv(sequence_output_csv)
        print(f"\nSequence similarity matrix saved to: {sequence_output_csv}")
    
    return gower_similarity_df, sequence_similarity_df

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run CARLA Scenario Similarity Analysis.")
    parser.add_argument('--log_dir', type=str, required=True, help="Directory containing the SCTrans.log files.")
    parser.add_argument('--features_dir', type=str, default="extracted_features", help="Directory to save/load extracted features.")
    parser.add_argument('--output_dir', type=str, default="similarity_results", help="Directory to save similarity matrices.")
    parser.add_argument('--skip_extraction', action='store_true', help="Skip the feature extraction step and run analysis on existing features.")
    
    args = parser.parse_args()

    if not args.skip_extraction:
        # Ensure CARLA server is running before this step
        run_feature_extraction(args.log_dir, args.features_dir)
    
    if not os.path.exists(args.features_dir):
        print(f"Error: Features directory '{args.features_dir}' not found. Cannot run analysis.")
    else:
        run_similarity_analysis(args.features_dir, args.output_dir)