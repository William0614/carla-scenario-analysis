#!/usr/bin/env python3
"""
Example script for analyzing saved similarity matrices.
"""

import pandas as pd
import numpy as np
import os

def load_similarity_matrices(results_dir='similarity_results'):
    """Load both similarity matrices from CSV files."""
    gower_path = os.path.join(results_dir, 'gower_similarity_matrix.csv')
    sequence_path = os.path.join(results_dir, 'sequence_similarity_matrix.csv')
    
    gower_sim = pd.read_csv(gower_path, index_col=0)
    seq_sim = pd.read_csv(sequence_path, index_col=0)
    
    return gower_sim, seq_sim

def find_most_similar(target_scenario, gower_sim, seq_sim, top_n=10):
    """Find the most similar scenarios to a target scenario."""
    print(f"\n=== Most Similar Scenarios to: {target_scenario} ===\n")
    
    # Gower similarity
    print(f"Top {top_n} by Gower Similarity (holistic features):")
    gower_similar = gower_sim[target_scenario].sort_values(ascending=False).head(top_n + 1)
    for i, (scenario, score) in enumerate(gower_similar.items(), 1):
        if scenario != target_scenario:
            print(f"  {i-1}. {scenario}: {score:.4f}")
    
    # Sequence similarity
    print(f"\nTop {top_n} by Sequence Similarity (action patterns):")
    seq_similar = seq_sim[target_scenario].sort_values(ascending=False).head(top_n + 1)
    for i, (scenario, score) in enumerate(seq_similar.items(), 1):
        if scenario != target_scenario:
            print(f"  {i-1}. {scenario}: {score:.4f}")

def find_combined_similar(target_scenario, gower_sim, seq_sim, top_n=10, 
                         gower_weight=0.5, seq_weight=0.5):
    """
    Find similar scenarios using a weighted combination of both metrics.
    
    Args:
        gower_weight: Weight for Gower similarity (0-1)
        seq_weight: Weight for sequence similarity (0-1)
    """
    # Both metrics are now in 0-1 range, so direct weighted combination
    combined = gower_weight * gower_sim[target_scenario] + seq_weight * seq_sim[target_scenario]
    
    print(f"\n=== Combined Similarity (Gower: {gower_weight}, Seq: {seq_weight}) ===")
    print(f"Target: {target_scenario}\n")
    
    combined_similar = combined.sort_values(ascending=False).head(top_n + 1)
    for i, (scenario, score) in enumerate(combined_similar.items(), 1):
        if scenario != target_scenario:
            g_score = gower_sim.loc[target_scenario, scenario]
            s_score = seq_sim.loc[target_scenario, scenario]
            print(f"  {i-1}. {scenario}")
            print(f"      Combined: {score:.4f}, Gower: {g_score:.4f}, Seq: {s_score:.4f}")

def get_statistics(gower_sim, seq_sim):
    """Get basic statistics about the similarity matrices."""
    print("\n=== Similarity Matrix Statistics ===\n")
    
    # Get lower triangle (excluding diagonal) for unique pairs
    gower_vals = gower_sim.values[np.tril_indices_from(gower_sim.values, -1)]
    seq_vals = seq_sim.values[np.tril_indices_from(seq_sim.values, -1)]
    
    print("Gower Similarity:")
    print(f"  Mean: {gower_vals.mean():.4f}")
    print(f"  Std:  {gower_vals.std():.4f}")
    print(f"  Min:  {gower_vals.min():.4f}")
    print(f"  Max:  {gower_vals.max():.4f}")
    
    print("\nSequence Similarity:")
    print(f"  Mean: {seq_vals.mean():.4f}")
    print(f"  Std:  {seq_vals.std():.4f}")
    print(f"  Min:  {seq_vals.min():.4f}")
    print(f"  Max:  {seq_vals.max():.4f}")

def main():
    # Load matrices
    print("Loading similarity matrices...")
    gower_sim, seq_sim = load_similarity_matrices()
    print(f"Loaded matrices: {gower_sim.shape[0]} scenarios")
    
    # Get statistics
    get_statistics(gower_sim, seq_sim)
    
    # Example: Find similar scenarios
    example_scenario = gower_sim.index[0]  # Use first scenario as example
    find_most_similar(example_scenario, gower_sim, seq_sim, top_n=5)
    
    # Example: Combined similarity
    find_combined_similar(example_scenario, gower_sim, seq_sim, top_n=5,
                         gower_weight=0.7, seq_weight=0.3)

if __name__ == '__main__':
    main()
