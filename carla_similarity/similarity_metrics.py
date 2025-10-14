#!/usr/bin/env python3
"""
Similarity Metrics Module for CARLA Scenario Analysis

Consolidates all similarity metrics tested across Phase 1, 2, and 3 research.
"""

import numpy as np
import math
from scipy.spatial.distance import euclidean, manhattan, cosine
from scipy.stats import pearsonr, spearmanr
from difflib import SequenceMatcher


class DistanceBasedMetrics:
    """
    Distance-based similarity metrics for 32-dimensional feature vectors.
    Best performer: Cosine Similarity with Z-score normalization.
    """
    
    @staticmethod
    def euclidean_distance(features1, features2):
        """Euclidean distance between two feature vectors."""
        return euclidean(features1, features2)
    
    @staticmethod
    def manhattan_distance(features1, features2):
        """Manhattan (L1) distance between two feature vectors.""" 
        return manhattan(features1, features2)
    
    @staticmethod
    def cosine_similarity(features1, features2):
        """Cosine similarity between two feature vectors."""
        return 1 - cosine(features1, features2)
    
    @staticmethod
    def chebyshev_distance(features1, features2):
        """Chebyshev (L∞) distance between two feature vectors."""
        return max(abs(a - b) for a, b in zip(features1, features2))
    
    @staticmethod
    def minkowski_distance(features1, features2, p=3):
        """Minkowski distance with parameter p."""
        return sum(abs(a - b) ** p for a, b in zip(features1, features2)) ** (1/p)
    
    @staticmethod
    def pearson_correlation(features1, features2):
        """Pearson correlation coefficient."""
        try:
            corr, _ = pearsonr(features1, features2)
            return corr if not np.isnan(corr) else 0
        except:
            return 0
    
    @staticmethod
    def spearman_correlation(features1, features2):
        """Spearman rank correlation coefficient."""
        try:
            corr, _ = spearmanr(features1, features2)
            return corr if not np.isnan(corr) else 0
        except:
            return 0


class SequenceBasedMetrics:
    """
    Sequence-based similarity metrics for behavioral action sequences.
    Best performer: Longest Common Subsequence (LCS).
    """
    
    @staticmethod
    def longest_common_subsequence(seq1, seq2):
        """
        Longest Common Subsequence (LCS) similarity.
        Optimal for behavioral sequence comparison.
        """
        if not seq1 or not seq2:
            return 0.0
        
        # Dynamic programming LCS implementation
        m, n = len(seq1), len(seq2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i-1] == seq2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        
        lcs_length = dp[m][n]
        max_length = max(len(seq1), len(seq2))
        
        return lcs_length / max_length if max_length > 0 else 0.0
    
    @staticmethod
    def edit_distance_similarity(seq1, seq2):
        """Normalized edit distance (Levenshtein) similarity."""
        if not seq1 and not seq2:
            return 1.0
        if not seq1 or not seq2:
            return 0.0
        
        # Dynamic programming edit distance
        m, n = len(seq1), len(seq2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        # Initialize base cases
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
        
        # Fill DP table
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i-1] == seq2[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
        
        edit_dist = dp[m][n]
        max_length = max(len(seq1), len(seq2))
        
        return 1 - (edit_dist / max_length) if max_length > 0 else 0.0
    
    @staticmethod
    def sequence_matcher_similarity(seq1, seq2):
        """Python's SequenceMatcher similarity."""
        return SequenceMatcher(None, seq1, seq2).ratio()
    
    @staticmethod
    def ngram_jaccard_similarity(seq1, seq2, n=2):
        """N-gram Jaccard similarity for sequences."""
        if not seq1 or not seq2:
            return 0.0
        
        # Generate n-grams
        ngrams1 = set(tuple(seq1[i:i+n]) for i in range(len(seq1)-n+1))
        ngrams2 = set(tuple(seq2[i:i+n]) for i in range(len(seq2)-n+1))
        
        if not ngrams1 and not ngrams2:
            return 1.0
        if not ngrams1 or not ngrams2:
            return 0.0
        
        intersection = len(ngrams1.intersection(ngrams2))
        union = len(ngrams1.union(ngrams2))
        
        return intersection / union if union > 0 else 0.0
    
    @staticmethod
    def dtw_similarity(seq1, seq2):
        """Dynamic Time Warping (DTW) similarity."""
        if not seq1 or not seq2:
            return 0.0
        
        m, n = len(seq1), len(seq2)
        dtw = [[float('inf')] * (n + 1) for _ in range(m + 1)]
        dtw[0][0] = 0
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                cost = abs(seq1[i-1] - seq2[j-1]) if isinstance(seq1[i-1], (int, float)) else (0 if seq1[i-1] == seq2[j-1] else 1)
                dtw[i][j] = cost + min(dtw[i-1][j], dtw[i][j-1], dtw[i-1][j-1])
        
        max_length = max(len(seq1), len(seq2))
        normalized_dtw = dtw[m][n] / max_length if max_length > 0 else 0
        
        return max(0, 1 - normalized_dtw)


class SetBasedMetrics:
    """
    Set-based similarity metrics for categorical feature sets.
    Best performer: Jaccard Index.
    """
    
    @staticmethod
    def jaccard_similarity(set1, set2):
        """Jaccard similarity coefficient."""
        if not set1 and not set2:
            return 1.0
        if not set1 or not set2:
            return 0.0
        
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        return intersection / union if union > 0 else 0.0
    
    @staticmethod
    def dice_similarity(set1, set2):
        """Dice similarity coefficient."""
        if not set1 and not set2:
            return 1.0
        if not set1 or not set2:
            return 0.0
        
        intersection = len(set1.intersection(set2))
        total_elements = len(set1) + len(set2)
        
        return (2 * intersection) / total_elements if total_elements > 0 else 0.0
    
    @staticmethod
    def overlap_similarity(set1, set2):
        """Overlap coefficient."""
        if not set1 and not set2:
            return 1.0
        if not set1 or not set2:
            return 0.0
        
        intersection = len(set1.intersection(set2))
        min_size = min(len(set1), len(set2))
        
        return intersection / min_size if min_size > 0 else 0.0
    
    @staticmethod
    def cosine_set_similarity(set1, set2):
        """Cosine similarity for sets."""
        if not set1 and not set2:
            return 1.0
        if not set1 or not set2:
            return 0.0
        
        intersection = len(set1.intersection(set2))
        magnitude_product = math.sqrt(len(set1) * len(set2))
        
        return intersection / magnitude_product if magnitude_product > 0 else 0.0


class NormalizationUtils:
    """Utilities for feature vector normalization."""
    
    @staticmethod
    def z_score_normalize(features_dict):
        """Z-score normalization of feature vectors."""
        if not features_dict:
            return features_dict
        
        # Collect all feature vectors
        all_vectors = [f['combined_vector'] for f in features_dict.values() if f and 'combined_vector' in f]
        if not all_vectors:
            return features_dict
        
        # Calculate means and stds for each dimension
        feature_matrix = np.array(all_vectors)
        means = np.mean(feature_matrix, axis=0)
        stds = np.std(feature_matrix, axis=0)
        
        # Avoid division by zero
        stds = np.where(stds == 0, 1, stds)
        
        # Normalize each vector
        normalized_dict = {}
        for name, features in features_dict.items():
            if features and 'combined_vector' in features:
                normalized_vector = (np.array(features['combined_vector']) - means) / stds
                normalized_features = features.copy()
                normalized_features['combined_vector'] = normalized_vector.tolist()
                normalized_dict[name] = normalized_features
            else:
                normalized_dict[name] = features
        
        return normalized_dict
    
    @staticmethod
    def min_max_normalize(features_dict):
        """Min-max normalization to [0,1] range."""
        if not features_dict:
            return features_dict
        
        # Collect all feature vectors
        all_vectors = [f['combined_vector'] for f in features_dict.values() if f and 'combined_vector' in f]
        if not all_vectors:
            return features_dict
        
        # Calculate min and max for each dimension
        feature_matrix = np.array(all_vectors)
        mins = np.min(feature_matrix, axis=0)
        maxs = np.max(feature_matrix, axis=0)
        
        # Avoid division by zero
        ranges = maxs - mins
        ranges = np.where(ranges == 0, 1, ranges)
        
        # Normalize each vector
        normalized_dict = {}
        for name, features in features_dict.items():
            if features and 'combined_vector' in features:
                normalized_vector = (np.array(features['combined_vector']) - mins) / ranges
                normalized_features = features.copy()
                normalized_features['combined_vector'] = normalized_vector.tolist()
                normalized_dict[name] = normalized_features
            else:
                normalized_dict[name] = features
        
        return normalized_dict