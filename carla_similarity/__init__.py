"""
CARLA Scenario Similarity Analysis Framework

A comprehensive toolkit for analyzing similarity between CARLA simulation scenarios
using multi-dimensional feature extraction and various similarity metrics.

This package consolidates research from Phase 1 (distance-based), Phase 2 (set-based),
and Phase 3 (sequence-based) similarity analysis approaches with both basic and
multi-dimensional ground truth validation.
"""

__version__ = "1.2.0"
__author__ = "CARLA Research Team"

# Core components
try:
    from .feature_extraction import FeatureExtractor
    from .similarity_metrics import (
        DistanceBasedMetrics,
        SequenceBasedMetrics, 
        SetBasedMetrics,
        NormalizationUtils
    )
    from .ground_truth import (
        BasicGroundTruth,
        MultiDimensionalGroundTruth
    )
    from .evaluation import SimilarityEvaluator
    
    __all__ = [
        'FeatureExtractor',
        'DistanceBasedMetrics',
        'SequenceBasedMetrics',
        'SetBasedMetrics',
        'NormalizationUtils',
        'BasicGroundTruth',
        'MultiDimensionalGroundTruth',
        'SimilarityEvaluator'
    ]
    
except ImportError as e:
    print(f"Warning: Some dependencies may not be available: {e}")
    print("Please ensure CARLA and scenario_runner are properly installed.")
    
    # Provide minimal functionality even without full dependencies
    __all__ = []