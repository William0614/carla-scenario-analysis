# Phase 2: Set-Based Jaccard Similarity Analysis

**Date:** September 16, 2025  
**Phase:** 2 - Set-Based Similarity Metrics  
**Method:** Jaccard Coefficient on Categorical Feature Sets  

## Quick Results

✅ **Successfully implemented Jaccard coefficient analysis**  
✅ **Best F1 Score: 0.598** (threshold=0.5)  
✅ **Best Accuracy: 61.4%**  
✅ **15 unique categorical features extracted**  
✅ **29.0% similarity rate** (consistent with Phase 1)  

## Performance Summary

| Metric | Phase 1 (Best) | Phase 2 (Best) | Difference |
|--------|----------------|----------------|------------|
| F1 Score | 0.644 | 0.598 | -0.046 |
| Accuracy | 81.5% | 61.4% | -20.1% |
| Method | Z-Score + Minkowski | Jaccard(0.5) | - |

## Key Features Identified

Most common categorical features across 174 scenarios:

1. `traffic_scenario` (99.4%)
2. `speed_peaks` (89.7%) 
3. `turning_scenario` (81.0%)
4. `sharp_maneuvers` (79.9%)
5. `frequent_stopping` (79.3%)

## Files Generated

- `phase2_set_based_jaccard_research_fixed.py` - Main analysis script
- `phase2_jaccard_results_20250916_112654.json` - Detailed results data
- `phase2_jaccard_analysis_20250916_112654.png` - Visualization plots
- `PHASE2_REPORT.md` - Comprehensive analysis report

## Conclusions

Phase 2 demonstrates that set-based similarity can effectively identify scenario redundancy, though with lower performance than distance-based methods. The 4.6-point F1 difference suggests continuous features capture more nuanced similarities than categorical sets.

**Next:** Phase 3 - Sequence-based similarity metrics

---
*CARLA Scenario Similarity Research - Phase 2 Complete*
