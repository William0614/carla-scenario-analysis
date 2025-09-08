#!/usr/bin/env python3
"""
Visualize Phase 1 Distance Metrics Results
Creates comprehensive plots and analysis of the similarity metrics performance.
"""

import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path

def load_results():
    """Load the Phase 1 results from JSON file."""
    results_file = "/home/ads/ads_testing/phase1_distance_metrics_results.json"
    with open(results_file, 'r') as f:
        data = json.load(f)
    return data

def create_performance_dataframe(results):
    """Convert results to pandas DataFrame for analysis."""
    eval_results = results['evaluation_results']
    
    df_data = []
    for result in eval_results:
        metric_parts = result['metric_name'].split('_')
        normalization = metric_parts[0]
        metric = '_'.join(metric_parts[1:])
        
        df_data.append({
            'normalization': normalization,
            'metric': metric,
            'threshold': result['threshold'],
            'accuracy': result['accuracy'],
            'precision': result['precision'],
            'recall': result['recall'],
            'f1_score': result['f1_score'],
            'pearson_correlation': result['pearson_correlation'],
            'spearman_correlation': result['spearman_correlation'],
            'metric_name': result['metric_name']
        })
    
    return pd.DataFrame(df_data)

def plot_performance_heatmap(df, metric_col='f1_score', title_suffix='F1-Score'):
    """Create heatmap of performance metrics."""
    # Pivot table for heatmap
    pivot_df = df.pivot(index='normalization', columns='metric', values=metric_col)
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(pivot_df, annot=True, cmap='viridis', fmt='.3f', 
                cbar_kws={'label': title_suffix})
    plt.title(f'Phase 1 Distance Metrics Performance - {title_suffix}')
    plt.xlabel('Distance Metric')
    plt.ylabel('Normalization Method')
    plt.tight_layout()
    plt.savefig(f'/home/ads/ads_testing/phase1_heatmap_{metric_col}.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_top_performers(df, top_n=10):
    """Plot top performing metric combinations."""
    # Sort by F1-score and get top performers
    top_performers = df.nlargest(top_n, 'f1_score')
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # F1-Score
    bars1 = ax1.bar(range(len(top_performers)), top_performers['f1_score'])
    ax1.set_title('Top 10 F1-Scores')
    ax1.set_ylabel('F1-Score')
    ax1.set_xticks(range(len(top_performers)))
    ax1.set_xticklabels(top_performers['metric_name'], rotation=45, ha='right')
    
    # Add value labels on bars
    for i, bar in enumerate(bars1):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom')
    
    # Accuracy
    bars2 = ax2.bar(range(len(top_performers)), top_performers['accuracy'])
    ax2.set_title('Corresponding Accuracy')
    ax2.set_ylabel('Accuracy')
    ax2.set_xticks(range(len(top_performers)))
    ax2.set_xticklabels(top_performers['metric_name'], rotation=45, ha='right')
    
    # Precision vs Recall
    ax3.scatter(top_performers['recall'], top_performers['precision'], 
               s=100, alpha=0.7, c=top_performers['f1_score'], cmap='viridis')
    ax3.set_xlabel('Recall')
    ax3.set_ylabel('Precision')
    ax3.set_title('Precision vs Recall for Top Performers')
    ax3.grid(True, alpha=0.3)
    
    # Add labels for points
    for i, row in top_performers.iterrows():
        ax3.annotate(row['metric_name'].replace('_', '\n'), 
                    (row['recall'], row['precision']),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=8, alpha=0.8)
    
    # Correlation comparison
    x_pos = np.arange(len(top_performers))
    width = 0.35
    
    bars_pearson = ax4.bar(x_pos - width/2, top_performers['pearson_correlation'], 
                          width, label='Pearson', alpha=0.8)
    bars_spearman = ax4.bar(x_pos + width/2, top_performers['spearman_correlation'], 
                           width, label='Spearman', alpha=0.8)
    
    ax4.set_title('Correlation Coefficients')
    ax4.set_ylabel('Correlation')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(top_performers['metric_name'], rotation=45, ha='right')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/home/ads/ads_testing/phase1_top_performers.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_normalization_comparison(df):
    """Compare performance across normalization methods."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.ravel()
    
    metrics_to_plot = ['f1_score', 'accuracy', 'pearson_correlation', 'spearman_correlation']
    titles = ['F1-Score', 'Accuracy', 'Pearson Correlation', 'Spearman Correlation']
    
    for i, (metric, title) in enumerate(zip(metrics_to_plot, titles)):
        norm_performance = df.groupby('normalization')[metric].agg(['mean', 'std', 'max'])
        
        x_pos = np.arange(len(norm_performance))
        
        # Plot mean with error bars
        axes[i].bar(x_pos, norm_performance['mean'], 
                   yerr=norm_performance['std'], 
                   capsize=5, alpha=0.7, label='Mean ± Std')
        
        # Plot max as points
        axes[i].scatter(x_pos, norm_performance['max'], 
                       color='red', s=50, zorder=5, label='Best')
        
        axes[i].set_title(f'{title} by Normalization Method')
        axes[i].set_ylabel(title)
        axes[i].set_xticks(x_pos)
        axes[i].set_xticklabels(norm_performance.index)
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/home/ads/ads_testing/phase1_normalization_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_metric_comparison(df):
    """Compare performance across distance metrics."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.ravel()
    
    metrics_to_plot = ['f1_score', 'accuracy', 'pearson_correlation', 'spearman_correlation']
    titles = ['F1-Score', 'Accuracy', 'Pearson Correlation', 'Spearman Correlation']
    
    for i, (metric, title) in enumerate(zip(metrics_to_plot, titles)):
        metric_performance = df.groupby('metric')[metric].agg(['mean', 'std', 'max'])
        
        x_pos = np.arange(len(metric_performance))
        
        # Plot mean with error bars
        axes[i].bar(x_pos, metric_performance['mean'], 
                   yerr=metric_performance['std'], 
                   capsize=5, alpha=0.7, label='Mean ± Std')
        
        # Plot max as points
        axes[i].scatter(x_pos, metric_performance['max'], 
                       color='red', s=50, zorder=5, label='Best')
        
        axes[i].set_title(f'{title} by Distance Metric')
        axes[i].set_ylabel(title)
        axes[i].set_xticks(x_pos)
        axes[i].set_xticklabels(metric_performance.index, rotation=45, ha='right')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/home/ads/ads_testing/phase1_metric_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_summary_table(df):
    """Create a summary table of best performers."""
    print("\n" + "="*80)
    print("PHASE 1 DISTANCE METRICS - TOP PERFORMERS SUMMARY")
    print("="*80)
    
    # Best F1-Score
    best_f1 = df.loc[df['f1_score'].idxmax()]
    print(f"\n🏆 BEST F1-SCORE: {best_f1['f1_score']:.3f}")
    print(f"   Metric: {best_f1['metric_name']}")
    print(f"   Accuracy: {best_f1['accuracy']:.3f}")
    print(f"   Precision: {best_f1['precision']:.3f}")
    print(f"   Recall: {best_f1['recall']:.3f}")
    print(f"   Threshold: {best_f1['threshold']}")
    
    # Best Accuracy
    best_acc = df.loc[df['accuracy'].idxmax()]
    print(f"\n🎯 BEST ACCURACY: {best_acc['accuracy']:.3f}")
    print(f"   Metric: {best_acc['metric_name']}")
    print(f"   F1-Score: {best_acc['f1_score']:.3f}")
    print(f"   Precision: {best_acc['precision']:.3f}")
    print(f"   Recall: {best_acc['recall']:.3f}")
    print(f"   Threshold: {best_acc['threshold']}")
    
    # Best Correlation
    best_corr = df.loc[df['pearson_correlation'].idxmax()]
    print(f"\n📊 BEST CORRELATION: {best_corr['pearson_correlation']:.3f}")
    print(f"   Metric: {best_corr['metric_name']}")
    print(f"   F1-Score: {best_corr['f1_score']:.3f}")
    print(f"   Accuracy: {best_corr['accuracy']:.3f}")
    print(f"   Spearman: {best_corr['spearman_correlation']:.3f}")
    print(f"   Threshold: {best_corr['threshold']}")
    
    print("\n" + "="*80)
    print("TOP 5 F1-SCORE PERFORMERS")
    print("="*80)
    top_5 = df.nlargest(5, 'f1_score')[['metric_name', 'f1_score', 'accuracy', 'precision', 'recall', 'threshold']]
    print(top_5.to_string(index=False, float_format='%.3f'))
    
    print("\n" + "="*80)
    print("NORMALIZATION METHOD SUMMARY")
    print("="*80)
    norm_summary = df.groupby('normalization').agg({
        'f1_score': ['mean', 'max'],
        'accuracy': ['mean', 'max'],
        'pearson_correlation': ['mean', 'max']
    }).round(3)
    print(norm_summary)
    
    print("\n" + "="*80)
    print("DISTANCE METRIC SUMMARY")
    print("="*80)
    metric_summary = df.groupby('metric').agg({
        'f1_score': ['mean', 'max'],
        'accuracy': ['mean', 'max'],
        'pearson_correlation': ['mean', 'max']
    }).round(3)
    print(metric_summary)

def main():
    """Main execution function."""
    print("Loading Phase 1 Distance Metrics Results...")
    results = load_results()
    
    print(f"Experiment Info:")
    print(f"  - Phase: {results['experiment_info']['phase']}")
    print(f"  - Timestamp: {results['experiment_info']['timestamp']}")
    print(f"  - Number of scenarios: {results['experiment_info']['num_scenarios']}")
    print(f"  - Metrics tested: {len(results['experiment_info']['metrics_tested'])}")
    print(f"  - Normalization methods: {len(results['experiment_info']['normalization_methods'])}")
    
    # Create DataFrame
    df = create_performance_dataframe(results)
    print(f"\nTotal metric combinations evaluated: {len(df)}")
    
    # Create summary table
    create_summary_table(df)
    
    # Create visualizations
    print("\nCreating visualizations...")
    
    # Performance heatmaps
    plot_performance_heatmap(df, 'f1_score', 'F1-Score')
    plot_performance_heatmap(df, 'accuracy', 'Accuracy')
    plot_performance_heatmap(df, 'pearson_correlation', 'Pearson Correlation')
    
    # Top performers analysis
    plot_top_performers(df)
    
    # Comparison plots
    plot_normalization_comparison(df)
    plot_metric_comparison(df)
    
    print("\nVisualization files saved:")
    print("  - phase1_heatmap_f1_score.png")
    print("  - phase1_heatmap_accuracy.png")
    print("  - phase1_heatmap_pearson_correlation.png")
    print("  - phase1_top_performers.png")
    print("  - phase1_normalization_comparison.png")
    print("  - phase1_metric_comparison.png")
    
    print("\nPhase 1 analysis complete!")

if __name__ == "__main__":
    main()
