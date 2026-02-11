import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def compare_models():
    results = pd.read_csv('results/training_results.csv')
    benchmark = pd.read_csv('results/benchmark_results.csv')
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    sns.barplot(data=results, x='model', y='val_acc', ax=axes[0,0])
    axes[0,0].set_title('Validation Accuracy')
    axes[0,0].set_ylim(0, 1)
    
    sns.barplot(data=results, x='model', y='val_f1', ax=axes[0,1])
    axes[0,1].set_title('Validation F1-Score')
    axes[0,1].set_ylim(0, 1)
    
    sns.barplot(data=results, x='model', y='training_time', ax=axes[0,2])
    axes[0,2].set_title('Training Time (seconds)')
    
    sns.barplot(data=results, x='model', y='memory_mb', ax=axes[1,0])
    axes[1,0].set_title('Peak Memory Usage (MB)')
    
    sns.barplot(data=benchmark[benchmark['batch_size']==32], x='model', y='samples_per_sec', ax=axes[1,1])
    axes[1,1].set_title('Inference Speed (samples/sec)')
    
    sns.barplot(data=benchmark[benchmark['batch_size']==32], x='model', y='inference_time', ax=axes[1,2])
    axes[1,2].set_title('Inference Time (seconds)')
    
    plt.tight_layout()
    plt.savefig('results/model_comparison.png', dpi=150)
    
    comparison_table = results[['model', 'val_acc', 'val_f1', 'training_time', 'memory_mb']]
    comparison_table.to_csv('results/comparison_summary.csv', index=False)
    
    print("\n" + "="*50)
    print("MODEL COMPARISON SUMMARY")
    print("="*50)
    print(comparison_table.to_string(index=False))
    
    return comparison_table

if __name__ == "__main__":
    compare_models()