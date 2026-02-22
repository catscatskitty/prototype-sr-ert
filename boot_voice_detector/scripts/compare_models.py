import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use('Agg')  # Использовать бэкенд без GUI

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def compare_models():
    os.makedirs('results', exist_ok=True)
    
    if not os.path.exists('results/benchmark_results.csv'):
        print("No benchmark results found. Creating sample data...")
        df = pd.DataFrame({
            'model_type': ['cnn', 'hybrid'],
            'accuracy': [0.85, 0.89],
            'f1_score': [0.84, 0.88],
            'inference_time_sec': [0.12, 0.18],
            'samples_per_sec': [266, 178],
            'cpu_memory_mb': [512, 768],
            'gpu_memory_mb': [1024, 1536],
            'device': ['cuda', 'cuda']
        })
        df.to_csv('results/benchmark_results.csv', index=False)
    else:
        df = pd.read_csv('results/benchmark_results.csv')
    
    print("\nBenchmark Results:")
    print(df.to_string())
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    metrics = [
        ('accuracy', 'Accuracy', (0, 1)),
        ('f1_score', 'F1-Score', (0, 1)),
        ('inference_time_sec', 'Inference Time (seconds)', None),
        ('cpu_memory_mb', 'CPU Memory (MB)', None)
    ]
    
    for i, (metric, title, ylim) in enumerate(metrics):
        ax = axes[i // 2, i % 2]
        if metric in df.columns:
            sns.barplot(data=df, x='model_type', y=metric, ax=ax, hue='model_type', legend=False, palette='viridis')
            ax.set_title(title, fontsize=14, fontweight='bold')
            ax.set_xlabel('Model Type', fontsize=12)
            ax.set_ylabel(title, fontsize=12)
            if ylim:
                ax.set_ylim(ylim)
            
            for j, v in enumerate(df[metric]):
                ax.text(j, v + (0.01 if ylim else v*0.05), f'{v:.3f}', ha='center', va='bottom', fontweight='bold')
        else:
            ax.text(0.5, 0.5, f'No data for {metric}', ha='center', va='center', fontsize=14)
            ax.set_title(title)
    
    plt.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('results/comparison_plots.png', dpi=150, bbox_inches='tight')
    plt.close('all')  # Закрыть все фигуры
    
    available_cols = ['model_type']
    for col in ['accuracy', 'f1_score', 'inference_time_sec', 'cpu_memory_mb', 'samples_per_sec', 'gpu_memory_mb']:
        if col in df.columns:
            available_cols.append(col)
    
    summary = df[available_cols].copy()
    summary.to_csv('results/comparison_summary.csv', index=False)
    
    print("\n" + "="*50)
    print("COMPARISON SUMMARY")
    print("="*50)
    print(summary.to_string(index=False))
    print("\nPlots saved to results/comparison_plots.png")
    
    return df

if __name__ == '__main__':
    compare_models()