import torch
import numpy as np
import pandas as pd
import time
import psutil
import gc
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from torch.utils.data import DataLoader
from src.models.model_factory import ModelFactory
from src.models.trainer import Trainer
from scripts.data_preprocessing import load_and_prepare_data

def benchmark_models():
    device_input = input("Select device for benchmark (1: GPU/CUDA, 2: CPU): ")
    
    if device_input == '2':
        device = torch.device('cpu')
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        print("CPU mode selected")
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if not torch.cuda.is_available():
            print("CUDA not available, falling back to CPU")
            device = torch.device('cpu')
    
    datasets, _ = load_and_prepare_data()
    
    benchmark_results = []
    
    for model_type in ['cnn', 'hybrid']:
        for batch_size in [16, 32, 64]:
            if model_type == 'cnn':
                model = ModelFactory.create_model('cnn', input_dim=15, num_classes=2)
                loader = DataLoader(datasets['cnn']['val'], batch_size=batch_size)
            else:
                model = ModelFactory.create_model('hybrid', spec_dim=15, manual_features_dim=4, num_classes=2)
                loader = DataLoader(datasets['hybrid']['val'], batch_size=batch_size)
            
            model = model.to(device)
            model.eval()
            
            if device.type == 'cuda':
                torch.cuda.reset_peak_memory_stats()
            
            start_time = time.time()
            with torch.no_grad():
                for batch in loader:
                    if len(batch) == 3:
                        x1, x2, _ = batch
                        x1 = x1.to(device)
                        x2 = x2.to(device)
                        _ = model(x1, x2)
                    else:
                        x, _ = batch
                        x = x.to(device)
                        _ = model(x)
            
            if device.type == 'cuda':
                torch.cuda.synchronize()
                memory = torch.cuda.max_memory_allocated() / 1024 / 1024
            else:
                memory = 0
            
            inference_time = time.time() - start_time
            
            benchmark_results.append({
                'model': model_type,
                'batch_size': batch_size,
                'inference_time': inference_time,
                'samples_per_sec': len(loader.dataset) / inference_time,
                'memory_mb': memory,
                'device': str(device)
            })
    
    pd.DataFrame(benchmark_results).to_csv('results/benchmark_results.csv', index=False)
    print("\nBenchmark completed. Results saved.")
    return benchmark_results

if __name__ == "__main__":
    benchmark_models()