import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import pandas as pd
import numpy as np
import time
import psutil
from src.models.model_factory import ModelFactory
from src.models.predictor import Predictor
from scripts.data_preprocessing import DataPreprocessor
from sklearn.metrics import accuracy_score, f1_score
import gc

def benchmark_models():
    os.makedirs('results', exist_ok=True)
    
    print("="*60)
    print("BENCHMARKING MODELS")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    preprocessor = DataPreprocessor()
    _, _, test_loader, num_classes = preprocessor.prepare_data(batch_size=1)
    
    results = []
    
    for model_type in ['cnn', 'hybrid']:
        print(f"\n{'-'*40}")
        print(f"Benchmarking {model_type.upper()} model...")
        print('-'*40)
        
        model_path = f'saved_models/{model_type}_final.pth'
        if not os.path.exists(model_path):
            print(f"Model {model_path} not found. Skipping...")
            continue
        
        if model_type == 'cnn':
            model = ModelFactory.get_model('cnn', input_dim=14, num_classes=num_classes)
        else:
            model = ModelFactory.get_model('hybrid', cnn_input_dim=14, feature_dim=5, num_classes=num_classes)
        
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        predictor = Predictor(model, device)
        
        process = psutil.Process()
        mem_before = process.memory_info().rss / 1024 / 1024
        
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        
        start_time = time.time()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for i, batch in enumerate(test_loader):
                if model_type == 'hybrid':
                    cnn_input = batch['cnn_input'].to(device)
                    lstm_input = batch['lstm_input'].to(device)
                    hand_features = batch['hand_features'].to(device)
                    outputs = model(cnn_input, lstm_input, hand_features)
                else:
                    inputs = batch['cnn_input'].to(device)
                    outputs = model(inputs)
                
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(batch['label'].numpy())
                
                # Диагностика первых 10 предсказаний
                if i < 10:
                    true_label = batch['label'].numpy()[0]
                    pred_label = preds.cpu().numpy()[0]
                    print(f"  Sample {i}: True={true_label} ({'real' if true_label==1 else 'fake'}), Pred={pred_label} ({'real' if pred_label==1 else 'fake'})")
        
        inference_time = time.time() - start_time
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
            gpu_memory = torch.cuda.max_memory_allocated() / 1024 / 1024
        else:
            gpu_memory = 0
        
        mem_after = process.memory_info().rss / 1024 / 1024
        
        if len(all_preds) > 0 and len(all_labels) > 0:
            accuracy = accuracy_score(all_labels, all_preds)
            f1 = f1_score(all_labels, all_preds, average='weighted')
        else:
            accuracy = 0.0
            f1 = 0.0
        
        results.append({
            'model_type': model_type,
            'accuracy': accuracy,
            'f1_score': f1,
            'inference_time_sec': inference_time,
            'samples_per_sec': len(test_loader.dataset) / inference_time if inference_time > 0 else 0,
            'cpu_memory_mb': mem_after - mem_before,
            'gpu_memory_mb': gpu_memory,
            'device': str(device)
        })
        
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  F1-Score: {f1:.4f}")
        print(f"  Inference Time: {inference_time:.4f}s")
        print(f"  Samples/sec: {len(test_loader.dataset) / inference_time:.2f}")
        print(f"  CPU Memory: {mem_after - mem_before:.2f} MB")
        if gpu_memory > 0:
            print(f"  GPU Memory: {gpu_memory:.2f} MB")
        
        del model, predictor
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    
    df = pd.DataFrame(results)
    df.to_csv('results/benchmark_results.csv', index=False)
    
    print("\n" + "="*60)
    print("BENCHMARK RESULTS")
    print("="*60)
    print(df.to_string(index=False))
    print("\nResults saved to results/benchmark_results.csv")
    
    return df

if __name__ == '__main__':
    benchmark_models()