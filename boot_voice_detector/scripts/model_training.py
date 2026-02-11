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

def train_all_models():
    device_input = input("Select device (1: GPU/CUDA, 2: CPU): ")
    
    if device_input == '2':
        device = torch.device('cpu')
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        print("CPU mode selected")
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if not torch.cuda.is_available():
            print("CUDA not available, falling back to CPU")
            device = torch.device('cpu')
        else:
            print(f"GPU mode selected: {torch.cuda.get_device_name(0)}")
    
    os.makedirs('saved_models', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"CUDA version: {torch.version.cuda}")
    
    print("\nLoading and preparing data...")
    datasets, scalers = load_and_prepare_data()
    
    results = []
    
    for model_type in ['cnn', 'hybrid']:
        print(f"\n{'='*50}")
        print(f"Training {model_type.upper()} model...")
        print(f"{'='*50}")
        
        if model_type == 'cnn':
            model = ModelFactory.create_model('cnn', input_dim=15, num_classes=2)
            train_loader = DataLoader(datasets['cnn']['train'], batch_size=32, shuffle=True, num_workers=0)
            val_loader = DataLoader(datasets['cnn']['val'], batch_size=32, num_workers=0)
            trainer = Trainer(model, device, model_type='cnn')
        else:
            model = ModelFactory.create_model('hybrid', spec_dim=15, manual_features_dim=4, num_classes=2)
            train_loader = DataLoader(datasets['hybrid']['train'], batch_size=32, shuffle=True, num_workers=0)
            val_loader = DataLoader(datasets['hybrid']['val'], batch_size=32, num_workers=0)
            trainer = Trainer(model, device, model_type='hybrid')
        
        start_time = time.time()
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        
        process = psutil.Process()
        mem_before = process.memory_info().rss / 1024 / 1024
        
        history = trainer.train(train_loader, val_loader, epochs=10, lr=0.001)
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
            max_memory = torch.cuda.max_memory_allocated() / 1024 / 1024
        else:
            max_memory = 0
        
        mem_after = process.memory_info().rss / 1024 / 1024
        training_time = time.time() - start_time
        
        torch.save(model.state_dict(), f'saved_models/{model_type}_model.pt')
        
        results.append({
            'model': model_type,
            'device': str(device),
            'train_loss': history['train_loss'][-1],
            'val_acc': history['val_acc'][-1],
            'val_f1': history['val_f1'][-1],
            'training_time': training_time,
            'memory_mb': max_memory if device.type == 'cuda' else mem_after - mem_before,
            'epochs': 10
        })
        
        print(f"\n{model_type.upper()} training completed in {training_time:.2f}s")
        print(f"Final Val Acc: {history['val_acc'][-1]:.4f}, F1: {history['val_f1'][-1]:.4f}")
    
    pd.DataFrame(results).to_csv('results/training_results.csv', index=False)
    print("\n" + "="*50)
    print("TRAINING COMPLETED")
    print("="*50)
    print(results)
    
    return results, history

if __name__ == "__main__":
    train_all_models()