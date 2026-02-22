import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from src.models.model_factory import ModelFactory
from src.models.trainer import Trainer
from scripts.data_preprocessing import DataPreprocessor
import gc
import argparse

class ModelTrainer:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        if self.device.type == 'cuda':
            print(f"GPU: {torch.cuda.get_device_name(0)}")
        self.preprocessor = DataPreprocessor()
        self.train_loader, self.val_loader, self.test_loader, self.num_classes = self.preprocessor.prepare_data(batch_size=64)

    def train_model(self, model_type, fixed_epochs=None):
        if model_type == 'cnn':
            model = ModelFactory.get_model('cnn', input_dim=14, num_classes=self.num_classes)
            trainer = Trainer(model, self.device, learning_rate=0.0005, weight_decay=1e-4, patience=10)
        else:
            model = ModelFactory.get_model('hybrid', cnn_input_dim=14, feature_dim=5, num_classes=self.num_classes)
            trainer = Trainer(model, self.device, learning_rate=0.001, weight_decay=1e-5, patience=15)
        
        if fixed_epochs:
            trainer = Trainer(model, self.device, learning_rate=0.001, patience=None)
            print(f"\nTraining {model_type.upper()} for {fixed_epochs} fixed epochs...")
            epochs_completed = trainer.train_fixed(self.train_loader, fixed_epochs)
        else:
            trainer = Trainer(model, self.device, learning_rate=0.001, patience=15)
            print(f"\nTraining {model_type.upper()} until convergence...")
            epochs_completed = trainer.train(self.train_loader, self.val_loader, max_epochs=200)
        
        accuracy, f1 = trainer.evaluate(self.test_loader)
        
        results = {
            'model_type': model_type,
            'accuracy': accuracy,
            'f1_score': f1,
            'epochs': epochs_completed,
            'device': str(self.device)
        }
        
        os.makedirs('saved_models', exist_ok=True)
        torch.save(model.state_dict(), f'saved_models/{model_type}_final.pth')
        print(f"Model saved after {epochs_completed} epochs")
        print(f"Test Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
        
        del model, trainer
        gc.collect()
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
        
        return results

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fixed_epochs', type=int, help='Train with fixed number of epochs')
    parser.add_argument('--model', type=str, default='cnn', choices=['cnn', 'hybrid'])
    args = parser.parse_args()
    
    trainer = ModelTrainer()
    print(f"\n{'='*50}")
    results = trainer.train_model(args.model, fixed_epochs=args.fixed_epochs)
    print('='*50)