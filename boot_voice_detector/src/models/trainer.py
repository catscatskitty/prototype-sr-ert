import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

class ModelTrainer:
    def __init__(self, model, device, lr=0.001):
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', patience=5)
        self.history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
        
    def train_epoch(self, loader):
        self.model.train()
        losses = []
        preds = []
        labels = []
        
        for inputs, targets in loader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()
            
            losses.append(loss.item())
            preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
            labels.extend(targets.cpu().numpy())
        
        return np.mean(losses), accuracy_score(labels, preds), f1_score(labels, preds, average='weighted')
    
    def validate(self, loader):
        self.model.eval()
        losses = []
        preds = []
        labels = []
        
        with torch.no_grad():
            for inputs, targets in loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
                losses.append(loss.item())
                preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
                labels.extend(targets.cpu().numpy())
        
        return np.mean(losses), accuracy_score(labels, preds), f1_score(labels, preds, average='weighted')
    
    def train(self, train_loader, val_loader, epochs=100, patience=10):
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            train_loss, train_acc, train_f1 = self.train_epoch(train_loader)
            val_loss, val_acc, val_f1 = self.validate(val_loader)
            
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            
            self.scheduler.step(val_loss)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), 'saved_models/best_model.pth')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f'Early stopping at epoch {epoch+1}')
                    break
        
        return self.history