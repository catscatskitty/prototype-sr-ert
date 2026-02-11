import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

class Trainer:
    def __init__(self, model, device, model_type='cnn'):
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        self.model_type = model_type
        
    def train_epoch(self, dataloader, optimizer):
        self.model.train()
        total_loss = 0
        batches = 0
        for batch in dataloader:
            if self.model_type == 'hybrid' and len(batch) == 3:
                x_spec, x_feat, y = batch
                x_spec = x_spec.to(self.device)
                x_feat = x_feat.to(self.device)
                y = y.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(x_spec, x_feat)
            else:
                x, y = batch
                x = x.to(self.device)
                y = y.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(x)
            
            loss = self.criterion(outputs, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            batches += 1
        return total_loss / batches
    
    def evaluate(self, dataloader):
        self.model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for batch in dataloader:
                if self.model_type == 'hybrid' and len(batch) == 3:
                    x_spec, x_feat, y = batch
                    x_spec = x_spec.to(self.device)
                    x_feat = x_feat.to(self.device)
                    outputs = self.model(x_spec, x_feat)
                else:
                    x, y = batch
                    x = x.to(self.device)
                    outputs = self.model(x)
                
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y.numpy())
        
        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='weighted')
        return acc, f1
    
    def train(self, train_loader, val_loader, epochs=10, lr=0.001):
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        history = {'train_loss': [], 'val_acc': [], 'val_f1': []}
        
        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader, optimizer)
            val_acc, val_f1 = self.evaluate(val_loader)
            history['train_loss'].append(train_loss)
            history['val_acc'].append(val_acc)
            history['val_f1'].append(val_f1)
            print(f"Epoch {epoch+1}/{epochs} - Loss: {train_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}")
            
        return history