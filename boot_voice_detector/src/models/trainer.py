import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import accuracy_score, f1_score

class Trainer:
    def __init__(self, model, device, learning_rate=0.001, weight_decay=1e-5, patience=15):
        self.model = model
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=5)
        self.scaler = GradScaler() if device.type == 'cuda' else None
        self.patience = patience

    def train_epoch(self, dataloader):
        self.model.train()
        total_loss = 0
        
        for batch in dataloader:
            self.optimizer.zero_grad()
            
            if 'hybrid' in str(type(self.model)).lower():
                cnn_input = batch['cnn_input'].to(self.device, non_blocking=True)
                lstm_input = batch['lstm_input'].to(self.device, non_blocking=True)
                hand_features = batch['hand_features'].to(self.device, non_blocking=True)
                labels = batch['label'].to(self.device, non_blocking=True)
                
                if self.scaler:
                    with autocast():
                        outputs = self.model(cnn_input, lstm_input, hand_features)
                        loss = self.criterion(outputs, labels)
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    outputs = self.model(cnn_input, lstm_input, hand_features)
                    loss = self.criterion(outputs, labels)
                    loss.backward()
                    self.optimizer.step()
            else:
                inputs = batch['cnn_input'].to(self.device, non_blocking=True)
                labels = batch['label'].to(self.device, non_blocking=True)
                
                if self.scaler:
                    with autocast():
                        outputs = self.model(inputs)
                        loss = self.criterion(outputs, labels)
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)
                    loss.backward()
                    self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(dataloader)

    def train_fixed(self, train_loader, epochs):
        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            if (epoch + 1) % 10 == 0:
                print(f'Epoch {epoch+1}/{epochs}, Loss: {train_loss:.4f}')
        return epochs

    def train(self, train_loader, val_loader, max_epochs=200):
        best_loss = float('inf')
        counter = 0
        best_model_state = None
        
        for epoch in range(max_epochs):
            train_loss = self.train_epoch(train_loader)
            
            val_loss = 0
            self.model.eval()
            with torch.no_grad():
                for batch in val_loader:
                    if 'hybrid' in str(type(self.model)).lower():
                        cnn_input = batch['cnn_input'].to(self.device)
                        lstm_input = batch['lstm_input'].to(self.device)
                        hand_features = batch['hand_features'].to(self.device)
                        labels = batch['label'].to(self.device)
                        outputs = self.model(cnn_input, lstm_input, hand_features)
                    else:
                        inputs = batch['cnn_input'].to(self.device)
                        labels = batch['label'].to(self.device)
                        outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)
                    val_loss += loss.item()
            val_loss /= len(val_loader)
            
            self.scheduler.step(val_loss)
            
            if val_loss < best_loss:
                best_loss = val_loss
                best_model_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                counter = 0
            else:
                counter += 1
            
            if (epoch + 1) % 10 == 0:
                print(f'Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, LR: {self.optimizer.param_groups[0]["lr"]:.6f}')
            
            if self.patience and counter >= self.patience:
                print(f'Early stopping at epoch {epoch+1}')
                break
        
        if best_model_state:
            self.model.load_state_dict({k: v.to(self.device) for k, v in best_model_state.items()})
        
        return epoch + 1

    def evaluate(self, dataloader):
        self.model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in dataloader:
                if 'hybrid' in str(type(self.model)).lower():
                    cnn_input = batch['cnn_input'].to(self.device)
                    lstm_input = batch['lstm_input'].to(self.device)
                    hand_features = batch['hand_features'].to(self.device)
                    outputs = self.model(cnn_input, lstm_input, hand_features)
                else:
                    inputs = batch['cnn_input'].to(self.device)
                    outputs = self.model(inputs)
                
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(batch['label'].numpy())
        
        return accuracy_score(all_labels, all_preds), f1_score(all_labels, all_preds, average='weighted')