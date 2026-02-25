import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class CNNModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.input_dim = input_dim
        
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(128)
        
        self.pool = nn.MaxPool1d(2)
        self.dropout = nn.Dropout(0.3)
        
        conv_output_size = self._get_conv_output(input_dim)
        
        self.fc1 = nn.Linear(conv_output_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 2)
        
    def _get_conv_output(self, length):
        x = torch.rand(1, 1, length)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        return x.view(1, -1).size(1)
    
    def forward(self, x):
        x = x.unsqueeze(1)
        
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x

class HybridModel(nn.Module):
    def __init__(self, input_dim, cnn_channels=64, lstm_hidden=64):
        super().__init__()
        self.input_dim = input_dim
        
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(32, cnn_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(cnn_channels),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(cnn_channels, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1)
        )
        
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=lstm_hidden,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )
        
        lstm_output_size = lstm_hidden * 2
        
        self.fc = nn.Sequential(
            nn.Linear(lstm_output_size + input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(64, 2)
        )
        
    def forward(self, x):
        batch_size = x.size(0)
        
        cnn_input = x.unsqueeze(1)
        cnn_features = self.cnn(cnn_input)
        cnn_features = cnn_features.squeeze(-1)
        
        lstm_input = cnn_features.unsqueeze(1).repeat(1, 5, 1)
        lstm_out, (hidden, cell) = self.lstm(lstm_input)
        
        lstm_features = torch.cat([hidden[-2], hidden[-1]], dim=1)
        
        combined = torch.cat([lstm_features, x], dim=1)
        
        output = self.fc(combined)
        
        return output

class SimpleCNN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )
    
    def forward(self, x):
        return self.fc(x)

def create_model(model_type, input_dim, **kwargs):
    if model_type == 'cnn':
        return CNNModel(input_dim)
    elif model_type == 'hybrid':
        return HybridModel(input_dim, **kwargs)
    elif model_type == 'simple':
        return SimpleCNN(input_dim)
    else:
        raise ValueError(f"Unknown model type: {model_type}")