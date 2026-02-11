import torch
import torch.nn as nn
import torch.nn.functional as F

class HybridModel(nn.Module):
    def __init__(self, spec_dim, manual_features_dim, num_classes=2):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(2)
        self.lstm = nn.LSTM(64, 64, batch_first=True, bidirectional=True)
        
        conv_output_dim = spec_dim
        for _ in range(2):
            conv_output_dim = conv_output_dim // 2
        self.cnn_fc = nn.Linear(128 * conv_output_dim, 64)
        
        self.feature_fc = nn.Linear(manual_features_dim, 32)
        self.combined_fc1 = nn.Linear(96, 64)
        self.combined_fc2 = nn.Linear(64, num_classes)
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, spec, manual_features):
        x = spec.unsqueeze(1)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.permute(0, 2, 1)
        lstm_out, _ = self.lstm(x)
        lstm_out = lstm_out.reshape(lstm_out.size(0), -1)
        cnn_out = F.relu(self.cnn_fc(lstm_out))
        
        feat_out = F.relu(self.feature_fc(manual_features))
        combined = torch.cat([cnn_out, feat_out], dim=1)
        combined = self.dropout(F.relu(self.combined_fc1(combined)))
        return self.combined_fc2(combined)