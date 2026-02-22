import torch
import torch.nn as nn
import torch.nn.functional as F

class HybridModel(nn.Module):
    def __init__(self, cnn_input_dim=14, feature_dim=5, num_classes=2):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool = nn.MaxPool1d(2)
        
        self.lstm = nn.LSTM(feature_dim, 128, batch_first=True, bidirectional=True, dropout=0.3, num_layers=2)
        
        cnn_output_size = 128 * (cnn_input_dim // 4)
        self.fc_cnn = nn.Linear(cnn_output_size, 256)
        self.fc_combined = nn.Linear(256 + 256 + feature_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc_out = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(0.4)

    def forward(self, cnn_input, lstm_input, hand_features):
        cnn_x = cnn_input.unsqueeze(1)
        cnn_x = self.pool(F.relu(self.bn1(self.conv1(cnn_x))))
        cnn_x = self.pool(F.relu(self.bn2(self.conv2(cnn_x))))
        cnn_x = cnn_x.view(cnn_x.size(0), -1)
        cnn_features = F.relu(self.fc_cnn(cnn_x))
        cnn_features = self.dropout(cnn_features)

        lstm_out, _ = self.lstm(lstm_input)
        lstm_features = lstm_out[:, -1, :]

        combined = torch.cat([cnn_features, lstm_features, hand_features], dim=1)
        combined = F.relu(self.fc_combined(combined))
        combined = self.dropout(combined)
        combined = F.relu(self.fc2(combined))
        combined = self.dropout(combined)
        return self.fc_out(combined)