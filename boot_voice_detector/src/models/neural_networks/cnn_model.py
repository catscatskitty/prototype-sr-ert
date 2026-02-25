# src/models/neural_networks/cnn_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN1D(nn.Module):
    def __init__(self, input_dim, num_classes=1):
        super(CNN1D, self).__init__()
        
        # Convolutional layers with max pooling and dropout for regularization
        self.conv_block = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )

        self.conv2_block = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )

        self.conv3_block = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )

        # Flatten output of convolutional layers
        self._to_linear = None
        self._get_conv_output(input_dim)
        
        # Dense layers with ReLU activations and sigmoid output for binary classification
        self.fc1 = nn.Linear(self._to_linear, 128)
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(128, num_classes)
        
    def _get_conv_output(self, input_dim):
        with torch.no_grad():
            dummy = torch.zeros(1, 1, input_dim)
            dummy = self.conv_block(dummy)
            dummy = self.conv2_block(dummy)
            dummy = self.conv3_block(dummy)
            self._to_linear = dummy.view(1, -1).size(1)
    
    def forward(self, x):
        # Input preparation (unsqueeze for batch dimension and convolutional layers)
        x = x.unsqueeze(1)

        # Convolutional block
        x = self.conv_block(x)
        x = self.conv2_block(x)
        x = self.conv3_block(x)

        # Flatten output of convolutional layers
        x = x.view(x.size(0), -1)

        # Dense layers with ReLU activations and dropout for regularization
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        # Return raw logits; apply sigmoid externally when needed (e.g., for BCEWithLogitsLoss and thresholding).
        return x
