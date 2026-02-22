import torch
import numpy as np

class Predictor:
    def __init__(self, model, device):
        self.model = model
        self.device = device

    def predict(self, features):
        self.model.eval()
        with torch.no_grad():
            if 'hybrid' in str(type(self.model)).lower():
                cnn_input = torch.FloatTensor(features['cnn_input']).unsqueeze(0).to(self.device)
                lstm_input = torch.FloatTensor(features['lstm_input']).unsqueeze(0).to(self.device)
                hand_features = torch.FloatTensor(features['hand_features']).unsqueeze(0).to(self.device)
                outputs = self.model(cnn_input, lstm_input, hand_features)
            else:
                inputs = torch.FloatTensor(features['cnn_input']).unsqueeze(0).to(self.device)
                outputs = self.model(inputs)
            return torch.softmax(outputs, dim=1).cpu().numpy()[0]