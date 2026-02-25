import torch
import numpy as np

class ModelPredictor:
    def __init__(self, model, device, scaler_spec=None, scaler_vad=None):
        self.model = model.to(device)
        self.device = device
        self.scaler_spec = scaler_spec
        self.scaler_vad = scaler_vad
        self.model.eval()
        
    def predict(self, features):
        if isinstance(features, np.ndarray):
            features = torch.FloatTensor(features)
        
        if len(features.shape) == 1:
            features = features.unsqueeze(0)
        
        features = features.to(self.device)
        
        with torch.no_grad():
            outputs = self.model(features)
            probabilities = torch.softmax(outputs, dim=1)
            predictions = torch.argmax(outputs, dim=1)
        
        return predictions.cpu().numpy(), probabilities.cpu().numpy()
    
    def predict_proba(self, features):
        preds, probs = self.predict(features)
        return probs