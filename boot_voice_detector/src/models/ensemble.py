import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

class EnsembleModel:
    def __init__(self, models, weights=None):
        self.models = models
        self.weights = weights if weights else [1/len(models)] * len(models)
        
    def predict(self, x):
        all_preds = []
        for model in self.models:
            model.eval()
            with torch.no_grad():
                outputs = model(x)
                preds = torch.softmax(outputs, dim=1)
                all_preds.append(preds.cpu().numpy())
        
        weighted_preds = np.average(all_preds, axis=0, weights=self.weights)
        return np.argmax(weighted_preds, axis=1), weighted_preds
    
    def predict_proba(self, x):
        _, probs = self.predict(x)
        return probs