import torch

class Predictor:
    def __init__(self, model_path, device='cpu'):
        self.device = device
        self.model = torch.load(model_path, map_location=device)
        self.model.eval()
    
    def predict(self, features):
        with torch.no_grad():
            features_tensor = torch.tensor(features, dtype=torch.float32).to(self.device)
            output = self.model(features_tensor)
            return torch.argmax(output, dim=1).cpu().numpy()