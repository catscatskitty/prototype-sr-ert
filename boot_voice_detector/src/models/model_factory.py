import torch
from src.models.neural_networks.cnn_model import CNNModel
from src.models.neural_networks.hybrid_model import HybridModel

class ModelFactory:
    @staticmethod
    def get_model(model_type, **kwargs):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if model_type == 'cnn':
            model = CNNModel(**kwargs)
        elif model_type == 'hybrid':
            model = HybridModel(**kwargs)
        else:
            raise ValueError(f'Unknown model type: {model_type}')
        return model.to(device)