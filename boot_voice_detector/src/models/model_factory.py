import torch
from src.models.neural_networks.cnn_model import CNNModel
from src.models.neural_networks.hybrid_model import HybridModel

class ModelFactory:
    @staticmethod
    def create_model(model_type, **kwargs):
        if model_type == 'cnn':
            return CNNModel(input_dim=kwargs.get('input_dim', 15), num_classes=kwargs.get('num_classes', 2))
        elif model_type == 'hybrid':
            return HybridModel(
                spec_dim=kwargs.get('spec_dim', 15),
                manual_features_dim=kwargs.get('manual_features_dim', 8),
                num_classes=kwargs.get('num_classes', 2)
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")