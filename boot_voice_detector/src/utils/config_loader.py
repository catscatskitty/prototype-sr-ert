# src/utils/config_loader.py
import yaml
import os

class ConfigLoader:
    @staticmethod
    def load_yaml(config_path):
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    @staticmethod
    def get_audio_config():
        return ConfigLoader.load_yaml('config/audio_config.yaml')
    
    @staticmethod
    def get_model_config():
        return ConfigLoader.load_yaml('config/model_config.yaml')