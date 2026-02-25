import numpy as np
import pandas as pd
from src.feature_ex.audio_analyzer import AudioAnalyzer
from src.feature_ex.vad_snr import VADSNRAnalyzer

class FeatureExtractor:
    def __init__(self):
        self.audio_analyzer = AudioAnalyzer()
        self.vad_analyzer = VADSNRAnalyzer()
    
    def extract_all(self, audio_path):
        audio_features = self.audio_analyzer.analyze(audio_path)
        vad_features = self.vad_analyzer.analyze(audio_path)
        return {**audio_features, **vad_features}