import numpy as np
import librosa
from src.feature_ex.spectrogram_analysis import SpectrogramAnalyzer

class AudioAnalyzer:
    def __init__(self):
        self.spectrogram_analyzer = SpectrogramAnalyzer()
    
    def analyze(self, audio_path):
        features = self.spectrogram_analyzer.extract_features(audio_path)
        return features