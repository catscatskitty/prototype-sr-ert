import numpy as np
import librosa
import torch
import torchaudio

class SpectrogramAnalyzer:
    def __init__(self, n_mels=128, n_fft=2048, hop_length=512):
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
    
    def extract(self, audio_path):
        y, sr = librosa.load(audio_path, sr=None)
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=self.n_mels, 
                                                  n_fft=self.n_fft, hop_length=self.hop_length)
        log_mel = librosa.power_to_db(mel_spec)
        return log_mel
    
    def extract_features(self, audio_path):
        y, sr = librosa.load(audio_path, sr=None)
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=self.n_mels)
        log_mel = librosa.power_to_db(mel_spec)
        
        features = {
            'mel_mean': np.mean(log_mel),
            'mel_std': np.std(log_mel),
            'mel_max': np.max(log_mel),
            'mel_min': np.min(log_mel),
            'energy': np.sum(y**2),
            'duration': len(y)/sr
        }
        
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=4)
        for i in range(4):
            features[f'mfcc{i+1}_mean'] = np.mean(mfccs[i])
            features[f'mfcc{i+1}_std'] = np.std(mfccs[i])
        
        return features

SpectrogramExtractor = SpectrogramAnalyzer