# src/utils/preprocessing.py
import numpy as np
import librosa

def preprocess_audio(audio_path, target_sr=16000):
    audio, sr = librosa.load(audio_path, sr=target_sr)
    return audio, sr