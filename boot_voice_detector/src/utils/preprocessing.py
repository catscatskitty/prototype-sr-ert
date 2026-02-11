import os
import numpy as np
import pandas as pd
import librosa
import librosa.display
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import yaml
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')
class Preprocessing:
    def __init__(self, config_path='config/audio_config.yaml'):
        """Инициализация препроцессора с конфигурацией"""
        self.config = self._load_config(config_path)
        self.scaler = None
        self.label_encoder = None
    def _load_config(self, config_path):
        """Загрузка конфигурации из YAML файла"""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    def load_and_preprocess_data(self, data_path, test_size=0.2, random_state=42):
        print(f"Загрузка данных из {data_path}...")
        metadata = self._load_metadata(data_path)
        spectrograms = []
        manual_features_list = []
        labels_list = []
        for idx, row in metadata.iterrows():
            try:
                audio_path = row['audio_path']
                label = row['label']
                y, sr = librosa.load(audio_path, sr=self.config['sample_rate'])
                spectrogram = self._extract_spectrogram(y, sr)
                manual_features = self._extract_manual_features(y, sr)
                spectrograms.append(spectrogram)
                manual_features_list.append(manual_features)
                labels_list.append(label)
                if idx % 100 == 0:
                    print(f"Обработано {idx}/{len(metadata)} файлов")
            except Exception as e:
                print(f"Ошибка при обработке файла {row['audio_path']}: {e}")
                continue
        spectrograms = np.array(spectrograms)
        manual_features = np.array(manual_features_list)
        labels = np.array(labels_list)
        self.label_encoder = LabelEncoder()
        labels_encoded = self.label_encoder.fit_transform(labels)
        joblib.dump(self.label_encoder, 'models/label_encoder.pkl')
        print(f"Загружено {len(spectrograms)} семплов")
        print(f"Размер спектрограмм: {spectrograms.shape}")
        print(f"Размер ручных признаков: {manual_features.shape}")
        return spectrograms, manual_features, labels_encoded
    def _load_metadata(self, data_path):
        metadata_path = os.path.join(data_path, 'metadata.csv')
        if os.path.exists(metadata_path):
            metadata = pd.read_csv(metadata_path)
            metadata['audio_path'] = metadata['filename'].apply(
                lambda x: os.path.join(data_path, 'audio', x)
            )
        else:
            audio_dir = os.path.join(data_path, 'audio')
            audio_files = []
            labels = []
            for root, dirs, files in os.walk(audio_dir):
                for file in files:
                    if file.endswith(('.wav', '.mp3', '.flac')):
                        audio_files.append(os.path.join(root, file))
                        label = os.path.basename(root)
                        labels.append(label)
            metadata = pd.DataFrame({
                'filename': [os.path.basename(f) for f in audio_files],
                'label': labels,
                'audio_path': audio_files
            })
        return metadata
    def _extract_spectrogram(self, y, sr):
        stft = librosa.stft(y, 
                          n_fft=self.config['n_fft'],
                          hop_length=self.config['hop_length'],
                          win_length=self.config['win_length'])
        spectrogram = np.abs(stft)
        spectrogram = librosa.amplitude_to_db(spectrogram, ref=np.max)
        spectrogram = (spectrogram - spectrogram.min()) / (spectrogram.max() - spectrogram.min())
        target_shape = self.config['spectrogram_shape']
        if spectrogram.shape != target_shape:
            import cv2
            spectrogram = cv2.resize(spectrogram, target_shape[:2], interpolation=cv2.INTER_LINEAR)
        if len(spectrogram.shape) == 2:
            spectrogram = np.expand_dims(spectrogram, axis=-1)
        return spectrogram
    def _extract_manual_features(self, y, sr):
        features = []
        features.extend(self._extract_time_domain_features(y))
        features.extend(self._extract_frequency_domain_features(y, sr))
        features.extend(self._extract_mfcc_features(y, sr))
        features.extend(self._extract_spectral_features(y, sr))
        return np.array(features)
    def _extract_time_domain_features(self, y):
        """Извлечение признаков из временной области"""
        features = []
        features.append(np.mean(np.abs(y)))
        features.append(np.std(y))
        features.append(np.max(np.abs(y)))
        features.append(np.sum(y**2))
        features.append(librosa.feature.zero_crossing_rate(y).mean())
        return features
    def _extract_frequency_domain_features(self, y, sr):
        """Извлечение признаков из частотной области"""
        features = []
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        features.append(spectral_centroid.mean())
        features.append(spectral_centroid.std())
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        features.append(spectral_bandwidth.mean())
        features.append(spectral_bandwidth.std())
        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        features.extend(spectral_contrast.mean(axis=1))
        return features
    def _extract_mfcc_features(self, y, sr):
        """Извлечение MFCC признаков"""
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=self.config['n_mfcc'])
        features = []
        for i in range(mfccs.shape[0]):
            features.append(mfccs[i].mean())
            features.append(mfccs[i].std())
        return features
    def _extract_spectral_features(self, y, sr):
        """Извлечение спектральных признаков"""
        features = []
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        features.append(spectral_rolloff.mean())
        spectral_flatness = librosa.feature.spectral_flatness(y=y)
        features.append(spectral_flatness.mean())
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr)
        features.append(mel_spec.mean())
        features.append(mel_spec.std())
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        features.extend(chroma.mean(axis=1))
        return features
    def preprocess_single_audio(self, audio_path):
        y, sr = librosa.load(audio_path, sr=self.config['sample_rate'])
        spectrogram = self._extract_spectrogram(y, sr)
        manual_features = self._extract_manual_features(y, sr)
        if self.scaler is not None:
            manual_features = self.scaler.transform(manual_features.reshape(1, -1)).flatten()
        return spectrogram, manual_features
    def save_preprocessor(self, path='models/preprocessor.pkl'):
        """Сохранение препроцессора"""
        joblib.dump({
            'scaler': self.scaler,
            'config': self.config,
            'label_encoder': self.label_encoder
        }, path)
    def load_preprocessor(self, path='models/preprocessor.pkl'):
        """Загрузка препроцессора"""
        data = joblib.load(path)
        self.scaler = data['scaler']
        self.config = data['config']
        self.label_encoder = data['label_encoder']
def load_and_preprocess_data(data_path, test_size=0.2, random_state=42, 
                           config_path='config/audio_config.yaml'):
    preprocessor = AudioPreprocessor(config_path)
    return preprocessor.load_and_preprocess_data(data_path, test_size, random_state)
def preprocess_audio_file(audio_path, config_path='config/audio_config.yaml'):
    preprocessor = AudioPreprocessor(config_path)
    return preprocessor.preprocess_single_audio(audio_path)
def split_data(spectrograms, manual_features, labels, test_size=0.2, random_state=42):
    X_train_spec, X_test_spec, X_train_manual, X_test_manual, y_train, y_test = train_test_split(
        spectrograms, manual_features, labels,
        test_size=test_size,
        random_state=random_state,
        stratify=labels
    )
    return X_train_spec, X_test_spec, X_train_manual, X_test_manual, y_train, y_test
def normalize_features(train_features, test_features=None):
    scaler = StandardScaler()
    train_normalized = scaler.fit_transform(train_features)
    if test_features is not None:
        test_normalized = scaler.transform(test_features)
        return train_normalized, test_normalized, scaler
    return train_normalized, scaler