import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import torch
from torch.utils.data import Dataset, DataLoader

class AudioDataset(Dataset):
    def __init__(self, spectrogram_data, vad_data, labels):
        self.spectrogram_data = torch.FloatTensor(spectrogram_data)
        self.vad_data = torch.FloatTensor(vad_data)
        self.labels = torch.LongTensor(labels)
        self.sequence_length = 5
        
        n_samples = len(self.vad_data)
        n_features = self.vad_data.shape[1]
        self.lstm_features = torch.zeros(n_samples, self.sequence_length, n_features)
        
        for i in range(n_samples):
            start = max(0, i - self.sequence_length + 1)
            seq = self.vad_data[start:i+1]
            self.lstm_features[i, self.sequence_length - len(seq):, :] = seq

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'cnn_input': self.spectrogram_data[idx],
            'lstm_input': self.lstm_features[idx],
            'hand_features': self.vad_data[idx],
            'label': self.labels[idx]
        }

class DataPreprocessor:
    def __init__(self):
        self.scaler_spec = StandardScaler()
        self.scaler_vad = StandardScaler()
        self.label_encoder = LabelEncoder()

    def load_data(self):
        spec_df = pd.read_csv('data/acoustic_analysis/spectrogram_features.csv')
        vad_df = pd.read_csv('data/vad_snr_analysis/annotations_with_vad_snr.csv')
        raw_df = pd.read_csv('data/raw_annotations.csv')
        
        spec_cols = ['mel_mean','mel_std','mel_max','mel_min','energy','duration',
                     'mfcc1_mean','mfcc1_std','mfcc2_mean','mfcc2_std',
                     'mfcc3_mean','mfcc3_std','mfcc4_mean','mfcc4_std']
        vad_cols = ['duration_sec','speech_segments','speech_duration_sec','speech_ratio','snr_db']
        
        vad_df = vad_df.fillna(0)
        
        spec_df['filename_clean'] = spec_df['filename'].str.extract(r'([^\\/]+)$')[0]
        vad_df['filename_clean'] = vad_df['filename'].str.extract(r'([^\\/]+)$')[0]
        raw_df['filename_clean'] = raw_df['audio_path'].str.extract(r'([^\\/]+)$')[0]
        
        merged = pd.merge(spec_df[['filename_clean'] + spec_cols], 
                         vad_df[['filename_clean'] + vad_cols], on='filename_clean', how='inner')
        
        label_dict = dict(zip(raw_df['filename_clean'], raw_df['authenticity']))
        merged['label'] = merged['filename_clean'].map(label_dict)
        merged = merged.dropna(subset=['label'])
        merged['label'] = self.label_encoder.fit_transform(merged['label'])
        
        return merged[spec_cols].values, merged[vad_cols].values, merged['label'].values

    def prepare_data(self, batch_size=64, test_size=0.2, val_size=0.1):
        spec_data, vad_data, labels = self.load_data()
        
        spec_scaled = self.scaler_spec.fit_transform(spec_data)
        vad_scaled = self.scaler_vad.fit_transform(vad_data)
        
        X_spec_temp, X_spec_test, X_vad_temp, X_vad_test, y_temp, y_test = train_test_split(
            spec_scaled, vad_scaled, labels, test_size=test_size, random_state=42, stratify=labels)
        
        val_ratio = val_size / (1 - test_size)
        X_spec_train, X_spec_val, X_vad_train, X_vad_val, y_train, y_val = train_test_split(
            X_spec_temp, X_vad_temp, y_temp, test_size=val_ratio, random_state=42, stratify=y_temp)
        
        train_dataset = AudioDataset(X_spec_train, X_vad_train, y_train)
        val_dataset = AudioDataset(X_spec_val, X_vad_val, y_val)
        test_dataset = AudioDataset(X_spec_test, X_vad_test, y_test)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
        
        return train_loader, val_loader, test_loader, len(np.unique(labels))