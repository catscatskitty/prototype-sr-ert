import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import TensorDataset

def load_and_prepare_data():
    print("Loading spectrogram features...")
    spec_df = pd.read_csv('data/acoustic_analysis/spectrogram_features.csv')
    print(f"Loaded {len(spec_df)} spectrogram samples")
    
    print("Loading VAD SNR features...")
    vad_df = pd.read_csv('data/vad_snr_analysis/annotations_with_vad_snr.csv')
    print(f"Loaded {len(vad_df)} VAD samples")
    
    spec_features = spec_df.drop(['filename', 'audio_path', 'authenticity'], axis=1).values.astype(np.float32)
    spec_labels = (spec_df['authenticity'] == 'fake').astype(int).values
    
    vad_features = []
    vad_labels_list = []
    
    for idx, row in vad_df.iterrows():
        try:
            snr = float(row['snr_db']) if pd.notna(row['snr_db']) else 0.0
            speech_ratio = float(row['speech_ratio']) if pd.notna(row['speech_ratio']) else 0.0
            speech_segments = float(row['speech_segments']) if pd.notna(row['speech_segments']) else 0.0
            speech_duration = float(row['speech_duration_sec']) if pd.notna(row['speech_duration_sec']) else 0.0
            vad_features.append([snr, speech_ratio, speech_segments, speech_duration])
            vad_labels_list.append(1 if row['authenticity'] == 'fake' else 0)
        except Exception as e:
            print(f"Error processing row {idx}: {e}")
            vad_features.append([0.0, 0.0, 0.0, 0.0])
            vad_labels_list.append(0)
    
    vad_features = np.array(vad_features, dtype=np.float32)
    vad_labels = np.array(vad_labels_list, dtype=np.int64)
    
    print(f"Processed {len(vad_features)} VAD features")
    
    scaler_spec = StandardScaler()
    spec_features = scaler_spec.fit_transform(spec_features)
    
    scaler_vad = StandardScaler()
    vad_features = scaler_vad.fit_transform(vad_features)
    
    print("Splitting datasets...")
    X_spec_train, X_spec_test, y_train, y_test = train_test_split(
        spec_features, spec_labels, test_size=0.2, random_state=42, stratify=spec_labels
    )
    X_spec_train, X_spec_val, y_train, y_val = train_test_split(
        X_spec_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    X_vad_train, X_vad_test, y_vad_train, y_vad_test = train_test_split(
        vad_features, vad_labels, test_size=0.2, random_state=42, stratify=vad_labels
    )
    X_vad_train, X_vad_val, y_vad_train, y_vad_val = train_test_split(
        X_vad_train, y_vad_train, test_size=0.2, random_state=42, stratify=y_vad_train
    )
    
    min_train = min(len(X_spec_train), len(X_vad_train))
    min_val = min(len(X_spec_val), len(X_vad_val))
    min_test = min(len(X_spec_test), len(X_vad_test))
    
    print(f"Train size: {min_train}, Val size: {min_val}, Test size: {min_test}")
    
    return {
        'cnn': {
            'train': TensorDataset(torch.FloatTensor(X_spec_train), torch.LongTensor(y_train)),
            'val': TensorDataset(torch.FloatTensor(X_spec_val), torch.LongTensor(y_val)),
            'test': TensorDataset(torch.FloatTensor(X_spec_test), torch.LongTensor(y_test))
        },
        'hybrid': {
            'train': TensorDataset(
                torch.FloatTensor(X_spec_train[:min_train]),
                torch.FloatTensor(X_vad_train[:min_train]),
                torch.LongTensor(y_train[:min_train])
            ),
            'val': TensorDataset(
                torch.FloatTensor(X_spec_val[:min_val]),
                torch.FloatTensor(X_vad_val[:min_val]),
                torch.LongTensor(y_val[:min_val])
            ),
            'test': TensorDataset(
                torch.FloatTensor(X_spec_test[:min_test]),
                torch.FloatTensor(X_vad_test[:min_test]),
                torch.LongTensor(y_test[:min_test])
            )
        }
    }, (scaler_spec, scaler_vad)