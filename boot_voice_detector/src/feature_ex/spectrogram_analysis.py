import torch
import torchaudio
import torch.nn as nn
import numpy as np
import pandas as pd
import os
from pathlib import Path
from tqdm import tqdm
class SpectrogramAnalysis:
    def __init__(self, sample_rate=16000, n_fft=512, n_mels=128):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.n_mels = n_mels
        self.transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            n_mels=n_mels,
            hop_length=160
        )
    def extract_features(self, audio_path):
        try:
            waveform, sr = torchaudio.load(audio_path)
            if sr != self.sample_rate:
                resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
                waveform = resampler(waveform)
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            mel_spec = self.transform(waveform)
            mel_spec_db = torchaudio.transforms.AmplitudeToDB()(mel_spec)
            features = {
                'mel_mean': mel_spec_db.mean().item(),
                'mel_std': mel_spec_db.std().item(),
                'mel_max': mel_spec_db.max().item(),
                'mel_min': mel_spec_db.min().item(),
                'energy': (mel_spec_db ** 2).mean().item(),
                'duration': waveform.shape[1] / self.sample_rate
            }
            mfcc_transform = torchaudio.transforms.MFCC(
                sample_rate=self.sample_rate,
                n_mfcc=13,
                melkwargs={'n_fft': self.n_fft, 'n_mels': self.n_mels}
            )
            mfcc = mfcc_transform(waveform)
            for i in range(min(4, mfcc.shape[1])):
                features[f'mfcc{i+1}_mean'] = mfcc[0, i].mean().item()
                features[f'mfcc{i+1}_std'] = mfcc[0, i].std().item()
            return features, mel_spec_db
        except Exception as e:
            print(f"Error processing {audio_path}: {e}")
            return None, None
def analyze_all_audio():
    annotations = pd.read_csv('data/raw/raw_annotations.csv')
    if 'filename' in annotations.columns:
        file_col = 'filename'
    elif 'file' in annotations.columns:
        file_col = 'file'
    elif 'audio_path' in annotations.columns:
        file_col = 'audio_path'
    else:
        file_col = annotations.columns[0]
    extractor = SpectrogramAnalyzer.SpectrogramExtractor()
    results = []
    for idx, row in tqdm(annotations.iterrows(), total=len(annotations)):
        filename = str(row[file_col])
        if os.path.isabs(filename):
            audio_path = filename
        else:
            possible_paths = [
                f'data/raw/{filename}',
                f'data/raw/{os.path.basename(filename)}',
                filename
            ]
            audio_path = None
            for path in possible_paths:
                if os.path.exists(path):
                    audio_path = path
                    break
            if audio_path is None:
                print(f"File not found: {filename}")
                continue
        features, spec = extractor.extract_features(audio_path)
        if features:
            features['filename'] = os.path.basename(filename)
            features['audio_path'] = audio_path
            if 'authenticity' in annotations.columns:
                features['authenticity'] = row['authenticity']
            elif 'label' in annotations.columns:
                features['authenticity'] = row['label']
            else:
                features['authenticity'] = 'unknown'
            results.append(features)
    df = pd.DataFrame(results)
    output_path = 'gpu_acoustic_analysis/spectrogram_features.csv'
    os.makedirs('gpu_acoustic_analysis', exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\nAnalysis complete! Saved {len(df)} samples to {output_path}")
    print(f"Features extracted: {len(df.columns)}")
    return df
class SpectrogramCNN(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)
        self.flatten_size = 128 * 16 * 16  
        self.fc1 = nn.Linear(self.flatten_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
def train_on_spectrograms(df, epochs=50):
    """Обучение CNN на признаках спектрограмм"""
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [col for col in numeric_cols if col not in ['authenticity']]
    X = df[numeric_cols].fillna(0).values
    y = LabelEncoder().fit_transform(df['authenticity'].fillna('unknown'))
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    X_train_t = torch.FloatTensor(X_train).unsqueeze(1)  
    y_train_t = torch.LongTensor(y_train)
    X_test_t = torch.FloatTensor(X_test).unsqueeze(1)
    y_test_t = torch.LongTensor(y_test)
    model = SpectrogramCNN(num_classes=len(np.unique(y)))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    X_train_t = X_train_t.to(device)
    y_train_t = y_train_t.to(device)
    X_test_t = X_test_t.to(device)
    y_test_t = y_test_t.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_t)
        loss = criterion(outputs, y_train_t)
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                test_outputs = model(X_test_t)
                test_acc = (test_outputs.argmax(1) == y_test_t).float().mean().item()
            print(f"Epoch {epoch}: Loss={loss.item():.4f}, Test Acc={test_acc:.4f}")
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test_t)
        test_acc = (test_outputs.argmax(1) == y_test_t).float().mean().item()
    print(f"\nFinal Test Accuracy: {test_acc:.4f}")
    torch.save(model.state_dict(), 'models/spectrogram_cnn.pth')
    print("Model saved to models/spectrogram_cnn.pth")
    return model, test_acc
if __name__ == "__main__":
    print("Starting spectrogram analysis...")
    df = analyze_all_audio()
    if len(df) > 100 and 'authenticity' in df.columns:
        print("\nTraining CNN on spectrogram features...")
        train_on_spectrograms(df, epochs=50)
    else:
        print(f"\nNot enough data for training: {len(df)} samples")