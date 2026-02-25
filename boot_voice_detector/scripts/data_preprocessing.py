# scripts/data_preprocessing.py

import os
import pickle

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


class AudioDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


def load_and_merge_feature_csvs():
    """
    Load spectrogram and VAD/SNR feature CSVs and align them by relative_path/audio_path.

    This uses the same path semantics you've fixed in:
    - data/acoustic_analysis/spectrogram_features.csv
    - data/vad_snr_analysis/vad_snr_results.csv
    - data/linguistic_analysis/linguistic_features.csv
    """
    spec_path = "data/acoustic_analysis/spectrogram_features.csv"
    vad_path = "data/vad_snr_analysis/vad_snr_results.csv"
    ling_path = "data/linguistic_analysis/linguistic_features.csv"

    spec_df = pd.read_csv(spec_path)
    vad_df = pd.read_csv(vad_path)
    ling_df = pd.read_csv(ling_path)

    print(f"Spectrogram data: {len(spec_df)} rows")
    print(f"VAD SNR data: {len(vad_df)} rows")
    print(f"Linguistic data: {len(ling_df)} rows")

    # Determine join key: prefer 'relative_path', fall back to 'audio_path'
    join_key = None
    for key in ["relative_path", "audio_path"]:
        if key in spec_df.columns and key in vad_df.columns and key in ling_df.columns:
            join_key = key
            break
    if join_key is None:
        raise ValueError(
            "Neither 'relative_path' nor 'audio_path' found in spectrogram, VAD, and linguistic CSVs; "
            "cannot align rows by file."
        )

    print(f"Aligning spectrogram, VAD/SNR, and linguistic CSVs using key='{join_key}'")

    # Ensure each CSV has a filename column for traceability
    if "filename" not in spec_df.columns:
        spec_df["filename"] = spec_df[join_key].apply(lambda p: os.path.basename(str(p)))
    if "filename" not in vad_df.columns:
        vad_df["filename"] = vad_df[join_key].apply(lambda p: os.path.basename(str(p)))
    if "filename" not in ling_df.columns:
        ling_df["filename"] = ling_df[join_key].apply(lambda p: os.path.basename(str(p)))

    # Ensure numeric VAD columns are clean
    for col in ["snr_db", "speech_ratio", "speech_duration_sec", "duration_sec"]:
        if col in vad_df.columns:
            vad_df[col] = pd.to_numeric(vad_df[col], errors="coerce").fillna(0)

    # Columns we need from each side
    spec_feature_cols = [
        "mel_mean",
        "mel_std",
        "mel_max",
        "mel_min",
        "energy",
        "duration",
        "mfcc1_mean",
        "mfcc1_std",
        "mfcc2_mean",
        "mfcc2_std",
        "mfcc3_mean",
        "mfcc3_std",
        "mfcc4_mean",
        "mfcc4_std",
    ]
    vad_feature_cols = ["duration_sec", "speech_ratio", "snr_db", "speech_duration_sec"]
    ling_feature_cols = [
        "word_count",
        "char_count",
        "vowel_count",
        "consonant_count",
        "avg_word_length",
        "has_cyrillic",
        "has_latin",
    ]

    # Merge spectrogram+VAD
    merged = spec_df.merge(
        vad_df[[join_key, "filename"] + [c for c in vad_feature_cols if c in vad_df.columns]],
        on=join_key,
        how="inner",
        suffixes=("", "_vad"),
    )
    # Merge linguistic
    merged = merged.merge(
        ling_df[[join_key, "filename"] + ling_feature_cols],
        on=join_key,
        how="inner",
        suffixes=("", "_ling"),
    )

    print(f"Merged feature rows: {len(merged)}")

    X_spec = merged[spec_feature_cols].values
    X_vad = merged[vad_feature_cols].values
    X_ling = merged[ling_feature_cols].values
    X = np.hstack([X_spec, X_vad, X_ling])

    y = (merged["authenticity"] == "fake").astype(int).values
    print(f"Labels: {len(y)} samples, {sum(y)} fake, {len(y) - sum(y)} real")

    return merged, X, y


def load_spectrogram_data():
    """
    Backwards-compatible helper used by scripts.__init__.
    Returns only the spectrogram feature matrix and labels.
    """
    merged, X, y = load_and_merge_feature_csvs()
    spec_feature_cols = [
        "mel_mean",
        "mel_std",
        "mel_max",
        "mel_min",
        "energy",
        "duration",
        "mfcc1_mean",
        "mfcc1_std",
        "mfcc2_mean",
        "mfcc2_std",
        "mfcc3_mean",
        "mfcc3_std",
        "mfcc4_mean",
        "mfcc4_std",
    ]
    X_spec = merged[spec_feature_cols].values
    feature_cols = spec_feature_cols
    return X_spec, y, feature_cols


def load_vad_snr_data():
    """
    Backwards-compatible helper used by scripts.__init__.
    Returns only the VAD/SNR feature matrix and labels.
    """
    merged, X, y = load_and_merge_feature_csvs()
    vad_feature_cols = ["duration_sec", "speech_ratio", "snr_db", "speech_duration_sec"]
    X_vad = merged[vad_feature_cols].values
    feature_cols = vad_feature_cols
    return X_vad, y, feature_cols


def prepare_dataset():
    """Prepare dataset for training using aligned spectrogram + VAD/SNR CSVs."""
    _, X, y = load_and_merge_feature_csvs()

    # Train/test split with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # StandardScaler normalization
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Persist scaler so inference (API) can reuse the exact normalization
    os.makedirs("saved_models", exist_ok=True)
    with open(os.path.join("saved_models", "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)
    
    # Create PyTorch Dataset class
    train_dataset = AudioDataset(X_train, y_train)
    test_dataset = AudioDataset(X_test, y_test)
    
    print(f"Train: {len(train_dataset)} samples, {sum(y_train)} fake")
    print(f"Test: {len(test_dataset)} samples, {sum(y_test)} fake")
    
    return train_dataset, test_dataset, scaler


if __name__ == '__main__':
    # Load and prepare dataset
    train_dataset, test_dataset, scaler = prepare_dataset()
