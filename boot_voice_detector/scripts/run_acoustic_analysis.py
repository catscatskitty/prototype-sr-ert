"""
scripts/run_acoustic_analysis.py

Run acoustic (spectrogram-based) analysis on all audio files listed in
data/raw_annotations.csv and write a rich feature table to:

    data/acoustic_analysis/spectrogram_features.csv

The feature set is:
- Core features already used for training:
  - mel_mean, mel_std, mel_max, mel_min
  - energy, duration
  - mfcc1_mean/std ... mfcc4_mean/std
- Additional potentially useful metrics:
  - spectral_centroid_mean/std
  - spectral_bandwidth_mean/std
  - spectral_flatness_mean
  - spectral_rolloff_mean
  - zcr_mean (zero-crossing rate)

These extra metrics are written to the CSV but the current model only
uses the original 14 + 4 VAD/SNR features. You can extend the model
later to exploit them.
"""

import os
import sys
from typing import Dict, Any

import numpy as np
import pandas as pd
import librosa

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.feature_ex.spectrogram_analysis import SpectrogramAnalyzer


def normalize_path(p: str) -> str:
    return p.replace("\\", os.sep).replace("/", os.sep)


def compute_extra_spectral_features(y: np.ndarray, sr: int) -> Dict[str, Any]:
    """Compute additional spectral statistics helpful for deepfake detection."""
    features: Dict[str, Any] = {}

    if y.size == 0:
        # Leave zeros; caller can fill defaults
        return {
            "spectral_centroid_mean": 0.0,
            "spectral_centroid_std": 0.0,
            "spectral_bandwidth_mean": 0.0,
            "spectral_bandwidth_std": 0.0,
            "spectral_flatness_mean": 0.0,
            "spectral_rolloff_mean": 0.0,
            "zcr_mean": 0.0,
        }

    # Spectral centroid & bandwidth
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)

    features["spectral_centroid_mean"] = float(np.mean(centroid))
    features["spectral_centroid_std"] = float(np.std(centroid))
    features["spectral_bandwidth_mean"] = float(np.mean(bandwidth))
    features["spectral_bandwidth_std"] = float(np.std(bandwidth))

    # Flatness & rolloff
    flatness = librosa.feature.spectral_flatness(y=y)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    features["spectral_flatness_mean"] = float(np.mean(flatness))
    features["spectral_rolloff_mean"] = float(np.mean(rolloff))

    # Zero-crossing rate
    zcr = librosa.feature.zero_crossing_rate(y)
    features["zcr_mean"] = float(np.mean(zcr))

    return features


def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    annotations_path = os.path.join(project_root, "data", "raw_annotations.csv")
    out_dir = os.path.join(project_root, "data", "acoustic_analysis")
    out_path = os.path.join(out_dir, "spectrogram_features.csv")

    if not os.path.exists(annotations_path):
        raise FileNotFoundError(f"raw_annotations.csv not found at {annotations_path}")

    print(f"[acoustic] Loading annotations from {annotations_path}")
    ann = pd.read_csv(annotations_path)
    print(f"[acoustic] Rows: {len(ann)}")

    spec_analyzer = SpectrogramAnalyzer()

    rows = []
    for idx, row in ann.iterrows():
        rel = row.get("audio_path") or row.get("relative_path")
        if not isinstance(rel, str) or not rel:
            continue

        rel_norm = normalize_path(rel)
        # All audio files live under the project_root/data folder.
        audio_path = rel_norm
        if not os.path.isabs(audio_path):
            audio_path = os.path.join(project_root, "data", audio_path)
        audio_path = os.path.normpath(audio_path)

        if not os.path.exists(audio_path):
            print(f"[acoustic][warn] Missing audio: {audio_path}")
            continue

        try:
            # Core spectrogram-based features
            spec_feats = spec_analyzer.extract_features(audio_path)
            # Load raw audio once for extra spectral metrics
            y, sr = librosa.load(audio_path, sr=None)
            extra_feats = compute_extra_spectral_features(y, sr)
        except Exception as e:
            print(f"[acoustic][warn] Failed on {audio_path}: {e}")
            continue

        authenticity = str(row.get("authenticity", "")).strip()
        filename = os.path.basename(audio_path)

        feature_row = {
            **spec_feats,
            **extra_feats,
            "filename": filename,
            "audio_path": rel_norm,
            "relative_path": rel_norm,
            "authenticity": authenticity,
        }
        rows.append(feature_row)

    if not rows:
        print("[acoustic] No features extracted; nothing to write.")
        return

    os.makedirs(out_dir, exist_ok=True)
    df_out = pd.DataFrame(rows)
    df_out.to_csv(out_path, index=False)
    print(f"[acoustic] Wrote {len(df_out)} rows to {out_path}")


if __name__ == "__main__":
    main()

