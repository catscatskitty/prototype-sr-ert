"""
scripts/augment_audio.py

Offline augmentation for deepfake training data.

Reads data/raw_annotations.csv and, for each REAL clip, creates one or more
augmented versions under:

    data/audio_aug/<orig_name>_augX.wav

and appends corresponding rows to data/raw_annotations.csv.

Augmentations (for real clips):
- Additive noise at random SNR (20–30 dB)
- Small speed change (±5–8%)

This increases diversity of the real class so the model must rely on deeper
artifacts to detect fakes.
"""

import os
import sys
import argparse
from typing import Tuple

import numpy as np
import pandas as pd
import librosa
import soundfile as sf


def normalize_path(p: str) -> str:
    return p.replace("\\", os.sep).replace("/", os.sep)


def add_noise(y: np.ndarray, snr_db: float) -> np.ndarray:
    """Add white noise to achieve approximately the given SNR in dB."""
    if y.size == 0:
        return y
    signal_power = np.mean(y**2)
    if signal_power <= 0:
        return y
    snr_linear = 10 ** (snr_db / 10.0)
    noise_power = signal_power / snr_linear
    noise = np.random.randn(len(y)) * np.sqrt(noise_power)
    return y + noise


def change_speed(y: np.ndarray, rate: float) -> np.ndarray:
    """Time-stretch the signal by the given rate."""
    if y.size == 0:
        return y
    return librosa.effects.time_stretch(y, rate=rate)


def make_augmented_versions(
    y: np.ndarray,
    sr: int,
    snr_range: Tuple[float, float] = (20.0, 30.0),
    speed_range: Tuple[float, float] = (0.92, 1.08),
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Produce two augmented variants:
    - y_noise: additive noise
    - y_speed: time-stretched
    """
    snr_db = np.random.uniform(*snr_range)
    y_noise = add_noise(y, snr_db)

    speed = np.random.uniform(*speed_range)
    try:
        y_speed = change_speed(y, rate=speed)
    except Exception:
        y_speed = y

    # Normalize to avoid clipping
    for arr in (y_noise, y_speed):
        if np.max(np.abs(arr)) > 1.0:
            arr /= np.max(np.abs(arr)) + 1e-8

    return y_noise.astype(np.float32), y_speed.astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description="Augment real audio clips to balance training data.")
    parser.add_argument(
        "--max-real-aug",
        type=int,
        default=1,
        help="Number of augmented copies per real clip (default: 1, using noise+speed combo).",
    )

    args = parser.parse_args()

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    annotations_path = os.path.join(project_root, "data", "raw_annotations.csv")
    audio_root = os.path.join(project_root, "data")
    aug_dir = os.path.join(audio_root, "audio_aug")

    if not os.path.exists(annotations_path):
        raise FileNotFoundError(f"raw_annotations.csv not found at {annotations_path}")

    print(f"[augment] Loading annotations from {annotations_path}")
    df = pd.read_csv(annotations_path)
    print(f"[augment] Rows: {len(df)}")

    os.makedirs(aug_dir, exist_ok=True)

    aug_rows = []
    real_mask = df["authenticity"].astype(str).str.lower() == "real"
    real_df = df[real_mask].copy()
    print(f"[augment] Real clips to augment: {len(real_df)}")

    for idx, row in real_df.iterrows():
        rel = row.get("audio_path") or row.get("relative_path")
        if not isinstance(rel, str) or not rel:
            continue

        rel_norm = normalize_path(rel)
        src_path = rel_norm
        if not os.path.isabs(src_path):
            src_path = os.path.join(audio_root, src_path)
        src_path = os.path.normpath(src_path)

        if not os.path.exists(src_path):
            print(f"[augment][warn] Missing audio: {src_path}")
            continue

        try:
            y, sr = librosa.load(src_path, sr=None)
        except Exception as e:
            print(f"[augment][warn] Failed to load {src_path}: {e}")
            continue

        # Generate up to max-real-aug augmented copies
        y_noise, y_speed = make_augmented_versions(y, sr)
        base_name = os.path.splitext(os.path.basename(src_path))[0]

        created = 0
        # First augmented version: noise + speed combined
        if created < args.max_real_aug:
            y_mix = add_noise(y_speed, np.random.uniform(20.0, 30.0))
            out_name = f"{base_name}_aug_mix1.wav"
            out_path = os.path.join(aug_dir, out_name)
            try:
                sf.write(out_path, y_mix, sr)
                rel_out = normalize_path(os.path.join("audio_aug", out_name))
                new_row = row.copy()
                new_row["audio_path"] = rel_out
                new_row["relative_path"] = rel_out
                aug_rows.append(new_row)
                created += 1
            except Exception as e:
                print(f"[augment][warn] Failed to write {out_path}: {e}")

        # Second augmented version: noise only (if requested)
        if created < args.max_real_aug:
            out_name = f"{base_name}_aug_noise.wav"
            out_path = os.path.join(aug_dir, out_name)
            try:
                sf.write(out_path, y_noise, sr)
                rel_out = normalize_path(os.path.join("audio_aug", out_name))
                new_row = row.copy()
                new_row["audio_path"] = rel_out
                new_row["relative_path"] = rel_out
                aug_rows.append(new_row)
                created += 1
            except Exception as e:
                print(f"[augment][warn] Failed to write {out_path}: {e}")

    if not aug_rows:
        print("[augment] No augmented rows created.")
        return

    # Append augmented rows to annotations and save
    df_aug = pd.DataFrame(aug_rows)
    df_out = pd.concat([df, df_aug], ignore_index=True)

    backup_path = annotations_path + ".pre_aug.bak"
    print(f"[augment] Backing up original annotations to {backup_path}")
    os.replace(annotations_path, backup_path)

    df_out.to_csv(annotations_path, index=False)
    print(f"[augment] Wrote augmented annotations to {annotations_path} (total rows: {len(df_out)})")


if __name__ == "__main__":
    main()

