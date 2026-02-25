"""
scripts/run_vad_snr_analysis.py

Run VAD/SNR analysis on all audio files listed in data/raw_annotations.csv
and write a feature table to:

    data/vad_snr_analysis/vad_snr_results.csv

Core features:
- duration_sec
- speech_ratio
- snr_db
- speech_duration_sec
- avg_segment_snr
- num_speech_segments

These correspond closely to what VADSNRAnalyzer computes today and are
designed to capture how "speech-like" and noisy each clip is, which
helps the model distinguish natural from synthetic audio.
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.feature_ex.vad_snr import VADSNRAnalyzer


def normalize_path(p: str) -> str:
    return p.replace("\\", os.sep).replace("/", os.sep)


def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    annotations_path = os.path.join(project_root, "data", "raw_annotations.csv")
    out_dir = os.path.join(project_root, "data", "vad_snr_analysis")
    out_path = os.path.join(out_dir, "vad_snr_results.csv")

    if not os.path.exists(annotations_path):
        raise FileNotFoundError(f"raw_annotations.csv not found at {annotations_path}")

    print(f"[vad] Loading annotations from {annotations_path}")
    ann = pd.read_csv(annotations_path)
    print(f"[vad] Rows: {len(ann)}")

    vad_analyzer = VADSNRAnalyzer()

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
            print(f"[vad][warn] Missing audio: {audio_path}")
            continue

        try:
            vad_feats = vad_analyzer.analyze(audio_path)
        except Exception as e:
            print(f"[vad][warn] Failed on {audio_path}: {e}")
            continue

        # Derive robust counts/aggregate metrics from analyzer outputs.
        # VADSNRAnalyzer returns:
        # - speech_ratio
        # - snr_db
        # - speech_duration_sec
        # - duration_sec
        # - avg_segment_snr
        duration_sec = float(vad_feats.get("duration_sec", 0.0))
        speech_duration_sec = float(vad_feats.get("speech_duration_sec", 0.0))
        speech_ratio = float(vad_feats.get("speech_ratio", 0.0))
        snr_db = float(vad_feats.get("snr_db", 0.0))
        avg_segment_snr = float(vad_feats.get("avg_segment_snr", snr_db))

        # Approximate number of speech segments from speech_duration and hop length
        # (not stored explicitly, but useful as a rough proxy)
        num_speech_segments = 0
        if speech_duration_sec > 0 and duration_sec > 0:
            # Very rough: assume each segment ~0.5s
            num_speech_segments = int(max(1, round(speech_duration_sec / 0.5)))

        authenticity = str(row.get("authenticity", "")).strip()
        filename = os.path.basename(audio_path)

        feature_row = {
            "audio_path": rel_norm,
            "relative_path": rel_norm,
            "filename": filename,
            "authenticity": authenticity,
            "duration_sec": duration_sec,
            "speech_ratio": speech_ratio,
            "snr_db": snr_db,
            "speech_duration_sec": speech_duration_sec,
            "avg_segment_snr": avg_segment_snr,
            "num_speech_segments": float(num_speech_segments),
        }
        rows.append(feature_row)

    if not rows:
        print("[vad] No features extracted; nothing to write.")
        return

    os.makedirs(out_dir, exist_ok=True)
    df_out = pd.DataFrame(rows)
    df_out.to_csv(out_path, index=False)
    print(f"[vad] Wrote {len(df_out)} rows to {out_path}")


if __name__ == "__main__":
    main()

