"""
scripts/run_linguistic_analysis.py

Run linguistic analysis on all text transcripts listed in data/raw_annotations.csv
and write a feature table to:

    data/linguistic_analysis/linguistic_features.csv

Features (from src/feature_ex/linguistic_analyzer.py):
- word_count
- char_count
- vowel_count
- consonant_count
- avg_word_length
- has_cyrillic (0/1)
- has_latin (0/1)
"""

import os
import sys

import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.feature_ex.linguistic_analyzer import LinguisticAnalyzer


def normalize_path(p: str) -> str:
    return p.replace("\\", os.sep).replace("/", os.sep)


def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    annotations_path = os.path.join(project_root, "data", "raw_annotations.csv")
    out_dir = os.path.join(project_root, "data", "linguistic_analysis")
    out_path = os.path.join(out_dir, "linguistic_features.csv")

    if not os.path.exists(annotations_path):
        raise FileNotFoundError(f"raw_annotations.csv not found at {annotations_path}")

    print(f"[ling] Loading annotations from {annotations_path}")
    ann = pd.read_csv(annotations_path)
    print(f"[ling] Rows: {len(ann)}")

    ling = LinguisticAnalyzer()

    rows = []
    for idx, row in ann.iterrows():
        rel = row.get("audio_path") or row.get("relative_path")
        if not isinstance(rel, str) or not rel:
            continue

        text = row.get("text", "")
        feats = ling.analyze(text)

        rel_norm = normalize_path(rel)
        authenticity = str(row.get("authenticity", "")).strip()
        filename = os.path.basename(rel_norm)

        feature_row = {
            "audio_path": rel_norm,
            "relative_path": rel_norm,
            "filename": filename,
            "authenticity": authenticity,
            "word_count": feats["word_count"],
            "char_count": feats["char_count"],
            "vowel_count": feats["vowel_count"],
            "consonant_count": feats["consonant_count"],
            "avg_word_length": feats["avg_word_length"],
            "has_cyrillic": int(feats["has_cyrillic"]),
            "has_latin": int(feats["has_latin"]),
        }
        rows.append(feature_row)

    if not rows:
        print("[ling] No linguistic features extracted; nothing to write.")
        return

    os.makedirs(out_dir, exist_ok=True)
    df_out = pd.DataFrame(rows)
    df_out.to_csv(out_path, index=False)
    print(f"[ling] Wrote {len(df_out)} rows to {out_path}")


if __name__ == "__main__":
    main()

