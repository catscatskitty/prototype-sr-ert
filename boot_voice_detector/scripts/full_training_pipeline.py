"""
scripts/full_training_pipeline.py

Automated end-to-end pipeline (no new augmentation):
1. Run acoustic analysis on all audio (using current raw_annotations.csv)
2. Run VAD/SNR analysis on all audio
3. Run linguistic analysis on all transcripts
4. Train the CNN model

Usage:

    python scripts/full_training_pipeline.py
"""

import os
import sys
import subprocess


def run(cmd: list[str]):
    print(f"[pipeline] Running: {' '.join(cmd)}")
    subprocess.check_call(cmd)


def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    py = sys.executable

    os.chdir(project_root)

    # 1. Acoustic analysis
    run([py, "scripts/run_acoustic_analysis.py"])

    # 2. VAD/SNR analysis
    run([py, "scripts/run_vad_snr_analysis.py"])

    # 3. Linguistic analysis
    run([py, "scripts/run_linguistic_analysis.py"])

    # 4. Training
    run([py, "scripts/model_training.py"])

    print("[pipeline] Done.")


if __name__ == "__main__":
    main()


