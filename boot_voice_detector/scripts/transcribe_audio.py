"""
scripts/transcribe_audio.py

Transcribe all audio files listed in data/raw_annotations.csv and write
the transcripts into the "text" column (overwriting any existing text).

Optimized for macOS / Apple Silicon:
- Uses PyTorch MPS backend when available (device="mps").
- Falls back to CUDA or CPU otherwise.

Dependencies (install once in your venv):

    pip install openai-whisper

Usage:

    cd boot_voice_detector
    python scripts/transcribe_audio.py

Options (see --help):
- --model: whisper model size (tiny, base, small, medium, large)
- --language: language hint (e.g. "ru", "en"), or leave None to auto-detect
"""

import os
import sys
import argparse

import pandas as pd


def normalize_path(p: str) -> str:
    return p.replace("\\", os.sep).replace("/", os.sep)


def detect_device() -> str:
    try:
        import torch

        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"
    except Exception:
        return "cpu"


def main():
    parser = argparse.ArgumentParser(description="Transcribe audio into raw_annotations.csv[text].")
    parser.add_argument(
        "--model",
        type=str,
        default="small",
        help="Whisper model size (tiny, base, small, medium, large). Default: small.",
    )
    parser.add_argument(
        "--language",
        type=str,
        default=None,
        help='Optional language hint (e.g. "ru", "en"). If omitted, whisper auto-detects.',
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional limit on number of rows to transcribe (for testing).",
    )
    parser.add_argument(
        "--force-mps",
        action="store_true",
        help="Force using the MPS backend (Apple Silicon) and do NOT fall back to CPU on backend errors.",
    )
    parser.add_argument(
        "--backend",
        type=str,
        choices=["auto", "mlx"],
        default="auto",
        help="Backend to use: 'auto' (PyTorch Whisper) or 'mlx' (mlx-whisper on Apple Silicon).",
    )

    args = parser.parse_args()

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    annotations_path = os.path.join(project_root, "data", "raw_annotations.csv")

    if not os.path.exists(annotations_path):
        raise FileNotFoundError(f"raw_annotations.csv not found at {annotations_path}")

    print(f"[transcribe] Loading annotations from {annotations_path}")
    df = pd.read_csv(annotations_path)
    print(f"[transcribe] Rows: {len(df)}")

    # Ensure "text" column exists
    if "text" not in df.columns:
        df["text"] = ""

    audio_root = os.path.join(project_root, "data")
    total = len(df)
    processed = 0

    # Backend: MLX (mlx-whisper)
    if args.backend == "mlx":
        try:
            import mlx_whisper  # type: ignore
        except ImportError as e:
            raise SystemExit(
                "The 'mlx-whisper' package is not installed. Install it with:\n"
                "  pip install mlx-whisper\n"
            ) from e

        print("[transcribe] Using MLX backend (mlx-whisper)")

        for idx, row in df.iterrows():
            if args.max_samples is not None and processed >= args.max_samples:
                break

            rel = row.get("audio_path") or row.get("relative_path")
            if not isinstance(rel, str) or not rel:
                continue

            rel_norm = normalize_path(rel)
            audio_path = rel_norm
            if not os.path.isabs(audio_path):
                audio_path = os.path.join(audio_root, audio_path)
            audio_path = os.path.normpath(audio_path)

            if not os.path.exists(audio_path):
                print(f"[transcribe][warn] Missing audio: {audio_path}")
                continue

            try:
                print(f"[transcribe] ({processed+1}/{total}) {audio_path}")
                result = mlx_whisper.transcribe(audio_path)
                text = (result.get("text") or "").strip()
            except Exception as e:
                print(f"[transcribe][warn] Failed to transcribe {audio_path} with MLX: {e}")
                text = ""

            df.at[idx, "text"] = text
            processed += 1

    # Backend: auto (PyTorch Whisper with MPS/CPU)
    else:
        # Import whisper lazily so we can show a clear error if it's missing
        try:
            import whisper  # type: ignore
        except ImportError as e:
            raise SystemExit(
                "The 'whisper' package is not installed. Install it with:\n"
                "  pip install openai-whisper\n"
            ) from e

        # Choose device
        if args.force_mps:
            device = "mps"
            print("[transcribe] --force-mps enabled: forcing device='mps' (no CPU fallback).")
        else:
            device = detect_device()
        print(f"[transcribe] Preferred device: {device}")

        # Try to load on the preferred device.
        # If --force-mps is set, we let any backend errors propagate.
        load_device = device
        print(f"[transcribe] Loading whisper model: {args.model} on {load_device}")
        if args.force_mps:
            model = whisper.load_model(args.model, device=load_device)
        else:
            try:
                model = whisper.load_model(args.model, device=load_device)
            except NotImplementedError as e:
                print(f"[transcribe][warn] Failed to load model on {load_device}: {e}")
                load_device = "cpu"
                print(f"[transcribe] Falling back to CPU for Whisper model.")
                model = whisper.load_model(args.model, device=load_device)
            except RuntimeError as e:
                # Catch SparseMPS and similar backend issues
                if "SparseMPS" in str(e) or "aten::empty.memory_format" in str(e):
                    print(f"[transcribe][warn] Backend issue on {load_device}: {e}")
                    load_device = "cpu"
                    print(f"[transcribe] Falling back to CPU for Whisper model.")
                    model = whisper.load_model(args.model, device=load_device)
                else:
                    raise

        for idx, row in df.iterrows():
            if args.max_samples is not None and processed >= args.max_samples:
                break

            rel = row.get("audio_path") or row.get("relative_path")
            if not isinstance(rel, str) or not rel:
                continue

            rel_norm = normalize_path(rel)
            audio_path = rel_norm
            if not os.path.isabs(audio_path):
                audio_path = os.path.join(audio_root, audio_path)
            audio_path = os.path.normpath(audio_path)

            if not os.path.exists(audio_path):
                print(f"[transcribe][warn] Missing audio: {audio_path}")
                continue

            try:
                print(f"[transcribe] ({processed+1}/{total}) {audio_path}")
                result = model.transcribe(
                    audio_path,
                    language=args.language,
                    fp16=(load_device != "cpu"),
                )
                text = (result.get("text") or "").strip()
            except Exception as e:
                print(f"[transcribe][warn] Failed to transcribe {audio_path}: {e}")
                text = ""

            # Overwrite previous text
            df.at[idx, "text"] = text
            processed += 1

    print(f"[transcribe] Transcribed {processed} rows.")

    # Backup and save
    backup_path = annotations_path + ".pre_transcribe.bak"
    print(f"[transcribe] Backing up original annotations to {backup_path}")
    os.replace(annotations_path, backup_path)

    df.to_csv(annotations_path, index=False)
    print(f"[transcribe] Wrote updated annotations with transcripts to {annotations_path}")


if __name__ == "__main__":
    main()

