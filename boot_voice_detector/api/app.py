# api/app.py (обновленная версия для предсказаний)
from flask import Flask, request, jsonify, render_template, send_from_directory
import torch
import numpy as np
import os
import sys
import librosa
import tempfile
import pickle
import pandas as pd
from werkzeug.utils import secure_filename
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.device_utils import get_device
from src.models.neural_networks.cnn_model import CNN1D
from src.feature_ex.feature_extractor import FeatureExtractor
from src.feature_ex.linguistic_analyzer import LinguisticAnalyzer

app = Flask(__name__, template_folder='../templates', static_folder='../static')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.config['UPLOAD_FOLDER'] = tempfile.gettempdir()

device = get_device()

# Feature column definitions (must match training)
SPEC_FEATURE_COLS = [
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
VAD_FEATURE_COLS = ["duration_sec", "speech_ratio", "snr_db", "speech_duration_sec"]
LING_FEATURE_COLS = [
    "word_count",
    "char_count",
    "vowel_count",
    "consonant_count",
    "avg_word_length",
    "has_cyrillic",
    "has_latin",
]

# Input dimension must match the total feature vector length used in training:
# 14 acoustic + 4 VAD/SNR + 7 linguistic = 25
input_dim = len(SPEC_FEATURE_COLS) + len(VAD_FEATURE_COLS) + len(LING_FEATURE_COLS)
model = CNN1D(input_dim=input_dim).to(device)
model_path = 'saved_models/best_cnn_model.pth'
scaler_path = 'saved_models/scaler.pkl'

scaler = None
feature_extractor = FeatureExtractor()
linguistic_analyzer = LinguisticAnalyzer()
DECISION_THRESHOLD = 0.5

# In-memory lookup tables for precomputed CSV features
csv_features_df = None
csv_by_filename = None
csv_by_relpath = None

if os.path.exists(model_path):
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model checkpoint from epoch: {checkpoint.get('epoch', 'unknown')} "
              f"with val_acc={checkpoint.get('val_acc', 'unknown')}")
        if 'decision_threshold' in checkpoint:
            DECISION_THRESHOLD = float(checkpoint['decision_threshold'])
            print(f"Using learned decision threshold from training: {DECISION_THRESHOLD:.3f}")
    else:
        model.load_state_dict(checkpoint)
    print(f"Model loaded successfully on {device}")
else:
    print("No trained model found. Please run training first.")

if os.path.exists(scaler_path):
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    print("Loaded feature scaler for inference.")
else:
    print("Warning: scaler.pkl not found; API will use unscaled features, which may reduce accuracy.")

# Allow manual override of decision threshold via environment variable for quick tuning.
thr_env = os.environ.get("FAKE_DECISION_THRESHOLD")
if thr_env is not None:
    try:
        DECISION_THRESHOLD = float(thr_env)
        print(f"Overriding decision threshold from env FAKE_DECISION_THRESHOLD={DECISION_THRESHOLD:.3f}")
    except ValueError:
        print(f"Invalid FAKE_DECISION_THRESHOLD value '{thr_env}', keeping threshold={DECISION_THRESHOLD:.3f}")

# Load precomputed CSV features for exact training-time feature reuse
spec_csv_path = "data/acoustic_analysis/spectrogram_features.csv"
vad_csv_path = "data/vad_snr_analysis/vad_snr_results.csv"
ling_csv_path = "data/linguistic_analysis/linguistic_features.csv"

if os.path.exists(spec_csv_path) and os.path.exists(vad_csv_path) and os.path.exists(ling_csv_path):
    try:
        spec_df = pd.read_csv(spec_csv_path)
        vad_df = pd.read_csv(vad_csv_path)
        ling_df = pd.read_csv(ling_csv_path)

        # Determine join key: prefer 'relative_path', fall back to 'audio_path'
        join_key = None
        for key in ["relative_path", "audio_path"]:
            if key in spec_df.columns and key in vad_df.columns and key in ling_df.columns:
                join_key = key
                break

        if join_key is not None:
            # Ensure filename column exists in all
            if "filename" not in spec_df.columns:
                spec_df["filename"] = spec_df[join_key].apply(lambda p: os.path.basename(str(p)))
            if "filename" not in vad_df.columns:
                vad_df["filename"] = vad_df[join_key].apply(lambda p: os.path.basename(str(p)))
            if "filename" not in ling_df.columns:
                ling_df["filename"] = ling_df[join_key].apply(lambda p: os.path.basename(str(p)))

            # Ensure VAD numeric cols are clean
            for col in ["snr_db", "speech_ratio", "speech_duration_sec", "duration_sec"]:
                if col in vad_df.columns:
                    vad_df[col] = pd.to_numeric(vad_df[col], errors="coerce").fillna(0)

            # Merge spectrogram + VAD
            merged = spec_df.merge(
                vad_df[[join_key, "filename"] + [c for c in VAD_FEATURE_COLS if c in vad_df.columns]],
                on=join_key,
                how="inner",
                suffixes=("", "_vad"),
            )
            # Merge linguistic
            merged = merged.merge(
                ling_df[[join_key, "filename"] + LING_FEATURE_COLS],
                on=join_key,
                how="inner",
                suffixes=("", "_ling"),
            )

            if len(merged) > 0:
                csv_features_df = merged
                csv_by_filename = merged.set_index("filename")
                csv_by_relpath = merged.set_index(join_key)
                print(
                    f"Loaded {len(merged)} precomputed feature rows from CSVs "
                    f"using join key '{join_key}'."
                )
        else:
            print(
            "Warning: neither 'relative_path' nor 'audio_path' present in all CSVs; "
                "cannot build CSV feature lookup."
            )
    except Exception as e:
        print(f"Warning: failed to initialize CSV feature lookup: {e}")
else:
    print("CSV feature files not found; API will rely on on-the-fly feature extraction.")

model.eval()

def extract_features_from_audio(audio_path):
    """
    Extract features using the same pipeline as was used to build the
    spectrogram and VAD/SNR CSVs for training.
    This ensures the API feature vector matches the training feature schema.
    """
    # Use the shared FeatureExtractor to compute all low-level acoustic + VAD features
    feat_dict = feature_extractor.extract_all(audio_path)
    if not feat_dict:
        return None

    # Start with spectrogram + VAD/SNR features
    spec_vad_cols = SPEC_FEATURE_COLS + VAD_FEATURE_COLS
    spec_vad = np.array([[float(feat_dict.get(col, 0.0)) for col in spec_vad_cols]], dtype=np.float32)

    # We don't have transcript text at prediction time, so pad linguistic features with zeros.
    ling_pad = np.zeros((1, len(LING_FEATURE_COLS)), dtype=np.float32)

    raw_features = np.hstack([spec_vad, ling_pad])

    # Apply the same StandardScaler used during training, if available
    if scaler is not None:
        features = scaler.transform(raw_features)
    else:
        features = raw_features
    
    return features


def get_features_from_csv(filename: str | None = None, relpath: str | None = None):
    """
    Try to retrieve precomputed features from the CSVs using filename or relative path.
    Returns a (1, 18) float32 array or None if not found.
    """
    if csv_features_df is None:
        return None

    row = None
    if filename and csv_by_filename is not None:
        key = os.path.basename(filename)
        if key in csv_by_filename.index:
            row = csv_by_filename.loc[key]
    if row is None and relpath and csv_by_relpath is not None:
        if relpath in csv_by_relpath.index:
            row = csv_by_relpath.loc[relpath]

    if row is None:
        return None

    ordered_cols = SPEC_FEATURE_COLS + VAD_FEATURE_COLS + LING_FEATURE_COLS
    raw_features = np.array([[float(row[col]) for col in ordered_cols]], dtype=np.float32)
    if scaler is not None:
        return scaler.transform(raw_features)
    return raw_features

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' in request.files:
            file = request.files['file']
            if file.filename == '':
                return jsonify({'success': False, 'error': 'No file selected'}), 400
            
            filename = secure_filename(file.filename)

            # First, try to use precomputed CSV features by filename
            features = get_features_from_csv(filename=filename)

            if features is None:
                # Fall back to on-the-fly feature extraction
                temp_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(temp_path)
                
                features = extract_features_from_audio(temp_path)
                
                try:
                    os.unlink(temp_path)
                except Exception:
                    pass
            
            if features is None:
                return jsonify({'success': False, 'error': 'Could not extract features from audio'}), 400
        else:
            data = request.get_json()
            if not data:
                return jsonify({'success': False, 'error': 'No JSON body provided'}), 400

            # Priority: if client provides filename/relative_path, try CSV lookup
            filename = data.get('filename')
            relpath = data.get('relative_path') or data.get('audio_path')
            features = get_features_from_csv(filename=filename, relpath=relpath)

            if features is None:
                if 'features' not in data:
                    return jsonify({'success': False, 'error': 'No features provided'}), 400
                raw = np.array(data['features']).reshape(1, -1).astype(np.float32)

                # Accept 18-d (spec+VAD) or full 25-d (spec+VAD+ling) and pad if needed.
                if raw.shape[1] == len(SPEC_FEATURE_COLS) + len(VAD_FEATURE_COLS):
                    ling_pad = np.zeros((1, len(LING_FEATURE_COLS)), dtype=np.float32)
                    raw_features = np.hstack([raw, ling_pad])
                elif raw.shape[1] == len(SPEC_FEATURE_COLS) + len(VAD_FEATURE_COLS) + len(LING_FEATURE_COLS):
                    raw_features = raw
                else:
                    return jsonify({
                        'success': False,
                        'error': f'Unexpected feature length {raw.shape[1]}; expected 18 or 25.'
                    }), 400

                if scaler is not None:
                    features = scaler.transform(raw_features)
                else:
                    features = raw_features
        
        with torch.no_grad():
            inputs = torch.FloatTensor(features).to(device)
            logits = model(inputs)
            probs_fake = torch.sigmoid(logits)[0][0]
            probs_real = 1.0 - probs_fake
            
            print(f"Probabilities - Real: {probs_real:.4f}, Fake: {probs_fake:.4f}")
            
            # Binary decision based on fake probability and learned threshold
            pred = 1 if probs_fake >= DECISION_THRESHOLD else 0
        
        return jsonify({
            'success': True,
            'prediction': 'fake' if pred == 1 else 'real',
            'confidence': float(probs_fake.cpu().numpy() if pred == 1 else probs_real.cpu().numpy()),
            'probabilities': {
                'real': float(probs_real.cpu().numpy()),
                'fake': float(probs_fake.cpu().numpy())
            }
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)