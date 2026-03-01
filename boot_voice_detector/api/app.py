# api/fixed_app_with_web_support.py
from flask import Flask, request, jsonify, render_template, send_from_directory
import torch
import numpy as np
import os
import sys
import librosa
import soundfile as sf
import tempfile
import pickle
import pandas as pd
from werkzeug.utils import secure_filename
from flask_cors import CORS
import traceback
import uuid
import subprocess
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.device_utils import get_device
from src.models.neural_networks.cnn_model import CNN1D
from src.feature_ex.feature_extractor import FeatureExtractor
from src.feature_ex.linguistic_analyzer import LinguisticAnalyzer

app = Flask(__name__, 
            template_folder=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'templates'),
            static_folder=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'static'))
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max
app.config['UPLOAD_FOLDER'] = os.path.join(tempfile.gettempdir(), 'fake_voice_detector')
app.config['ALLOWED_EXTENSIONS'] = {'wav', 'mp3', 'm4a', 'ogg', 'flac', 'aac', 'wma', 'webm'}
CORS(app)

# Создаем папку для загрузок
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

device = get_device()

# Feature column definitions
SPEC_FEATURE_COLS = [
    "mel_mean", "mel_std", "mel_max", "mel_min", "energy", "duration",
    "mfcc1_mean", "mfcc1_std", "mfcc2_mean", "mfcc2_std", "mfcc3_mean",
    "mfcc3_std", "mfcc4_mean", "mfcc4_std",
]
VAD_FEATURE_COLS = ["duration_sec", "speech_ratio", "snr_db", "speech_duration_sec"]
LING_FEATURE_COLS = [
    "word_count", "char_count", "vowel_count", "consonant_count",
    "avg_word_length", "has_cyrillic", "has_latin",
]

input_dim = len(SPEC_FEATURE_COLS) + len(VAD_FEATURE_COLS) + len(LING_FEATURE_COLS)

# Инициализация компонентов
model = None
scaler = None
feature_extractor = None
linguistic_analyzer = None
DECISION_THRESHOLD = 0.5

def allowed_file(filename):
    """Проверка разрешенного расширения файла"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def convert_to_wav(input_path, output_path=None):
    """
    Конвертирует аудиофайл в WAV формат используя ffmpeg или librosa
    """
    if output_path is None:
        output_path = os.path.join(app.config['UPLOAD_FOLDER'], f"converted_{uuid.uuid4()}.wav")
    
    # Пробуем конвертировать через ffmpeg (наиболее надежно)
    try:
        # Проверяем доступен ли ffmpeg
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
        
        # Конвертируем в WAV с параметрами 16kHz, моно
        cmd = [
            'ffmpeg', '-i', input_path,
            '-ar', '16000',  # частота дискретизации 16kHz
            '-ac', '1',      # моно
            '-c:a', 'pcm_s16le',  # 16-bit PCM
            '-y',            # перезаписывать выходной файл
            output_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0 and os.path.exists(output_path):
            return output_path
        else:
            print(f"FFmpeg conversion failed: {result.stderr}")
    except Exception as e:
        print(f"FFmpeg not available or error: {e}")
    
    # Если ffmpeg недоступен, пробуем через librosa + soundfile
    try:
        # Загружаем аудио через librosa
        y, sr = librosa.load(input_path, sr=16000, mono=True)
        
        # Сохраняем как WAV через soundfile
        sf.write(output_path, y, 16000, subtype='PCM_16')
        
        if os.path.exists(output_path):
            return output_path
    except Exception as e:
        print(f"Librosa conversion failed: {e}")
    
    return None

def load_audio_safe(file_path):
    """
    Безопасная загрузка аудио с поддержкой разных форматов
    """
    # Проверяем расширение файла
    file_ext = os.path.splitext(file_path)[1].lower()
    
    # Если это не WAV, конвертируем
    if file_ext != '.wav':
        print(f"Converting {file_ext} to WAV...")
        converted_path = convert_to_wav(file_path)
        if converted_path and os.path.exists(converted_path):
            try:
                # Загружаем сконвертированный файл
                y, sr = sf.read(converted_path)
                # Удаляем временный файл
                try:
                    os.unlink(converted_path)
                except:
                    pass
                return y, sr
            except Exception as e:
                print(f"Error loading converted file: {e}")
    
    # Пробуем загрузить напрямую
    try:
        # Сначала пробуем soundfile (поддерживает WAV, FLAC, OGG)
        y, sr = sf.read(file_path)
        # Конвертируем в моно если нужно
        if len(y.shape) > 1:
            y = np.mean(y, axis=1)
        return y, sr
    except Exception as e:
        print(f"Soundfile loading failed: {e}")
        
        # Пробуем librosa (поддерживает больше форматов)
        try:
            y, sr = librosa.load(file_path, sr=None, mono=True)
            return y, sr
        except Exception as e:
            print(f"Librosa loading failed: {e}")
    
    return None, None

def extract_features_for_web(audio_path):
    """
    Извлечение признаков специально для веб-интерфейса
    """
    try:
        print(f"Extracting features from: {audio_path}")
        
        # Загружаем аудио
        y, sr = load_audio_safe(audio_path)
        
        if y is None:
            print("Failed to load audio")
            return None
        
        print(f"Audio loaded: duration={len(y)/sr:.2f}s, sr={sr}")
        
        # Сохраняем временный WAV файл для feature extractor
        temp_wav = os.path.join(app.config['UPLOAD_FOLDER'], f"temp_{uuid.uuid4()}.wav")
        sf.write(temp_wav, y, sr)
        
        # Извлекаем признаки
        if feature_extractor is None:
            print("Feature extractor not initialized")
            return None
        
        feat_dict = feature_extractor.extract_all(temp_wav)
        
        # Удаляем временный файл
        try:
            os.unlink(temp_wav)
        except:
            pass
        
        if not feat_dict:
            print("Feature extraction returned empty dict")
            return None
        
        print(f"Extracted features: {len(feat_dict)}")
        
        # Собираем признаки в правильном порядке
        feature_vector = []
        
        # Добавляем спектральные признаки
        for col in SPEC_FEATURE_COLS:
            value = float(feat_dict.get(col, 0.0))
            feature_vector.append(value)
        
        # Добавляем VAD признаки
        for col in VAD_FEATURE_COLS:
            value = float(feat_dict.get(col, 0.0))
            feature_vector.append(value)
        
        # Добавляем нули для лингвистических признаков
        feature_vector.extend([0.0] * len(LING_FEATURE_COLS))
        
        raw_features = np.array([feature_vector], dtype=np.float32)
        
        # Применяем scaler если доступен
        if scaler is not None:
            try:
                features = scaler.transform(raw_features)
            except Exception as e:
                print(f"Scaler transform failed: {e}")
                features = raw_features
        else:
            features = raw_features
        
        return features
        
    except Exception as e:
        print(f"Error in extract_features_for_web: {e}")
        traceback.print_exc()
        return None

# Загрузка модели
model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                          'saved_models', 'best_cnn_model.pth')
scaler_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                          'saved_models', 'scaler.pkl')

print(f"Looking for model at: {model_path}")
print(f"Looking for scaler at: {scaler_path}")

if os.path.exists(model_path):
    try:
        model = CNN1D(input_dim=input_dim).to(device)
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            if 'decision_threshold' in checkpoint:
                DECISION_THRESHOLD = float(checkpoint['decision_threshold'])
        else:
            model.load_state_dict(checkpoint)
        
        model.eval()
        print(f"Model loaded successfully on {device}")
    except Exception as e:
        print(f"Error loading model: {e}")
        model = None
else:
    print("No trained model found")

if os.path.exists(scaler_path):
    try:
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
        print("Scaler loaded")
    except Exception as e:
        print(f"Error loading scaler: {e}")

# Инициализация экстракторов
try:
    feature_extractor = FeatureExtractor()
    linguistic_analyzer = LinguisticAnalyzer()
    print("Feature extractors initialized")
except Exception as e:
    print(f"Error initializing feature extractors: {e}")
    traceback.print_exc()

@app.route('/')
def index():
    """Главная страница"""
    try:
        return render_template('home.html')
    except Exception as e:
        print(f"Error rendering template: {e}")
        return jsonify({'error': 'Template not found'}), 404

@app.route('/static/<path:path>')
def serve_static(path):
    """Обслуживание статических файлов"""
    return send_from_directory(app.static_folder, path)

@app.route('/predict', methods=['POST'])
def predict():
    """Эндпоинт для предсказаний"""
    try:
        # Проверка модели
        if model is None:
            return jsonify({
                'success': False,
                'error': 'Модель не загружена. Пожалуйста, обучите модель сначала.'
            }), 500
        
        # Проверка файла
        if 'file' not in request.files:
            return jsonify({
                'success': False,
                'error': 'Не выбран файл'
            }), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({
                'success': False,
                'error': 'Не выбран файл'
            }), 400
        
        # Проверка расширения
        if not allowed_file(file.filename):
            return jsonify({
                'success': False,
                'error': f'Неподдерживаемый формат файла. Поддерживаются: {", ".join(app.config["ALLOWED_EXTENSIONS"])}'
            }), 400
        
        # Сохраняем файл
        filename = secure_filename(file.filename)
        unique_filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{filename}"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        
        file.save(file_path)
        print(f"File saved: {file_path}, size: {os.path.getsize(file_path)} bytes")
        
        # Извлекаем признаки
        features = extract_features_for_web(file_path)
        
        # Удаляем файл
        try:
            os.unlink(file_path)
        except:
            pass
        
        if features is None:
            return jsonify({
                'success': False,
                'error': 'Не удалось извлечь признаки из аудиофайла. Пожалуйста, проверьте формат файла.'
            }), 400
        
        # Предсказание
        with torch.no_grad():
            inputs = torch.FloatTensor(features).to(device)
            logits = model(inputs)
            probs_fake = torch.sigmoid(logits)[0][0].cpu().numpy()
            probs_real = 1.0 - probs_fake
            
            pred = 1 if probs_fake >= DECISION_THRESHOLD else 0
        
        # Формируем ответ
        result = {
            'success': True,
            'prediction': 'fake' if pred == 1 else 'real',
            'prediction_ru': 'Фейк (искусственный)' if pred == 1 else 'Реальный',
            'confidence': float(probs_fake if pred == 1 else probs_real),
            'confidence_percent': float(probs_fake if pred == 1 else probs_real) * 100,
            'probabilities': {
                'real': float(probs_real),
                'fake': float(probs_fake)
            },
            'threshold': float(DECISION_THRESHOLD)
        }
        
        print(f"Prediction: {result['prediction']}, confidence: {result['confidence']:.3f}")
        
        return jsonify(result)
        
    except Exception as e:
        print(f"Error in predict: {e}")
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': f'Ошибка при обработке: {str(e)}'
        }), 500

@app.route('/health', methods=['GET'])
def health():
    """Проверка состояния сервера"""
    return jsonify({
        'status': 'ok',
        'model_loaded': model is not None,
        'scaler_loaded': scaler is not None,
        'feature_extractor_loaded': feature_extractor is not None,
        'upload_folder': app.config['UPLOAD_FOLDER'],
        'upload_folder_exists': os.path.exists(app.config['UPLOAD_FOLDER']),
        'allowed_extensions': list(app.config['ALLOWED_EXTENSIONS'])
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print("="*50)
    print(f"Fake Voice Detector API")
    print("="*50)
    print(f"Server starting on port {port}")
    print(f"Upload folder: {app.config['UPLOAD_FOLDER']}")
    print(f"Model loaded: {model is not None}")
    print(f"Scaler loaded: {scaler is not None}")
    print(f"Feature extractor loaded: {feature_extractor is not None}")
    print(f"Supported formats: {', '.join(app.config['ALLOWED_EXTENSIONS'])}")
    print("="*50)
    
    # Проверка наличия ffmpeg
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
        print("FFmpeg is available - good for audio conversion")
    except:
        print("FFmpeg not found - audio conversion may be limited")
    
    app.run(debug=True, host='0.0.0.0', port=port)