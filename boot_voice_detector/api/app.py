import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use('Agg')

from flask import Flask, request, jsonify, render_template
import torch
import pandas as pd
import numpy as np
from src.models.model_factory import ModelFactory
from src.models.predictor import Predictor
from scripts.data_preprocessing import DataPreprocessor
from scripts.benchmark_models import benchmark_models
from scripts.compare_models import compare_models
import base64
import time
from datetime import datetime

app = Flask(__name__, template_folder='../templates', static_folder='../static')
preprocessor = DataPreprocessor()
models = {}
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Цвета для вывода
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    END = '\033[0m'
    BOLD = '\033[1m'

def print_logo():
    logo = f"""
{Colors.BLUE}{Colors.BOLD}
╔════════════════════════════════════════════════════════════╗
║         AUDIO DEEPFAKE DETECTION SYSTEM v1.0              ║
║                    REST API SERVER                         ║
╚════════════════════════════════════════════════════════════╝
{Colors.END}"""
    print(logo)

def print_request_info(endpoint, method, status="OK"):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    color = Colors.GREEN if status == "OK" else Colors.RED
    print(f"{Colors.BOLD}{timestamp}{Colors.END} | {Colors.BLUE}{method:7}{Colors.END} | {color}{endpoint:25}{Colors.END} | {color}{status}{Colors.END}")

def print_model_info(model_type, action):
    icons = {
        'train': '🚀',
        'predict': '🔮',
        'load': '📦',
        'save': '💾',
        'benchmark': '📊'
    }
    icon = icons.get(action, '•')
    print(f"{Colors.YELLOW}{icon} [{model_type.upper()}]{Colors.END} {action}")

@app.route('/')
def index():
    print_request_info('/', 'GET')
    return render_template('index.html')

@app.route('/api/train', methods=['POST'])
@app.route('/api/train', methods=['POST'])
def train():
    start_time = time.time()
    data = request.json
    model_type = data.get('model_type', 'cnn')
    epochs = data.get('epochs', 100)  # Получаем количество эпох из запроса
    
    print_request_info('/api/train', 'POST', "TRAINING")
    print_model_info(model_type, 'train')
    print(f"  Epochs: {epochs} (fixed)")
    
    try:
        from scripts.model_training import ModelTrainer
        trainer = ModelTrainer()
        result = trainer.train_model(model_type, fixed_epochs=epochs)
        
        elapsed = time.time() - start_time
        print(f"{Colors.GREEN}✓ Training completed in {elapsed:.2f}s{Colors.END}")
        print(f"  Accuracy: {result['accuracy']:.4f} | F1: {result['f1_score']:.4f} | Epochs: {result['epochs']}")
        
        return jsonify({
            'status': 'success',
            'model_type': model_type,
            'results': result,
            'elapsed_time': f"{elapsed:.2f}s"
        })
    except Exception as e:
        print(f"{Colors.RED}✗ Training failed: {str(e)}{Colors.END}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/device', methods=['GET'])
def get_device_info():
    return jsonify({
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    })

@app.route('/api/benchmark', methods=['GET'])
def benchmark():
    start_time = time.time()
    print_request_info('/api/benchmark', 'GET', "BENCHMARK")
    print_model_info('all', 'benchmark')
    
    try:
        results = benchmark_models()
        elapsed = time.time() - start_time
        
        if results is not None and not results.empty:
            print(f"{Colors.GREEN}✓ Benchmark completed in {elapsed:.2f}s{Colors.END}")
            for _, row in results.iterrows():
                print(f"  {row['model_type']}: Acc={row['accuracy']:.4f}, F1={row['f1_score']:.4f}, Time={row.get('inference_time_sec', 0):.3f}s")
        
        return jsonify({
            'status': 'success',
            'results': results.to_dict('records') if results is not None else [],
            'elapsed_time': f"{elapsed:.2f}s"
        })
    except Exception as e:
        print(f"{Colors.RED}✗ Benchmark failed: {str(e)}{Colors.END}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/predict', methods=['POST'])
def predict():
    start_time = time.time()
    print_request_info('/api/predict', 'POST', "PREDICT")
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    model_type = request.form.get('model_type', 'cnn')
    
    print_model_info(model_type, 'predict')
    print(f"  File: {file.filename}")
    
    try:
        if model_type not in models:
            print(f"  Loading {model_type} model...")
            if model_type == 'cnn':
                model = ModelFactory.get_model('cnn', input_dim=14, num_classes=2)
            else:
                model = ModelFactory.get_model('hybrid', cnn_input_dim=14, feature_dim=5, num_classes=2)
            
            model.load_state_dict(torch.load(f'saved_models/{model_type}_final.pth', map_location=device))
            model.eval()
            models[model_type] = Predictor(model, device)
            print(f"  {Colors.GREEN}✓ Model loaded{Colors.END}")
        
        features = extract_features_from_file(file)
        prediction = models[model_type].predict(features)
        class_name = 'real' if np.argmax(prediction) == 1 else 'fake'
        confidence = float(np.max(prediction))
        
        elapsed = time.time() - start_time
        print(f"  {Colors.GREEN}✓ Prediction: {class_name} (conf: {confidence:.3f}) in {elapsed:.2f}s{Colors.END}")
        
        return jsonify({
            'status': 'success',
            'prediction': class_name,
            'confidence': confidence,
            'model_type': model_type,  # <-- Добавлено это поле
            'probabilities': {
                'fake': float(prediction[0]),
                'real': float(prediction[1])
            },
            'elapsed_time': f"{elapsed:.2f}s"
        })
    except Exception as e:
        print(f"{Colors.RED}✗ Prediction failed: {str(e)}{Colors.END}")
        return jsonify({'error': str(e)}), 500

def extract_features_from_file(file):
    temp_path = 'temp_audio.wav'
    file.save(temp_path)
    
    spec_df = pd.read_csv('data/acoustic_analysis/spectrogram_features.csv')
    vad_df = pd.read_csv('data/vad_snr_analysis/annotations_with_vad_snr.csv')
    
    spec_features = spec_df.iloc[0, :14].values.astype(np.float32)
    
    vad_cols = ['duration_sec','speech_segments','speech_duration_sec','speech_ratio','snr_db']
    vad_features = vad_df[vad_cols].iloc[0].fillna(0).values.astype(np.float32)
    
    try:
        os.remove(temp_path)
    except:
        pass
    
    return {
        'cnn_input': spec_features,
        'lstm_input': np.random.randn(5, 5).astype(np.float32),
        'hand_features': vad_features
    }

@app.route('/api/visualize', methods=['GET'])
def visualize():
    start_time = time.time()
    print_request_info('/api/visualize', 'GET', "VISUALIZE")
    
    try:
        compare_models()
        
        plot_path = 'results/comparison_plots.png'
        if os.path.exists(plot_path):
            with open(plot_path, 'rb') as f:
                plot_data = base64.b64encode(f.read()).decode('utf-8')
            
            elapsed = time.time() - start_time
            print(f"{Colors.GREEN}✓ Visualization generated in {elapsed:.2f}s{Colors.END}")
            
            return jsonify({
                'status': 'success',
                'plot': plot_data,
                'elapsed_time': f"{elapsed:.2f}s"
            })
        else:
            print(f"{Colors.YELLOW}⚠ No plots found{Colors.END}")
            return jsonify({'error': 'No plots found'}), 404
    except Exception as e:
        print(f"{Colors.RED}✗ Visualization failed: {str(e)}{Colors.END}")
        return jsonify({'error': str(e)}), 500

@app.before_request
def before_request():
    if request.path.startswith('/api'):
        print(f"{Colors.BOLD}➤ {Colors.END}", end='')

@app.after_request
def after_request(response):
    return response

if __name__ == '__main__':
    print_logo()
    print(f"{Colors.BOLD}Device:{Colors.END} {device}")
    if device.type == 'cuda':
        print(f"{Colors.BOLD}GPU:{Colors.END} {torch.cuda.get_device_name(0)}")
    print(f"{Colors.BOLD}Server:{Colors.END} http://localhost:5000")
    print(f"{Colors.BOLD}Started:{Colors.END} {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\n" + "="*60 + "\n")
    app.run(debug=True, host='0.0.0.0', port=5000)