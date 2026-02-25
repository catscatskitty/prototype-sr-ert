# scripts/__init__.py
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.data_preprocessing import load_spectrogram_data, load_vad_snr_data, prepare_dataset, AudioDataset
from scripts.model_training import train_cnn_model
from scripts.benchmark_models import benchmark_models
from scripts.compare_models import compare_and_plot