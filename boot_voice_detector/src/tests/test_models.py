import torch
import numpy as np
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from boot_voice_detector.src.models.neural_networks.cnn_model import SpectrogramCNN, EnhancedSpectrogramCNN
from models.neural_networks.hybrid_model import HybridCNNLSTM
def test_spectrogram_cnn():
    """Тестирование базовой CNN"""
    print("Testing SpectrogramCNN...")
    model = SpectrogramCNN(
        input_shape=(1, 128, 128),
        num_classes=2
    )
    batch_size = 4
    test_input = torch.randn(batch_size, 1, 128, 128)
    output = model(test_input)
    assert output.shape == (batch_size, 2), f"Expected output shape {(batch_size, 2)}, got {output.shape}"
    print("✓ SpectrogramCNN forward pass works correctly")
    complexity = model.get_complexity()
    assert 'total_params' in complexity
    assert 'size_mb' in complexity
    print(f"  Total parameters: {complexity['total_params']:,}")
    print(f"  Model size: {complexity['size_mb']:.2f} MB")
    return True
def test_enhanced_cnn():
    """Тестирование улучшенной CNN"""
    print("\nTesting EnhancedSpectrogramCNN...")
    model = EnhancedSpectrogramCNN(
        input_shape=(1, 128, 128),
        num_classes=2,
        use_attention=True,
        use_residual=True
    )
    batch_size = 4
    test_input = torch.randn(batch_size, 1, 128, 128)
    output = model(test_input)
    assert output.shape == (batch_size, 2), f"Expected output shape {(batch_size, 2)}, got {output.shape}"
    print("✓ EnhancedSpectrogramCNN forward pass works correctly")
    return True
def test_hybrid_model():
    """Тестирование гибридной модели"""
    print("\nTesting HybridCNNLSTM...")
    model = HybridCNNLSTM(
        spectrogram_shape=(1, 128, 128),
        lstm_input_size=50,
        num_handcrafted_features=50,
        num_classes=2
    )
    batch_size = 4
    spectrogram = torch.randn(batch_size, 1, 128, 128)
    temporal_features = torch.randn(batch_size, 100, 50)
    handcrafted_features = torch.randn(batch_size, 50)
    output, attention_weights = model(spectrogram, temporal_features, handcrafted_features)
    assert output.shape == (batch_size, 2), f"Expected output shape {(batch_size, 2)}, got {output.shape}"
    assert attention_weights.shape == (batch_size, 100), f"Expected attention shape {(batch_size, 100)}, got {attention_weights.shape}"
    print("✓ HybridCNNLSTM forward pass works correctly")
    complexity = model.get_model_complexity()
    assert 'total_params' in complexity
    print(f"  Total parameters: {complexity['total_params']:,}")
    print(f"  CNN parameters: {complexity['cnn_params']:,}")
    print(f"  LSTM parameters: {complexity['lstm_params']:,}")
    return True
def test_model_training():
    """Тестирование процесса обучения"""
    print("\nTesting model training...")
    model = SpectrogramCNN(
        input_shape=(1, 64, 64),  
        num_classes=2,
        cnn_channels=[16, 32],  
        fc_units=[64]
    )
    batch_size = 8
    num_samples = 32
    X_train = torch.randn(num_samples, 1, 64, 64)
    y_train = torch.randint(0, 2, (num_samples,))
    dataset = torch.utils.data.TensorDataset(X_train, y_train)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()
    model.train()
    for batch_X, batch_y in dataloader:
        optimizer.zero_grad()
        output = model(batch_X)
        loss = criterion(output, batch_y)
        loss.backward()
        optimizer.step()
        break  
    print("✓ Model training step works correctly")
    return True
def test_gpu_availability():
    """Проверка доступности GPU"""
    print("\nTesting GPU availability...")
    if torch.cuda.is_available():
        print(f"✓ GPU is available: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA version: {torch.version.cuda}")
        print(f"  GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("⚠ GPU is not available, using CPU")
    return True
def run_all_tests():
    """Запуск всех тестов"""
    print("="*60)
    print("Running Model Tests")
    print("="*60)
    tests = [
        test_spectrogram_cnn,
        test_enhanced_cnn,
        test_hybrid_model,
        test_model_training,
        test_gpu_availability
    ]
    passed = 0
    failed = 0
    for test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"✗ {test_func.__name__} failed: {e}")
            failed += 1
    print("\n" + "="*60)
    print(f"Test Results: {passed} passed, {failed} failed")
    print("="*60)
    return failed == 0
if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)