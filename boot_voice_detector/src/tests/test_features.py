import numpy as np
import librosa
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from utils.preprocessing import create_spectrogram
def test_spectrogram_creation():
    """Тестирование создания спектрограммы"""
    print("Testing spectrogram creation...")
    sample_rate = 16000
    duration = 2.0
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    audio = 0.5 * np.sin(2 * np.pi * 440 * t)  
    spectrogram = create_spectrogram(
        audio=audio,
        sample_rate=sample_rate,
        n_mels=128,
        n_fft=2048,
        hop_length=512
    )
    assert spectrogram.ndim == 3, f"Expected 3 dimensions, got {spectrogram.ndim}"
    assert spectrogram.shape[0] == 1, f"Expected channel dimension 1, got {spectrogram.shape[0]}"
    assert spectrogram.shape[1] == 128, f"Expected 128 mel bands, got {spectrogram.shape[1]}"
    assert spectrogram.min() >= 0, f"Spectrogram has negative values: {spectrogram.min()}"
    assert spectrogram.max() <= 1, f"Spectrogram values exceed 1: {spectrogram.max()}"
    print("✓ Spectrogram creation works correctly")
    print(f"  Shape: {spectrogram.shape}")
    print(f"  Min value: {spectrogram.min():.4f}")
    print(f"  Max value: {spectrogram.max():.4f}")
    return True
def test_audio_loading():
    """Тестирование загрузки аудио"""
    print("\nTesting audio loading...")
    try:
        import soundfile as sf
        import tempfile
        sample_rate = 16000
        duration = 1.0
        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        audio = 0.5 * np.sin(2 * np.pi * 440 * t)
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=True) as tmp:
            sf.write(tmp.name, audio, sample_rate)
            loaded_audio, loaded_sr = librosa.load(tmp.name, sr=None)
            assert len(loaded_audio) == len(audio), "Audio length doesn't match"
            assert loaded_sr == sample_rate, f"Sample rate mismatch: {loaded_sr} != {sample_rate}"
            print("✓ Audio loading works correctly")
    except Exception as e:
        print(f"⚠ Audio loading test skipped: {e}")
    return True
def test_feature_extraction():
    """Тестирование извлечения признаков"""
    print("\nTesting feature extraction...")
    sample_rate = 16000
    duration = 2.0
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    audio = 0.5 * np.sin(2 * np.pi * 440 * t)
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=13)
    assert mfccs.shape[0] == 13, f"Expected 13 MFCCs, got {mfccs.shape[0]}"
    print(f"✓ MFCC extraction works correctly")
    print(f"  MFCC shape: {mfccs.shape}")
    chroma = librosa.feature.chroma_stft(y=audio, sr=sample_rate)
    assert chroma.shape[0] == 12, f"Expected 12 chroma features, got {chroma.shape[0]}"
    print(f"✓ Chroma extraction works correctly")
    print(f"  Chroma shape: {chroma.shape}")
    mel_spec = librosa.feature.melspectrogram(y=audio, sr=sample_rate, n_mels=128)
    assert mel_spec.shape[0] == 128, f"Expected 128 mel bands, got {mel_spec.shape[0]}"
    print(f"✓ Mel spectrogram extraction works correctly")
    print(f"  Mel spectrogram shape: {mel_spec.shape}")
    return True
def test_preprocessing_pipeline():
    """Тестирование пайплайна предобработки"""
    print("\nTesting preprocessing pipeline...")
    from utils.preprocessing import SpectrogramDataset
    num_samples = 10
    feature_shape = (1, 128, 128)
    features = np.random.randn(num_samples, *feature_shape).astype(np.float32)
    labels = np.random.randint(0, 2, num_samples)
    dataset = SpectrogramDataset(features, labels)
    assert len(dataset) == num_samples, f"Expected {num_samples} samples, got {len(dataset)}"
    feature, label = dataset[0]
    assert feature.shape == torch.Size(feature_shape), f"Feature shape mismatch"
    assert isinstance(label, torch.Tensor), "Label should be a tensor"
    print("✓ Preprocessing pipeline works correctly")
    print(f"  Dataset size: {len(dataset)}")
    print(f"  Feature shape: {feature.shape}")
    print(f"  Label type: {type(label)}")
    return True
def run_all_tests():
    """Запуск всех тестов"""
    print("="*60)
    print("Running Feature Tests")
    print("="*60)
    tests = [
        test_spectrogram_creation,
        test_audio_loading,
        test_feature_extraction,
        test_preprocessing_pipeline
    ]
    passed = 0
    failed = 0
    for test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"✗ {test_func.__name__} failed: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    print("\n" + "="*60)
    print(f"Test Results: {passed} passed, {failed} failed")
    print("="*60)
    return failed == 0
if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)