import subprocess
import sys
import os
import torch

def main():
    print("="*60)
    print("AUDIO DEEPFAKE DETECTION PIPELINE")
    print("="*60)
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
    else:
        print("CUDA not available, using CPU")
    
    steps = [
        ("Data Preprocessing", "scripts/data_preprocessing.py"),
        ("Model Training", "scripts/model_training.py"),
        ("Benchmarking (inference only)", "scripts/benchmark_models.py"),
        ("Comparison", "scripts/compare_models.py")
    ]
    
    for step_name, script in steps:
        print(f"\n{'='*60}")
        print(f"STEP: {step_name}")
        print('='*60)
        
        result = subprocess.run([sys.executable, script])
        if result.returncode != 0:
            print(f"✗ {step_name} failed!")
            sys.exit(1)
        print(f"✓ {step_name} completed successfully")
    
    print("\n" + "="*60)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("="*60)

if __name__ == '__main__':
    main()