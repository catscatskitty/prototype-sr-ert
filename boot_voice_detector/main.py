# run.py (master запуск)
import subprocess
import sys
import os

os.makedirs('saved_models', exist_ok=True)
os.makedirs('results', exist_ok=True)

print("Starting data preprocessing...")
print("="*50)
result = subprocess.run([sys.executable, 'scripts/data_preprocessing.py'], capture_output=True, text=True)
if result.returncode != 0:
    print(f"Errors during data preprocessing:\n{result.stderr}")
else:
print(result.stdout)

print("\nData preprocessing complete!")

print("\nStarting model training...")
print("="*50)
result = subprocess.run([sys.executable, 'scripts/model_training.py'], capture_output=True, text=True)
if result.returncode != 0:
    print(f"Errors during model training:\n{result.stderr}")
else:
    print(result.stdout)

print("\nModel training complete!")

print("\nStarting benchmarking models...")
print("="*50)
result = subprocess.run([sys.executable, 'scripts/benchmark_models.py'], capture_output=True, text=True)
if result.returncode != 0:
    print(f"Errors during benchmarking:\n{result.stderr}")
else:
    print(result.stdout)

print("\nBenchmarking complete!")

print("\nStarting comparison of models...")
print("="*50)
result = subprocess.run([sys.executable, 'scripts/compare_models.py'], capture_output=True, text=True)
if result.returncode != 0:
    print(f"Errors during comparison:\n{result.stderr}")
else:
    print(result.stdout)

print("\nComparison complete!")
