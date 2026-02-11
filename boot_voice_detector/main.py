import subprocess
import sys
import os

print("1. Training models...")
subprocess.run([sys.executable, "scripts/model_training.py"])

print("\n2. Benchmarking models...")
subprocess.run([sys.executable, "scripts/benchmark_models.py"])

print("\n3. Comparing models...")
subprocess.run([sys.executable, "scripts/compare_models.py"])

print("\nDone!")