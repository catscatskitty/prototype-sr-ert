"""Benchmark the CNN deepfake detection model on CPU/GPU and log performance."""

import os
import sys
import time

import psutil
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.device_utils import get_device
from src.models.neural_networks.cnn_model import CNN1D
from scripts.data_preprocessing import prepare_dataset


def benchmark_models(batch_size: int = 32):
    """Run a simple benchmark on the trained CNN model."""
    device = get_device()
    print(f"Benchmarking on device: {device}")

    # Prepare dataset and data loader
    train_dataset, test_dataset, _ = prepare_dataset()
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # Build model with correct input dimension
    input_dim = test_dataset[0][0].shape[0]
    model = CNN1D(input_dim=input_dim).to(device)

    # Load pre-trained weights if available
    ckpt_path = "saved_models/best_cnn_model.pth"
    if os.path.exists(ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location=device)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)
        print(f"Loaded checkpoint from {ckpt_path}")
    else:
        print(f"No checkpoint found at {ckpt_path}, benchmarking untrained model.")

    model.eval()

    # Measure CPU memory usage with psutil
    process = psutil.Process()
    initial_memory = process.memory_info().rss / 1024 / 1024

    # Track GPU memory if available (only on CUDA devices)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    # Benchmark model inference time and accuracy/F1
    start_time = time.time()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            logits = model(inputs)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).int().cpu().numpy().reshape(-1)

            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    if device.type == "cuda":
        torch.cuda.synchronize()
        gpu_memory = torch.cuda.max_memory_allocated() / 1024 / 1024
    else:
        gpu_memory = None

    # Calculate accuracy and F1 score
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)

    # Save results to benchmark_results.csv
    final_memory = process.memory_info().rss / 1024 / 1024
    peak_memory = final_memory - initial_memory

    inference_time_per_sample_ms = (
        (time.time() - start_time) / max(len(test_dataset), 1) * 1000.0
    )

    results = {
        "device": str(device),
        "accuracy": accuracy,
        "f1_score": f1,
        "inference_time_per_sample_ms": inference_time_per_sample_ms,
        "cpu_memory_mb": peak_memory,
        "gpu_memory_mb": gpu_memory,
        "total_samples": len(test_dataset),
    }

    os.makedirs("results", exist_ok=True)
    df = pd.DataFrame([results])
    df.to_csv("results/benchmark_results.csv", index=False)

    print(df)

    return results


if __name__ == "__main__":
    benchmark_models()