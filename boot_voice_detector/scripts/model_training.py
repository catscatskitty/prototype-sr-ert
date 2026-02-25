# scripts/model_training.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import pandas as pd
import numpy as np
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.device_utils import get_device
from src.utils.logger import setup_logger
from src.models.neural_networks.cnn_model import CNN1D
from scripts.data_preprocessing import prepare_dataset


def train_cnn_model(epochs=50, patience=10):
    device = get_device()
    logger = setup_logger(__name__)
    logger.info(f"Using device: {device}")
    
    train_dataset, test_dataset, _ = prepare_dataset()
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16)
    
    input_dim = train_dataset[0][0].shape[0]
    model = CNN1D(input_dim=input_dim).to(device)
    
    # Compute class counts and positive class weight for imbalance handling
    with torch.no_grad():
        class_counts_tensor = torch.bincount(train_dataset.labels, minlength=2)
        neg_count = class_counts_tensor[0].item()
        pos_count = class_counts_tensor[1].item()
    total = neg_count + pos_count
    logger.info(f"Class distribution in training set - real: {neg_count}, fake: {pos_count}, total: {total}")
    # For BCEWithLogitsLoss, pos_weight scales the loss of the positive class.
    # We want to up-weight positives when they are rarer: pos_weight = neg_count / pos_count.
    if pos_count == 0:
        pos_weight_value = 1.0
        logger.warning("No positive samples found; using pos_weight=1.0")
    else:
        pos_weight_value = neg_count / float(pos_count)
    pos_weight = torch.tensor([pos_weight_value], device=device, dtype=torch.float32)
    
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    best_loss = float('inf')
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': [], 'val_acc': [], 'val_f1': []}
    
    os.makedirs('saved_models', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    logger.info(f"Starting training for {epochs} epochs...")
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        batch_count = 0
        
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels.float().unsqueeze(1))
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            batch_count += 1
            
            if batch_idx % 10 == 0:
                logger.debug(
                    f"Epoch {epoch+1}/{epochs}, Batch {batch_idx}/{len(train_loader)}, "
                    f"Batch loss: {loss.item():.4f}"
                )
        
        model.eval()
        val_loss = 0
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels.float().unsqueeze(1))
                val_loss += loss.item()
                probs = torch.sigmoid(outputs).cpu().numpy().reshape(-1)
                preds = (probs > 0.5).astype(int)
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs)
        
        train_loss /= len(train_loader)
        val_loss /= len(test_loader)
        val_acc = accuracy_score(all_labels, all_preds)
        val_f1 = f1_score(all_labels, all_preds, average='binary', zero_division=0)

        # Tune decision threshold on validation set to maximize F1
        best_thr = 0.5
        best_thr_f1 = val_f1
        probs_arr = np.array(all_probs)
        labels_arr = np.array(all_labels)
        if len(probs_arr) > 0:
            for thr in np.linspace(0.1, 0.9, 17):
                thr_preds = (probs_arr >= thr).astype(int)
                thr_f1 = f1_score(labels_arr, thr_preds, average='binary', zero_division=0)
                if thr_f1 > best_thr_f1:
                    best_thr_f1 = thr_f1
                    best_thr = float(thr)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)
        
        # Log epoch metrics and prediction distribution
        fake_rate = float(np.mean(all_preds)) if len(all_preds) > 0 else 0.0
        real_rate = 1.0 - fake_rate
        logger.info(
            f"Epoch {epoch+1}: "
            f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, "
            f"val_acc={val_acc:.4f}, val_f1={val_f1:.4f}, "
            f"best_thr={best_thr:.3f}, best_thr_f1={best_thr_f1:.4f}, "
            f"pred_fake_rate={fake_rate:.3f}, pred_real_rate={real_rate:.3f}"
        )
        
        if val_loss < best_loss:
            best_loss = val_loss
            # Save a richer checkpoint so the API can inspect metrics if needed
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "val_loss": val_loss,
                "val_acc": val_acc,
                "val_f1": val_f1,
                "epoch": epoch + 1,
                "pos_weight": pos_weight_value,
                "decision_threshold": best_thr,
            }
            torch.save(checkpoint, 'saved_models/best_cnn_model.pth')
            patience_counter = 0
            logger.info("  -> New best model checkpoint saved.")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
    
        if (epoch + 1) % 5 == 0:
            cm = confusion_matrix(all_labels, all_preds)
            logger.info(f"Confusion matrix at epoch {epoch+1}:\n{cm}")

    pd.DataFrame(history).to_csv('results/training_history.csv', index=False)

    # Calculate precision/recall per class (robust to zero-division)
    class_precision = dict()
    class_recall = dict()

    for i in range(2):
        tp = np.sum((all_labels == i) & (all_preds == i))
        fp = np.sum((all_labels != i) & (all_preds == i))
        fn = np.sum((all_labels == i) & (all_preds != i))

        precision_den = tp + fp
        recall_den = tp + fn

        precision = tp / precision_den if precision_den > 0 else 0.0
        recall = tp / recall_den if recall_den > 0 else 0.0

        class_precision[i] = precision
        class_recall[i] = recall

    print("Precision/Recall per Class:")
    for i in range(2):
        logger.info(f"Class {i}: Precision={class_precision[i]:.4f}, Recall={class_recall[i]:.4f}")
    
    # Save predictions to CSV
    pd.DataFrame({'predictions': all_preds, 'labels': all_labels}).to_csv('results/predictions.csv', index=False)
    
    logger.info("Training completed.")
    
    return model, history

if __name__ == '__main__':
    train_cnn_model()