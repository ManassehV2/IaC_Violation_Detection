import torch
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score

def compute_enhanced_metrics(pred):
    logits, labels = pred.predictions, pred.label_ids
    probs = torch.sigmoid(torch.tensor(logits)).numpy()
    preds = (probs > 0.5).astype(int)

    print("[DEBUG] Prediction stats:")
    print(f"  - Logits shape: {logits.shape}")
    print(f"  - Labels shape: {labels.shape}")
    print(f"  - Probs range: [{probs.min():.4f}, {probs.max():.4f}]")
    print(f"  - Probs mean: {probs.mean():.4f}")
    print(f"  - Predictions sum: {preds.sum()} / {preds.size} ({preds.mean():.4f})")
    print(f"  - True labels sum: {labels.sum()} / {labels.size} ({labels.mean():.4f})")

    return {
        "f1_micro": f1_score(labels, preds, average="micro", zero_division=0),
        "f1_macro": f1_score(labels, preds, average="macro", zero_division=0),
        "precision": precision_score(labels, preds, average="micro", zero_division=0),
        "recall": recall_score(labels, preds, average="micro", zero_division=0),
    }