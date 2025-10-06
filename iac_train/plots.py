import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score
)
from sklearn.calibration import calibration_curve

def plot_training_history(history, output_dir, model_name):
    os.makedirs(output_dir, exist_ok=True)
    if not history or not history.get('train_loss'):
        return
    plt.figure(figsize=(12, 8))
    epochs = list(range(1, len(history['train_loss']) + 1))

    plt.subplot(2, 2, 1)
    plt.plot(epochs, history['train_loss'], 'b-', label='Training Loss')
    plt.plot(epochs, history['val_loss'], 'r-', label='Validation Loss')
    plt.title(f'{model_name} - Loss Curves')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend(); plt.grid(True, alpha=0.3)

    plt.subplot(2, 2, 2)
    plt.plot(epochs, history['train_f1'], 'b-', label='Training F1')
    plt.plot(epochs, history['val_f1'], 'r-', label='Validation F1')
    plt.title(f'{model_name} - F1 Score')
    plt.xlabel('Epoch'); plt.ylabel('F1'); plt.legend(); plt.grid(True, alpha=0.3)

    plt.subplot(2, 2, 3)
    plt.plot(epochs, history['train_precision'], 'b-', label='Training Precision')
    plt.plot(epochs, history['val_precision'], 'r-', label='Validation Precision')
    plt.title(f'{model_name} - Precision')
    plt.xlabel('Epoch'); plt.ylabel('Precision'); plt.legend(); plt.grid(True, alpha=0.3)

    plt.subplot(2, 2, 4)
    plt.plot(epochs, history['train_recall'], 'b-', label='Training Recall')
    plt.plot(epochs, history['val_recall'], 'r-', label='Validation Recall')
    plt.title(f'{model_name} - Recall')
    plt.xlabel('Epoch'); plt.ylabel('Recall'); plt.legend(); plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/training_history.png", dpi=300, bbox_inches='tight')
    plt.close()

def plot_confusion_matrix(y_true, y_pred, output_dir, model_name):
    os.makedirs(output_dir, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['No Violation', 'Violation'],
                yticklabels=['No Violation', 'Violation'])
    plt.title(f'{model_name} - Binary Confusion Matrix')
    plt.ylabel('True Label'); plt.xlabel('Predicted Label')
    plt.tight_layout(); plt.savefig(f"{output_dir}/confusion_matrix_binary.png", dpi=300, bbox_inches='tight'); plt.close()

def plot_multilabel_confusion_matrices(y_true, y_pred, output_dir, model_name, class_names, top_k=10):
    os.makedirs(output_dir, exist_ok=True)
    from sklearn.metrics import f1_score, confusion_matrix
    label_f1 = []
    for i in range(y_true.shape[1]):
        f1 = f1_score(y_true[:, i], y_pred[:, i], zero_division=0)
        label_f1.append((f1, i, class_names[i]))
    label_f1.sort(reverse=True, key=lambda x: x[0])
    top_labels = label_f1[:top_k]
    bottom_labels = label_f1[-top_k:]

    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    fig.suptitle(f'{model_name} - Top {top_k} Labels', fontsize=16)
    for idx, (f1, li, name) in enumerate(top_labels):
        r, c = idx // 5, idx % 5
        cm = confusion_matrix(y_true[:, li], y_pred[:, li])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[r, c],
                    xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'])
        axes[r, c].set_title(f'{name[:20]}...\nF1: {f1:.3f}', fontsize=10)
        axes[r, c].set_xlabel('Predicted'); axes[r, c].set_ylabel('True')
    plt.tight_layout(); plt.savefig(f"{output_dir}/confusion_matrices_top_{top_k}.png", dpi=300, bbox_inches='tight'); plt.close()

    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    fig.suptitle(f'{model_name} - Bottom {top_k} Labels', fontsize=16)
    for idx, (f1, li, name) in enumerate(bottom_labels):
        r, c = idx // 5, idx % 5
        cm = confusion_matrix(y_true[:, li], y_pred[:, li])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', ax=axes[r, c],
                    xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'])
        axes[r, c].set_title(f'{name[:20]}...\nF1: {f1:.3f}', fontsize=10)
        axes[r, c].set_xlabel('Predicted'); axes[r, c].set_ylabel('True')
    plt.tight_layout(); plt.savefig(f"{output_dir}/confusion_matrices_bottom_{top_k}.png", dpi=300, bbox_inches='tight'); plt.close()

def plot_label_distribution(y_true, output_dir, model_name, class_names):
    os.makedirs(output_dir, exist_ok=True)
    label_counts = y_true.sum(axis=0)
    label_percentages = (label_counts / len(y_true)) * 100
    sorted_idx = np.argsort(label_counts)[::-1]

    plt.figure(figsize=(15, 8))
    top = sorted_idx[:20]
    counts = label_counts[top]
    names = [class_names[i][:30] + "..." if len(class_names[i]) > 30 else class_names[i] for i in top]
    bars = plt.bar(range(len(counts)), counts, color='skyblue', edgecolor='navy', alpha=0.7)
    plt.xlabel('Security Rules'); plt.ylabel('Number of Violations')
    plt.title(f'{model_name} - Top 20 Most Frequent Security Violations')
    plt.xticks(range(len(names)), names, rotation=45, ha='right')
    for b, c in zip(bars, counts):
        plt.text(b.get_x() + b.get_width()/2, b.get_height() + 0.01*max(counts), f'{int(c)}', ha='center', va='bottom', fontsize=8)
    plt.tight_layout(); plt.savefig(f"{output_dir}/label_distribution_top20.png", dpi=300, bbox_inches='tight'); plt.close()

    plt.figure(figsize=(12, 6))
    plt.hist(label_percentages, bins=30, alpha=0.7, color='coral', edgecolor='black')
    plt.xlabel('Violation Percentage (%)'); plt.ylabel('Number of Rules')
    plt.title(f'{model_name} - Class Imbalance Distribution')
    plt.axvline(label_percentages.mean(), color='red', linestyle='--', label=f'Mean: {label_percentages.mean():.2f}%')
    plt.legend(); plt.tight_layout(); plt.savefig(f"{output_dir}/class_imbalance_distribution.png", dpi=300, bbox_inches='tight'); plt.close()

def plot_performance_metrics(y_true, y_pred, y_scores, output_dir, model_name, class_names):
    os.makedirs(output_dir, exist_ok=True)
    from sklearn.metrics import precision_score, recall_score, f1_score
    precision_scores, recall_scores, f1_scores, support_scores = [], [], [], []
    for i in range(y_true.shape[1]):
        precision_scores.append(precision_score(y_true[:, i], y_pred[:, i], zero_division=0))
        recall_scores.append(recall_score(y_true[:, i], y_pred[:, i], zero_division=0))
        f1_scores.append(f1_score(y_true[:, i], y_pred[:, i], zero_division=0))
        support_scores.append(y_true[:, i].sum())

    import pandas as pd
    df = pd.DataFrame({
        'Label': class_names,
        'Precision': precision_scores,
        'Recall': recall_scores,
        'F1-Score': f1_scores,
        'Support': support_scores
    }).sort_values('F1-Score', ascending=False)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'{model_name} - Per-Label Performance Metrics', fontsize=16)

    axes[0,0].hist(precision_scores, bins=20, alpha=0.7, color='blue', edgecolor='black')
    axes[0,0].set_xlabel('Precision'); axes[0,0].set_ylabel('Number of Labels'); axes[0,0].set_title('Precision Distribution')
    axes[0,0].axvline(np.mean(precision_scores), color='red', linestyle='--', label=f'Mean: {np.mean(precision_scores):.3f}'); axes[0,0].legend()

    axes[0,1].hist(recall_scores, bins=20, alpha=0.7, color='green', edgecolor='black')
    axes[0,1].set_xlabel('Recall'); axes[0,1].set_ylabel('Number of Labels'); axes[0,1].set_title('Recall Distribution')
    axes[0,1].axvline(np.mean(recall_scores), color='red', linestyle='--', label=f'Mean: {np.mean(recall_scores):.3f}'); axes[0,1].legend()

    axes[1,0].hist(f1_scores, bins=20, alpha=0.7, color='purple', edgecolor='black')
    axes[1,0].set_xlabel('F1-Score'); axes[1,0].set_ylabel('Number of Labels'); axes[1,0].set_title('F1-Score Distribution')
    axes[1,0].axvline(np.mean(f1_scores), color='red', linestyle='--', label=f'Mean: {np.mean(f1_scores):.3f}'); axes[1,0].legend()

    scatter = axes[1,1].scatter(recall_scores, precision_scores, c=support_scores, cmap='viridis', alpha=0.7, s=50)
    axes[1,1].set_xlabel('Recall'); axes[1,1].set_ylabel('Precision'); axes[1,1].set_title('Precision vs Recall (colored by support)')
    plt.colorbar(scatter, ax=axes[1,1], label='Support')

    plt.tight_layout(); plt.savefig(f"{output_dir}/performance_metrics_overview.png", dpi=300, bbox_inches='tight'); plt.close()

    top15, bot15 = df.head(15), df.tail(15)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))
    x = range(len(top15)); ax1.bar(x, top15['F1-Score'], alpha=0.7, color='green', edgecolor='black')
    ax1.set_xlabel('Security Rules'); ax1.set_ylabel('F1-Score'); ax1.set_title('Top 15 Performing Security Rules')
    ax1.set_xticks(range(len(top15))); ax1.set_xticklabels([l[:25]+"..." if len(l)>25 else l for l in top15['Label']], rotation=45, ha='right')

    x = range(len(bot15)); ax2.bar(x, bot15['F1-Score'], alpha=0.7, color='red', edgecolor='black')
    ax2.set_xlabel('Security Rules'); ax2.set_ylabel('F1-Score'); ax2.set_title('Bottom 15 Performing Security Rules')
    ax2.set_xticks(range(len(bot15))); ax2.set_xticklabels([l[:25]+"..." if len(l)>25 else l for l in bot15['Label']], rotation=45, ha='right')

    plt.tight_layout(); plt.savefig(f"{output_dir}/top_bottom_performers.png", dpi=300, bbox_inches='tight'); plt.close()

def plot_prediction_confidence(y_scores, y_true, output_dir, model_name):
    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    plt.hist(y_scores.flatten(), bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    plt.xlabel('Prediction Confidence'); plt.ylabel('Frequency')
    plt.title('Overall Prediction Confidence Distribution')
    plt.axvline(0.5, color='red', linestyle='--', label='Decision Threshold'); plt.legend()

    plt.subplot(2, 2, 2)
    positive_scores = y_scores[y_true == 1]; negative_scores = y_scores[y_true == 0]
    plt.hist(negative_scores, bins=30, alpha=0.7, label='True Negatives', color='blue')
    plt.hist(positive_scores, bins=30, alpha=0.7, label='True Positives', color='red')
    plt.xlabel('Prediction Confidence'); plt.ylabel('Frequency')
    plt.title('Confidence Distribution by True Label'); plt.legend()

    plt.subplot(2, 2, 3)
    y_true_flat = y_true.flatten(); y_scores_flat = y_scores.flatten()
    frac_pos, mean_pred = calibration_curve(y_true_flat, y_scores_flat, n_bins=10)
    plt.plot(mean_pred, frac_pos, "s-", label="Model")
    plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    plt.xlabel('Mean Predicted Probability'); plt.ylabel('Fraction of Positives')
    plt.title('Calibration Plot'); plt.legend()

    plt.subplot(2, 2, 4)
    high_thr, low_thr = 0.9, 0.1
    high_conf_correct = ((y_scores > high_thr) & (y_true == 1)).sum() + ((y_scores < low_thr) & (y_true == 0)).sum()
    high_conf_total = ((y_scores > high_thr) | (y_scores < low_thr)).sum()
    medium_mask = (y_scores >= low_thr) & (y_scores <= high_thr)
    medium_conf_correct = ((y_scores[medium_mask] > 0.5) == y_true[medium_mask]).sum()
    medium_conf_total = medium_mask.sum()

    cats = ['High Confidence (>0.9 or <0.1)', 'Medium Confidence (0.1-0.9)']
    accs = [high_conf_correct / max(high_conf_total, 1), medium_conf_correct / max(medium_conf_total, 1)]
    counts = [high_conf_total, medium_conf_total]
    bars = plt.bar(cats, accs, color=['green', 'orange'], alpha=0.7)
    plt.ylabel('Accuracy'); plt.title('Accuracy by Confidence Level'); plt.ylim(0, 1)
    for b, n in zip(bars, counts):
        plt.text(b.get_x() + b.get_width()/2, b.get_height() + 0.02, f'n={int(n)}', ha='center', va='bottom')

    plt.tight_layout(); plt.savefig(f"{output_dir}/prediction_confidence_analysis.png", dpi=300, bbox_inches='tight'); plt.close()

def plot_threshold_analysis(y_true, y_scores, output_dir, model_name):
    os.makedirs(output_dir, exist_ok=True)
    thresholds = np.arange(0.1, 1.0, 0.02)
    from sklearn.metrics import precision_score, recall_score, f1_score
    y_true_bin = (y_true.sum(axis=1) > 0).astype(int)
    y_scores_bin = y_scores.max(axis=1)

    precisions, recalls, f1s, accs = [], [], [], []
    for t in thresholds:
        y_pred_t = (y_scores_bin >= t).astype(int)
        precisions.append(precision_score(y_true_bin, y_pred_t, zero_division=0))
        recalls.append(recall_score(y_true_bin, y_pred_t, zero_division=0))
        f1s.append(f1_score(y_true_bin, y_pred_t, zero_division=0))
        accs.append((y_true_bin == y_pred_t).mean())

    opt_idx = np.argmax(f1s)
    opt_t, opt_f1 = thresholds[opt_idx], f1s[opt_idx]

    plt.figure(figsize=(12, 8))
    plt.subplot(2, 2, 1)
    plt.plot(thresholds, precisions, 'b-', label='Precision', linewidth=2)
    plt.plot(thresholds, recalls, 'r-', label='Recall', linewidth=2)
    plt.plot(thresholds, f1s, 'g-', label='F1-Score', linewidth=2)
    plt.axvline(opt_t, color='black', linestyle='--', label=f'Optimal (F1={opt_f1:.3f})')
    plt.xlabel('Threshold'); plt.ylabel('Score')
    plt.title('Precision, Recall, F1 vs Threshold'); plt.legend(); plt.grid(True, alpha=0.3)

    plt.subplot(2, 2, 2)
    plt.plot(thresholds, accs, 'purple', linewidth=2)
    plt.axvline(opt_t, color='black', linestyle='--', label='Optimal threshold')
    plt.xlabel('Threshold'); plt.ylabel('Accuracy'); plt.title('Accuracy vs Threshold'); plt.legend(); plt.grid(True, alpha=0.3)

    plt.subplot(2, 2, 3)
    plt.plot(recalls, precisions, 'b-', linewidth=2)
    plt.xlabel('Recall'); plt.ylabel('Precision'); plt.title('Precision-Recall Curve'); plt.grid(True, alpha=0.3)

    plt.subplot(2, 2, 4)
    plt.hist(y_scores_bin[y_true_bin == 0], bins=30, alpha=0.7, label='No Violation', color='blue', density=True)
    plt.hist(y_scores_bin[y_true_bin == 1], bins=30, alpha=0.7, label='Violation', color='red', density=True)
    plt.axvline(opt_t, color='black', linestyle='--', label=f'Optimal: {opt_t:.3f}')
    plt.axvline(0.5, color='gray', linestyle=':', label='Default: 0.5')
    plt.xlabel('Prediction Score'); plt.ylabel('Density'); plt.title('Score Distribution by Class'); plt.legend()

    plt.tight_layout(); plt.savefig(f"{output_dir}/threshold_analysis.png", dpi=300, bbox_inches='tight'); plt.close()
    return float(opt_t), float(opt_f1)

def plot_roc_pr_curves(y_true_bin, y_scores_bin, output_dir, model_name):
    os.makedirs(output_dir, exist_ok=True)
    fpr, tpr, _ = roc_curve(y_true_bin, y_scores_bin); roc_auc = auc(fpr, tpr)
    prec, rec, _ = precision_recall_curve(y_true_bin, y_scores_bin); pr_auc = average_precision_score(y_true_bin, y_scores_bin)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate'); plt.title(f'{model_name} - ROC Curve'); plt.legend(); plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(rec, prec, color='darkorange', lw=2, label=f'PR curve (AUC = {pr_auc:.3f})')
    plt.xlabel('Recall'); plt.ylabel('Precision'); plt.title(f'{model_name} - Precision-Recall Curve'); plt.legend(); plt.grid(True, alpha=0.3)

    plt.tight_layout(); plt.savefig(f"{output_dir}/roc_pr_curves.png", dpi=300, bbox_inches='tight'); plt.close()