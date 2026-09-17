"""
utils.py — Shared visualization and utility functions.

Plotting functions for confusion matrices, temporal attention heatmaps,
and prediction summaries.  Used by both pipeline/train.py and inference/.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ------------------------------------------------------------------ #
#  Training visualizations
# ------------------------------------------------------------------ #

def plot_temporal_attention(attn_weights, labels, class_names, months, save_path):
    """Plot average temporal attention per crop class (heatmap)."""
    num_classes = len(class_names)
    T = attn_weights.shape[1]

    avg_attn = np.zeros((num_classes, T))
    for c in range(num_classes):
        mask = labels == c
        if mask.sum() > 0:
            avg_attn[c] = attn_weights[mask].mean(axis=0)

    fig, ax = plt.subplots(figsize=(10, max(4, num_classes * 0.7)))
    im = ax.imshow(avg_attn, aspect="auto", cmap="YlOrRd", interpolation="nearest")
    ax.set_yticks(range(num_classes))
    ax.set_yticklabels(class_names)
    ax.set_xticks(range(T))
    ax.set_xticklabels(months, rotation=45, ha="right")
    ax.set_xlabel("Month")
    ax.set_ylabel("Crop Type")
    ax.set_title("Temporal Attention Weights by Crop Class")

    for i in range(num_classes):
        for j in range(T):
            ax.text(j, i, f"{avg_attn[i, j]:.2f}", ha="center", va="center",
                    fontsize=8,
                    color="black" if avg_attn[i, j] < 0.5 * avg_attn.max() else "white")

    plt.colorbar(im, ax=ax, label="Attention Weight")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved temporal attention plot: {save_path}")


def plot_confusion_matrix(cm, class_names, title, save_path):
    """Plot and save confusion matrix."""
    n = len(class_names)
    fig, ax = plt.subplots(figsize=(max(8, n), max(6, n * 0.8)))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(n), yticks=np.arange(n),
           xticklabels=class_names, yticklabels=class_names,
           ylabel="True label", xlabel="Predicted label", title=title)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    thresh = cm.max() / 2.0
    for i in range(n):
        for j in range(n):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black", fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


# ------------------------------------------------------------------ #
#  Inference visualizations
# ------------------------------------------------------------------ #

def plot_prediction_attention(results: dict, sample_idx: int,
                              save_path=None):
    """Plot attention weights for a single prediction."""
    if "attention" not in results:
        print("No attention weights available (non-MSTACNN model).")
        return

    attn = results["attention"][sample_idx]
    months = results["months"]
    pred = results["predictions"][sample_idx]
    conf = results["confidence"][sample_idx]

    fig, ax = plt.subplots(figsize=(10, 4))
    bars = ax.bar(months, attn, color="#2196F3", edgecolor="navy", alpha=0.8)

    # Highlight peak month
    peak_idx = np.argmax(attn)
    bars[peak_idx].set_color("#FF5722")
    bars[peak_idx].set_edgecolor("darkred")

    ax.set_xlabel("Month", fontsize=12)
    ax.set_ylabel("Attention Weight", fontsize=12)
    ax.set_title(f"Temporal Attention — Predicted: {pred} ({conf:.1%})",
                 fontsize=13, fontweight="bold")
    ax.set_ylim(0, max(attn) * 1.25)

    for i, (bar, val) in enumerate(zip(bars, attn)):
        ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.005,
                f"{val:.3f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Attention plot saved to {save_path}")
    else:
        plt.show()
    plt.close()


def plot_batch_summary(results: dict, save_path=None):
    """Plot summary of batch predictions: class distribution + confidence."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Class distribution
    preds = results["predictions"]
    unique, counts = np.unique(preds, return_counts=True)
    sort_idx = np.argsort(counts)[::-1]
    axes[0].barh(unique[sort_idx], counts[sort_idx], color="#4CAF50", edgecolor="darkgreen")
    axes[0].set_xlabel("Count")
    axes[0].set_title("Predicted Class Distribution")

    # Confidence distribution
    axes[1].hist(results["confidence"], bins=20, color="#2196F3",
                 edgecolor="navy", alpha=0.8)
    axes[1].axvline(np.mean(results["confidence"]), color="red",
                    linestyle="--", label=f'Mean: {np.mean(results["confidence"]):.2%}')
    axes[1].set_xlabel("Confidence")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Prediction Confidence Distribution")
    axes[1].legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Summary plot saved to {save_path}")
    else:
        plt.show()
    plt.close()
