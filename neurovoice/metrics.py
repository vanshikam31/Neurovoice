"""Emotion-recognition metrics computed from a confusion matrix (numpy only).

UAR (unweighted average recall) = mean of per-class recall; the standard SER metric because
it isn't dominated by big classes. Classes with no true samples are ignored in UAR/macro-F1,
which matters for per-dataset breakdowns (CREMA-D has no 'surprised').
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def confusion(y_true, y_pred, n_classes: int) -> np.ndarray:
    cm = np.zeros((n_classes, n_classes), dtype=int)
    np.add.at(cm, (np.asarray(y_true), np.asarray(y_pred)), 1)
    return cm


def summarize(y_true, y_pred, class_names: list[str]) -> dict:
    n = len(class_names)
    cm = confusion(y_true, y_pred, n)
    support = cm.sum(axis=1)
    predicted = cm.sum(axis=0)
    diag = np.diag(cm)
    recall = np.divide(diag, support, out=np.zeros(n), where=support > 0)
    precision = np.divide(diag, predicted, out=np.zeros(n), where=predicted > 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros(n), where=(precision + recall) > 0)
    present = support > 0
    return {
        "n": int(cm.sum()),
        "accuracy": float(diag.sum() / max(cm.sum(), 1)),
        "uar": float(recall[present].mean()) if present.any() else 0.0,
        "macro_f1": float(f1[present].mean()) if present.any() else 0.0,
        "per_class": {class_names[i]: {"recall": float(recall[i]), "precision": float(precision[i]),
                                       "f1": float(f1[i]), "support": int(support[i])} for i in range(n)},
        "confusion_matrix": cm.tolist(),
    }


def breakdown(meta: pd.DataFrame, y_true, y_pred, by: str, class_names: list[str]) -> dict:
    """Metrics per value of a metadata column (e.g. 'dataset' or 'gender')."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    out = {}
    for value, idx in meta.groupby(by).indices.items():
        s = summarize(y_true[idx], y_pred[idx], class_names)
        out[str(value)] = {k: s[k] for k in ("n", "accuracy", "uar", "macro_f1")}
    return out


def plot_confusion(cm, class_names: list[str], path, title: str = "") -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cm = np.asarray(cm, dtype=float)
    norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)     # row-normalized = recall
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(class_names)), class_names, rotation=45, ha="right")
    ax.set_yticks(range(len(class_names)), class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            if cm[i, j] > 0:
                ax.text(j, i, f"{norm[i, j]:.2f}", ha="center", va="center",
                        color="white" if norm[i, j] > 0.5 else "black", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
