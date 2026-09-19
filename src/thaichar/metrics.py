"""Evaluation metrics for Thai glyph classification."""

from __future__ import annotations

import numpy as np
import torch


def compute_metrics(
    y_true: np.ndarray,
    y_pred_logits: np.ndarray,
    class_counts_train: np.ndarray,
    minority_thresh: int = 50,
) -> dict:
    """Compute classification metrics.

    Parameters
    ----------
    y_true : ndarray[N] int
        Ground-truth labels.
    y_pred_logits : ndarray[N, C] float
        Raw logits from the model.
    class_counts_train : ndarray[C] int
        Per-class sample counts in the training set.
    minority_thresh : int
        Classes with train count < this are considered minority.

    Returns
    -------
    dict with keys: top1, top5, balanced_acc, macro_f1, minority_acc,
         n_classes_present, per_class_recall, confusion.
    """
    n_classes = y_pred_logits.shape[1]
    y_pred = y_pred_logits.argmax(axis=1)

    # Top-1
    top1 = float((y_pred == y_true).mean())

    # Top-5
    top5_preds = np.argsort(y_pred_logits, axis=1)[:, -5:]
    top5_hits = np.array([y_true[i] in top5_preds[i] for i in range(len(y_true))])
    top5 = float(top5_hits.mean())

    # Per-class recall
    per_class_recall = np.full(n_classes, np.nan)
    classes_present = []
    for c in range(n_classes):
        mask = y_true == c
        if mask.sum() > 0:
            per_class_recall[c] = float((y_pred[mask] == c).mean())
            classes_present.append(c)

    n_classes_present = len(classes_present)

    # Balanced accuracy = mean of per-class recall over classes present
    valid_recalls = per_class_recall[~np.isnan(per_class_recall)]
    balanced_acc = float(valid_recalls.mean()) if len(valid_recalls) > 0 else 0.0

    # Per-class precision and F1 for macro_f1
    per_class_f1 = []
    for c in classes_present:
        tp = float(((y_pred == c) & (y_true == c)).sum())
        fp = float(((y_pred == c) & (y_true != c)).sum())
        fn = float(((y_pred != c) & (y_true == c)).sum())
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        per_class_f1.append(f1)
    macro_f1 = float(np.mean(per_class_f1)) if per_class_f1 else 0.0

    # Minority accuracy: recall on classes with train n < minority_thresh
    minority_classes = [c for c in range(n_classes)
                        if c < len(class_counts_train) and class_counts_train[c] < minority_thresh]
    minority_correct = 0
    minority_total = 0
    for c in minority_classes:
        mask = y_true == c
        minority_total += int(mask.sum())
        minority_correct += int((y_pred[mask] == c).sum())
    minority_acc = float(minority_correct / minority_total) if minority_total > 0 else float('nan')

    # Confusion matrix
    confusion = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        confusion[t, p] += 1

    return {
        "top1": top1,
        "top5": top5,
        "balanced_acc": balanced_acc,
        "macro_f1": macro_f1,
        "minority_acc": minority_acc,
        "n_classes_present": n_classes_present,
        "per_class_recall": per_class_recall.tolist(),
        "confusion": confusion.tolist(),
    }


def tau_sweep(
    logits: np.ndarray,
    y: np.ndarray,
    log_prior: np.ndarray,
    class_counts_train: np.ndarray,
    taus: tuple[float, ...] = (0, 0.25, 0.5, 0.75),
) -> dict:
    """Sweep logit-adjustment τ values.

    Parameters
    ----------
    logits : ndarray[N, C]
    y : ndarray[N]
    log_prior : ndarray[C]
    class_counts_train : ndarray[C]
    taus : tuple of floats

    Returns
    -------
    dict: tau -> {top1, balanced_acc, macro_f1, minority_acc}
    """
    results = {}
    for tau in taus:
        adj_logits = logits - tau * log_prior[np.newaxis, :]
        m = compute_metrics(y, adj_logits, class_counts_train)
        results[str(tau)] = {
            "top1": m["top1"],
            "balanced_acc": m["balanced_acc"],
            "macro_f1": m["macro_f1"],
            "minority_acc": m["minority_acc"],
        }
    return results
