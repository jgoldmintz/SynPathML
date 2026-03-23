#!/usr/bin/env python3
# SynPathML
# Copyright (C) 2023-2026  Jacob Goldmintz
# All rights reserved. See LICENSE for terms.

"""
Shared utilities for ML pipeline.
Provides metrics computation, calibration, plotting, and logging functions.
"""

import logging
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_recall_curve,
    average_precision_score,
    roc_auc_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    brier_score_loss,
)


def setup_logging(output_dir: str, name: str = "ml_pipeline") -> logging.Logger:
    """Configure logging to both file and console."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if logger.handlers:
        logger.handlers.clear()

    file_handler = logging.FileHandler(output_path / f"{name}.log")
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(file_formatter)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter("%(levelname)s: %(message)s")
    console_handler.setFormatter(console_formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    Compute classification metrics for imbalanced binary classification.

    Args:
        y_true: Ground truth labels (0/1)
        y_pred: Predicted labels (0/1)
        y_prob: Predicted probabilities for positive class
        threshold: Classification threshold (default 0.5)

    Returns:
        Dictionary with PR-AUC, ROC-AUC, F1, MCC, precision@recall thresholds
    """
    metrics = {}

    # PR-AUC (primary metric for imbalanced data)
    metrics["pr_auc"] = average_precision_score(y_true, y_prob)

    # ROC-AUC (for reference, but less meaningful with imbalance)
    try:
        metrics["roc_auc"] = roc_auc_score(y_true, y_prob)
    except ValueError:
        metrics["roc_auc"] = np.nan

    # F1 score
    metrics["f1"] = f1_score(y_true, y_pred, zero_division=0)

    # Matthews Correlation Coefficient (robust to imbalance)
    metrics["mcc"] = matthews_corrcoef(y_true, y_pred)

    # Brier score (calibration)
    metrics["brier_score"] = brier_score_loss(y_true, y_prob)

    # Confusion matrix components
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    metrics["true_positives"] = int(tp)
    metrics["false_positives"] = int(fp)
    metrics["true_negatives"] = int(tn)
    metrics["false_negatives"] = int(fn)

    # Precision and recall at threshold
    if tp + fp > 0:
        metrics["precision"] = tp / (tp + fp)
    else:
        metrics["precision"] = 0.0

    if tp + fn > 0:
        metrics["recall"] = tp / (tp + fn)
    else:
        metrics["recall"] = 0.0

    # Precision at various recall thresholds
    precision_curve, recall_curve, thresholds = precision_recall_curve(y_true, y_prob)

    for target_recall in [0.3, 0.5, 0.7]:
        idx = np.searchsorted(recall_curve[::-1], target_recall)
        if idx < len(precision_curve):
            metrics[f"precision_at_recall_{target_recall}"] = float(
                precision_curve[::-1][idx]
            )
        else:
            metrics[f"precision_at_recall_{target_recall}"] = 0.0

    return metrics


def compute_metrics_by_subset(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    is_synonymous: np.ndarray,
    threshold: float = 0.5
) -> Dict[str, Dict[str, float]]:
    """
    Compute metrics separately for synonymous and nonsynonymous subsets.

    Args:
        y_true: Ground truth labels
        y_prob: Predicted probabilities
        is_synonymous: Boolean array indicating synonymous mutations
        threshold: Classification threshold

    Returns:
        Dictionary with 'all', 'synonymous', 'nonsynonymous' metric dicts
    """
    y_pred = (y_prob >= threshold).astype(int)

    results = {
        "all": compute_metrics(y_true, y_pred, y_prob, threshold)
    }

    syn_mask = is_synonymous.astype(bool)
    if syn_mask.sum() > 0:
        results["synonymous"] = compute_metrics(
            y_true[syn_mask],
            y_pred[syn_mask],
            y_prob[syn_mask],
            threshold
        )

    nonsyn_mask = ~syn_mask
    if nonsyn_mask.sum() > 0:
        results["nonsynonymous"] = compute_metrics(
            y_true[nonsyn_mask],
            y_pred[nonsyn_mask],
            y_prob[nonsyn_mask],
            threshold
        )

    return results


def calibration_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute calibration curve (reliability diagram data).

    Args:
        y_true: Ground truth labels
        y_prob: Predicted probabilities
        n_bins: Number of bins for calibration

    Returns:
        Tuple of (bin_centers, fraction_positives, bin_counts)
    """
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    fraction_positives = np.zeros(n_bins)
    bin_counts = np.zeros(n_bins)

    for i in range(n_bins):
        mask = (y_prob >= bin_edges[i]) & (y_prob < bin_edges[i + 1])
        bin_counts[i] = mask.sum()
        if bin_counts[i] > 0:
            fraction_positives[i] = y_true[mask].mean()

    return bin_centers, fraction_positives, bin_counts


def expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10
) -> float:
    """
    Compute Expected Calibration Error (ECE).

    Lower is better. ECE = 0 means perfectly calibrated.
    """
    bin_centers, fraction_positives, bin_counts = calibration_curve(
        y_true, y_prob, n_bins
    )

    total = bin_counts.sum()
    if total == 0:
        return 0.0

    ece = np.sum(
        bin_counts * np.abs(fraction_positives - bin_centers)
    ) / total

    return float(ece)


def compute_class_weights(y: np.ndarray) -> Dict[int, float]:
    """
    Compute balanced class weights for imbalanced classification.

    Args:
        y: Label array (0/1)

    Returns:
        Dictionary mapping class label to weight
    """
    n_samples = len(y)
    n_classes = 2

    class_counts = np.bincount(y.astype(int), minlength=2)

    weights = n_samples / (n_classes * class_counts)

    return {0: float(weights[0]), 1: float(weights[1])}


def save_json(data: Dict, filepath: Union[str, Path]) -> None:
    """Save dictionary to JSON file."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, "w") as f:
        json.dump(data, f, indent=2, default=str)


def load_json(filepath: Union[str, Path]) -> Dict:
    """Load dictionary from JSON file."""
    with open(filepath, "r") as f:
        return json.load(f)


def save_results_summary(
    metrics: Dict,
    output_dir: Union[str, Path],
    filename: str = "metrics.json"
) -> None:
    """Save metrics summary to JSON."""
    save_json(metrics, Path(output_dir) / filename)


# Plotting functions (optional matplotlib dependency)
def plot_pr_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    save_path: Optional[str] = None,
    title: str = "Precision-Recall Curve"
) -> None:
    """Plot precision-recall curve."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plot")
        return

    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = average_precision_score(y_true, y_prob)

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, linewidth=2, label=f"PR-AUC = {pr_auc:.3f}")

    # Baseline (random classifier)
    baseline = y_true.mean()
    plt.axhline(y=baseline, color="gray", linestyle="--", label=f"Baseline = {baseline:.3f}")

    plt.xlabel("Recall", fontsize=12)
    plt.ylabel("Precision", fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(loc="best")
    plt.grid(True, alpha=0.3)
    plt.xlim([0, 1])
    plt.ylim([0, 1])

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_feature_importance(
    importance_df: pd.DataFrame,
    importance_col: str = "RF_Importance",
    top_k: int = 20,
    save_path: Optional[str] = None,
    title: str = "Feature Importance"
) -> None:
    """Plot horizontal bar chart of feature importance."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plot")
        return

    df_sorted = importance_df.nlargest(top_k, importance_col)

    plt.figure(figsize=(10, max(6, top_k * 0.3)))
    plt.barh(
        range(len(df_sorted)),
        df_sorted[importance_col].values,
        color="steelblue"
    )
    plt.yticks(range(len(df_sorted)), df_sorted["Feature"].values)
    plt.xlabel(importance_col, fontsize=12)
    plt.title(title, fontsize=14)
    plt.gca().invert_yaxis()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_shap_summary(
    shap_values: np.ndarray,
    feature_names: List[str],
    X: np.ndarray,
    save_path: Optional[str] = None,
    max_display: int = 20
) -> None:
    """Plot SHAP summary plot."""
    try:
        import shap
        import matplotlib.pyplot as plt
    except ImportError:
        print("shap or matplotlib not available, skipping plot")
        return

    plt.figure(figsize=(10, max(6, max_display * 0.3)))
    shap.summary_plot(
        shap_values,
        X,
        feature_names=feature_names,
        max_display=max_display,
        show=False
    )

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_calibration_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
    save_path: Optional[str] = None,
    title: str = "Calibration Curve"
) -> None:
    """Plot reliability diagram."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plot")
        return

    bin_centers, fraction_positives, bin_counts = calibration_curve(
        y_true, y_prob, n_bins
    )
    ece = expected_calibration_error(y_true, y_prob, n_bins)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), gridspec_kw={"height_ratios": [3, 1]})

    # Calibration curve
    ax1.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")
    ax1.plot(bin_centers, fraction_positives, "o-", label=f"Model (ECE = {ece:.3f})")
    ax1.set_xlabel("Mean predicted probability")
    ax1.set_ylabel("Fraction of positives")
    ax1.set_title(title)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Histogram of predictions
    ax2.bar(bin_centers, bin_counts, width=1/n_bins * 0.8, alpha=0.7)
    ax2.set_xlabel("Mean predicted probability")
    ax2.set_ylabel("Count")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_attention_weights(
    weights: np.ndarray,
    mechanism_names: List[str],
    save_path: Optional[str] = None,
    title: str = "Mechanism Attention Weights"
) -> None:
    """Plot attention weights over mechanisms."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plot")
        return

    mean_weights = weights.mean(axis=0)
    std_weights = weights.std(axis=0)

    plt.figure(figsize=(10, 6))
    x = range(len(mechanism_names))
    plt.bar(x, mean_weights, yerr=std_weights, capsize=3, color="steelblue", alpha=0.8)
    plt.xticks(x, mechanism_names, rotation=45, ha="right")
    plt.ylabel("Attention Weight")
    plt.title(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_training_curves(
    train_losses: List[float],
    val_losses: List[float],
    train_metrics: Optional[List[float]] = None,
    val_metrics: Optional[List[float]] = None,
    metric_name: str = "PR-AUC",
    save_path: Optional[str] = None
) -> None:
    """Plot training and validation curves."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plot")
        return

    n_plots = 2 if train_metrics is not None else 1
    fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 5))

    if n_plots == 1:
        axes = [axes]

    # Loss curve
    epochs = range(1, len(train_losses) + 1)
    axes[0].plot(epochs, train_losses, label="Train")
    axes[0].plot(epochs, val_losses, label="Validation")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Metric curve
    if train_metrics is not None:
        axes[1].plot(epochs, train_metrics, label="Train")
        axes[1].plot(epochs, val_metrics, label="Validation")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel(metric_name)
        axes[1].set_title(f"Training {metric_name}")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()
