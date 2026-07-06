"""Evaluation utilities: accuracy, macro-F1, and Expected Calibration Error."""

import numpy as np
import torch
import torch.nn.functional as F


def accuracy(logits: torch.Tensor, y_true: torch.Tensor) -> float:
    preds = logits.argmax(dim=-1)
    return (preds == y_true).float().mean().item()


def macro_f1(logits: torch.Tensor, y_true: torch.Tensor, num_classes: int) -> float:
    preds = logits.argmax(dim=-1).cpu().numpy()
    y_true = y_true.cpu().numpy()
    f1s = []
    for c in range(num_classes):
        tp = np.sum((preds == c) & (y_true == c))
        fp = np.sum((preds == c) & (y_true != c))
        fn = np.sum((preds != c) & (y_true == c))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        f1s.append(f1)
    return float(np.mean(f1s))


def expected_calibration_error(logits: torch.Tensor, y_true: torch.Tensor,
                                n_bins: int = 10) -> float:
    """
    ECE as used in Figure 1 of the paper: bins predictions by confidence,
    compares average confidence to actual accuracy within each bin.
    """
    probs = F.softmax(logits, dim=-1)
    confidences, preds = probs.max(dim=-1)
    accuracies = (preds == y_true).float()

    confidences = confidences.cpu().numpy()
    accuracies = accuracies.cpu().numpy()

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(confidences)
    for i in range(n_bins):
        lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
        in_bin = (confidences > lo) & (confidences <= hi)
        prop_in_bin = in_bin.mean()
        if prop_in_bin > 0:
            acc_in_bin = accuracies[in_bin].mean()
            conf_in_bin = confidences[in_bin].mean()
            ece += np.abs(acc_in_bin - conf_in_bin) * prop_in_bin
    return float(ece * 100)  # reported as a percentage-like scale in the paper
