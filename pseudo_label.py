"""
Uncertainty-aware pseudo-labeling (paper Section 2.3, Eq. 4-5, 9-10).
"""

import torch
import torch.nn.functional as F


def predict_probs(logits: torch.Tensor) -> torch.Tensor:
    """Eq. 4: soft class prediction y_hat = softmax(h_phi(Z))."""
    return F.softmax(logits, dim=-1)


def prediction_entropy(probs: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Eq. 5: entropy of the predicted distribution, per sample."""
    return -(probs * torch.log(probs + eps)).sum(dim=-1)


def select_confident_pseudo_labels(logits_t: torch.Tensor, rho: float = 0.5):
    """
    Eq. 9: filtered index set I = {j | U_t^j < rho}.

    Args:
        logits_t: (Nt, K) raw target logits.
        rho: entropy threshold.
    Returns:
        mask: (Nt,) boolean tensor, True where sample is confident.
        pseudo_labels: (Nt,) hard argmax pseudo-labels (only meaningful where mask=True).
        entropy: (Nt,) per-sample entropy values (for logging/analysis).
    """
    probs = predict_probs(logits_t)
    entropy = prediction_entropy(probs)
    # Normalize entropy to [0, 1] by max possible entropy log(K) so that
    # the threshold rho behaves consistently across different K.
    max_entropy = torch.log(torch.tensor(float(logits_t.shape[-1])))
    norm_entropy = entropy / max_entropy
    mask = norm_entropy < rho
    pseudo_labels = probs.argmax(dim=-1)
    return mask, pseudo_labels, norm_entropy


def pseudo_label_loss(logits_t: torch.Tensor, rho: float = 0.5) -> torch.Tensor:
    """
    Eq. 10: L_PL = (1/|I|) * sum_{j in I} CE(h_phi(Z_t^j), y_hat_t^j).

    Returns 0 (as a tensor, so it stays differentiable-compatible) if no
    sample passes the confidence filter, which naturally happens early in
    training before the model calibrates.
    """
    mask, pseudo_labels, _ = select_confident_pseudo_labels(logits_t, rho=rho)
    if mask.sum() == 0:
        return torch.tensor(0.0, device=logits_t.device, requires_grad=True)
    loss = F.cross_entropy(logits_t[mask], pseudo_labels[mask])
    return loss
