"""
Causal-dependent Optimal Transport (paper Section 2.2, Eq. 2-3, 8).

Cost matrix:  C_ij = ||f_s(x_i^s) - f_t(x_j^t)||^2 + lambda * ||phi_i^s - phi_j^t||^2
OT plan:      gamma* = argmin_gamma <gamma, C> + eps * H(gamma)   (entropic OT, Sinkhorn)
OT loss:      L_OT = <gamma*, C>

phi_i^s / phi_j^t are per-SAMPLE causal descriptors. Since Granger graphs are
estimated per-CHANNEL (not per-sample), we derive a per-sample causal
descriptor by projecting each sample's channel-wise signal energy onto the
channel causal embedding Phi(G) (shape D x k), giving a k-dim vector per
sample. This keeps Eq. 2 well-defined at the sample level while still being
driven entirely by the Granger-causal structure.
"""

import numpy as np
import torch
import ot  # POT: Python Optimal Transport


def sample_causal_descriptor(x: torch.Tensor, phi_graph: torch.Tensor) -> torch.Tensor:
    """
    Derive a per-sample causal descriptor from raw/latent channel signals
    and the channel-level causal embedding Phi(G).

    Args:
        x: (B, T, D) batch of (raw or latent-projected) sequences.
        phi_graph: (D, k) channel causal embedding from laplacian_embedding.
    Returns:
        phi_sample: (B, k) per-sample causal descriptor.
    """
    # Use per-channel energy (mean abs activation over time) as the channel
    # "activity profile" for this sample, then project through Phi(G).
    channel_energy = x.abs().mean(dim=1)          # (B, D)
    phi_sample = channel_energy @ phi_graph        # (B, k)
    return phi_sample


def causal_cost_matrix(z_s: torch.Tensor, z_t: torch.Tensor,
                        phi_s: torch.Tensor, phi_t: torch.Tensor,
                        lam: float = 1.0) -> torch.Tensor:
    """
    Eq. 2: pairwise cost combining feature distance and causal descriptor
    consistency.

    Args:
        z_s: (Ns, z) source latent features.
        z_t: (Nt, z) target latent features.
        phi_s: (Ns, k) source causal descriptors.
        phi_t: (Nt, k) target causal descriptors.
        lam: weight on the causal term.
    Returns:
        C: (Ns, Nt) cost matrix.
    """
    feat_dist = torch.cdist(z_s, z_t, p=2) ** 2          # (Ns, Nt)
    causal_dist = torch.cdist(phi_s, phi_t, p=2) ** 2    # (Ns, Nt)
    C = feat_dist + lam * causal_dist
    return C


def sinkhorn_ot(C: torch.Tensor, eps: float = 0.01, max_iter: int = 200):
    """
    Eq. 3: entropy-regularized OT solved via Sinkhorn iterations (POT library).

    Args:
        C: (Ns, Nt) cost matrix (torch tensor, may require grad).
        eps: entropic regularization strength.
        max_iter: max Sinkhorn iterations.
    Returns:
        gamma: (Ns, Nt) torch tensor transport plan (same device/dtype as C,
               detached from the OT solver's internal numpy computation but
               reattached to the autograd graph via C so gradients flow
               through <gamma, C> as in Eq. 8; POT's gradient w.r.t. gamma
               itself is not backpropagated, consistent with treating gamma
               as fixed within an inner loop -- a standard practical choice
               for OT-based domain adaptation).
    """
    device = C.device
    dtype = C.dtype
    Ns, Nt = C.shape

    C_np = C.detach().cpu().numpy().astype(np.float64)
    a = np.ones(Ns, dtype=np.float64) / Ns
    b = np.ones(Nt, dtype=np.float64) / Nt

    gamma_np = ot.sinkhorn(a, b, C_np, reg=eps, numItermax=max_iter)
    gamma = torch.tensor(gamma_np, device=device, dtype=dtype)
    return gamma


def causal_ot_loss(z_s: torch.Tensor, z_t: torch.Tensor,
                    phi_s: torch.Tensor, phi_t: torch.Tensor,
                    lam: float = 1.0, eps: float = 0.01,
                    max_iter: int = 200) -> torch.Tensor:
    """
    Full Causal-OT loss (Eq. 8): L_OT = <gamma*, C>.

    gamma* is computed by Sinkhorn on the detached cost (standard practice --
    the OT plan is treated as constant within a training step), while C is
    kept attached to the autograd graph so gradients w.r.t. z_s, z_t, phi_s,
    phi_t (and thus theta) flow through the inner product <gamma*, C>.
    """
    C = causal_cost_matrix(z_s, z_t, phi_s, phi_t, lam=lam)
    gamma = sinkhorn_ot(C, eps=eps, max_iter=max_iter)
    loss = (gamma * C).sum()
    return loss
