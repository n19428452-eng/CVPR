"""
Central configuration for Causal-OT.
All hyperparameter defaults are taken directly from the paper
(Section 3, 'Implementation' paragraph).
"""

from dataclasses import dataclass


@dataclass
class CausalOTConfig:
    # ---- data ----
    seq_len: int = 128          # T: time steps per sample (128 for HAR/WISDM/HHAR)
    num_channels: int = 9       # D: number of sensor channels (dataset-dependent)
    num_classes: int = 6        # K: number of shared classes
    batch_size: int = 64

    # ---- feature extractor ----
    feature_dim: int = 128      # z: latent embedding size
    cnn_channels: int = 64
    tcn_channels: int = 64
    lstm_hidden: int = 64

    # ---- causal graph (Granger) ----
    max_lag: int = 5            # candidate VAR lag orders to search over via BIC
    p_value_thresh: float = 0.05
    causal_embed_dim: int = 8   # k: causal descriptor dimension (Laplacian eigvecs)
    graph_blend_alpha: float = 0.8   # alpha in [0.6, 0.9], blends raw-signal graph with latent graph
    graph_reestimate_every: int = 10  # epochs between re-estimating graphs on Z
    warmup_epochs: int = 5      # W_init: epochs before first re-estimation on Z

    # ---- OT ----
    ot_lambda: float = 1.0      # weight of causal term in the OT cost matrix (Eq. 2)
    sinkhorn_eps: float = 0.01  # entropic regularization epsilon (Sinkhorn OT regularization)
    sinkhorn_max_iter: int = 200

    # ---- pseudo-labeling ----
    entropy_threshold: float = 0.5  # rho: entropy threshold for confident pseudo-labels

    # ---- loss weights ----
    alpha: float = 1.0   # weight on L_OT
    beta: float = 1.0    # weight on L_PL

    # ---- optimization ----
    lr: float = 1e-3
    weight_decay: float = 1e-4
    num_epochs: int = 100

    device: str = "cuda"  # falls back to cpu automatically if unavailable
