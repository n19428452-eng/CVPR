"""
Data utilities.

Two ways to get data:

1. `make_synthetic_domains(...)` -- generates synthetic multivariate
   time-series with a known causal structure and a controllable domain
   shift, purely in numpy. Use this to verify the whole pipeline runs
   end-to-end without downloading anything.

2. `load_ucihar(...)` / `load_wisdm(...)` -- thin loaders for the real
   benchmarks used in the paper. They expect data already downloaded
   locally (see README for download links + expected folder layout) and
   return numpy arrays in the same (N, T, D) / (N,) format as the synthetic
   generator, so the rest of the pipeline is agnostic to the source.
"""

import numpy as np
import torch
from torch.utils.data import Dataset


class TimeSeriesDomainDataset(Dataset):
    """Simple wrapper: X (N, T, D) float, y (N,) int labels (or None for target)."""

    def __init__(self, X: np.ndarray, y: np.ndarray = None):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long) if y is not None else None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx], self.y[idx]
        return self.X[idx]


def make_synthetic_domains(n_samples: int = 500, seq_len: int = 128,
                            num_channels: int = 6, num_classes: int = 4,
                            domain_shift: float = 1.5, seed: int = 0):
    """
    Generate a source and target domain with:
      - A shared causal structure: channel 0 drives channels 1 and 2 with a
        lag (so Granger causality should detect 0 -> 1, 0 -> 2), remaining
        channels are noise.
      - Class-discriminative signal: sinusoid frequency depends on the label.
      - A domain shift: target domain has scaled amplitude + added offset,
        simulating sensor/device shift while preserving the causal skeleton
        (this is the scenario the paper targets: causal mechanisms are
        domain-invariant even though marginal feature distributions shift).

    Returns:
        (Xs, ys), (Xt, yt) -- source (labeled) and target (labels only
        returned for evaluation purposes -- do NOT use yt during training).
    """
    rng = np.random.RandomState(seed)

    def _gen_domain(n, amp_scale, offset, noise_scale):
        X = np.zeros((n, seq_len, num_channels))
        y = rng.randint(0, num_classes, size=n)
        t = np.linspace(0, 4 * np.pi, seq_len)
        for i in range(n):
            freq = 1.0 + y[i] * 0.5  # class controls frequency
            base = np.sin(freq * t) * amp_scale + offset
            X[i, :, 0] = base + rng.randn(seq_len) * noise_scale
            # channel 1, 2 are lagged/noisy functions of channel 0 (causal)
            lag = 3
            driven = np.roll(base, lag)
            driven[:lag] = 0
            X[i, :, 1] = 0.8 * driven + rng.randn(seq_len) * noise_scale
            X[i, :, 2] = 0.6 * driven + rng.randn(seq_len) * noise_scale
            # remaining channels: independent noise (non-causal / spurious)
            for c in range(3, num_channels):
                X[i, :, c] = rng.randn(seq_len) * noise_scale * 1.5
        return X, y

    Xs, ys = _gen_domain(n_samples, amp_scale=1.0, offset=0.0, noise_scale=0.2)
    Xt, yt = _gen_domain(n_samples, amp_scale=1.0 * domain_shift,
                          offset=0.5, noise_scale=0.3)
    return (Xs, ys), (Xt, yt)


# ---------------------------------------------------------------------------
# Real dataset loaders. These expect preprocessed .npy / .pt files following
# the AdaTime benchmark convention (https://github.com/emadeldeen24/AdaTime),
# which is what the paper uses for UCIHAR / WISDM / HHAR. See README.md
# "Using real datasets" section for exact download + preprocessing steps.
# ---------------------------------------------------------------------------

def load_domain_npy(feature_path: str, label_path: str):
    """
    Generic loader for a single domain stored as .npy files.

    Expects:
        feature_path -> array of shape (N, T, D) or (N, D, T) (auto-detected
                         and transposed to (N, T, D) if needed)
        label_path   -> array of shape (N,)
    """
    X = np.load(feature_path)
    y = np.load(label_path)
    if X.ndim != 3:
        raise ValueError(f"Expected 3D array (N,T,D), got shape {X.shape}")
    # Heuristic: if channel dim looks smaller than time dim in axis 1 vs 2,
    # assume (N, D, T) and transpose to (N, T, D).
    if X.shape[1] < X.shape[2] and X.shape[1] <= 32:
        X = X.transpose(0, 2, 1)
    return X, y
