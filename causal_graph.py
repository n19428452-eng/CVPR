"""
Granger-causal graph extraction (paper Section 2.2).

Given a multivariate time-series batch X in R^{N x T x D}, this module:
  1. Tests each channel for stationarity (Augmented Dickey-Fuller test).
  2. Selects the VAR lag order via BIC.
  3. Fits pairwise Granger-causality tests, retains edges with p < 0.05.
  4. Row-normalizes the resulting adjacency matrix W.
  5. Computes the graph Laplacian L = D - W and takes the first k
     nontrivial eigenvectors as per-channel causal descriptors phi_i.

Because per-batch statistical Granger tests are expensive, in practice you
call `compute_granger_causal_graph` once on a large pooled sample per domain
(not every mini-batch). The training loop re-estimates it periodically
(see config.graph_reestimate_every) and blends it with the previous graph
via Eq. 1: A^(t) = alpha * A_X + (1 - alpha) * A_Z^(t).
"""

import numpy as np
import warnings
from statsmodels.tsa.stattools import adfuller, grangercausalitytests

warnings.filterwarnings("ignore")  # statsmodels is noisy about small samples


def _is_stationary(series: np.ndarray, significance: float = 0.05) -> bool:
    """Augmented Dickey-Fuller test. Returns True if series is stationary."""
    try:
        result = adfuller(series, autolag="AIC")
        return result[1] < significance
    except Exception:
        # Degenerate (constant) series etc. -> treat as stationary to avoid crashing
        return True


def _difference_until_stationary(series: np.ndarray, max_diff: int = 2) -> np.ndarray:
    """Differencing fallback for non-stationary channels."""
    s = series.copy()
    for _ in range(max_diff):
        if _is_stationary(s):
            break
        s = np.diff(s, prepend=s[0])
    return s


def _select_lag_by_bic(data_2d: np.ndarray, max_lag: int) -> int:
    """
    Select VAR lag order via BIC using statsmodels VAR.
    data_2d: (T, 2) array for a pair of channels.
    """
    from statsmodels.tsa.api import VAR
    try:
        model = VAR(data_2d)
        best_lag, best_bic = 1, np.inf
        max_allowed = min(max_lag, data_2d.shape[0] // 3 - 1)
        max_allowed = max(1, max_allowed)
        for lag in range(1, max_allowed + 1):
            try:
                res = model.fit(lag)
                if res.bic < best_bic:
                    best_bic = res.bic
                    best_lag = lag
            except Exception:
                continue
        return best_lag
    except Exception:
        return 1


def compute_granger_causal_graph(X: np.ndarray, max_lag: int = 5,
                                  p_value_thresh: float = 0.05) -> np.ndarray:
    """
    Build a directed Granger-causal adjacency matrix for a pooled set of
    multivariate time-series samples.

    Args:
        X: array of shape (N, T, D) -- N samples, T time steps, D channels.
           Samples are concatenated along time to form long per-channel
           signals, which is a standard practical approximation for
           estimating a single domain-level causal graph.
        max_lag: maximum VAR lag order to search over via BIC.
        p_value_thresh: edges kept only if Granger test p-value < this.

    Returns:
        W: (D, D) row-normalized adjacency matrix. W[i, j] = influence
           score of channel i -> channel j (i Granger-causes j).
    """
    N, T, D = X.shape
    # Concatenate samples along time to get one long signal per channel.
    # (This treats samples as contiguous segments; fine for i.i.d. windows
    # drawn from the same regime, which is the common HAR/WISDM/HHAR setup.)
    signals = X.transpose(1, 0, 2).reshape(N * T, D)  # (N*T, D)

    # Step 1: stationarity check + differencing fallback per channel
    stationary_signals = np.zeros_like(signals)
    for d in range(D):
        stationary_signals[:, d] = _difference_until_stationary(signals[:, d])

    W = np.zeros((D, D), dtype=np.float64)

    for i in range(D):
        for j in range(D):
            if i == j:
                continue
            pair = stationary_signals[:, [j, i]]  # statsmodels convention: [target, cause]
            # subsample for tractability if very long
            if pair.shape[0] > 2000:
                idx = np.linspace(0, pair.shape[0] - 1, 2000).astype(int)
                pair = pair[idx]

            lag = _select_lag_by_bic(pair, max_lag)
            try:
                test_result = grangercausalitytests(pair, maxlag=lag, verbose=False)
                p_value = test_result[lag][0]["ssr_ftest"][1]
                if p_value < p_value_thresh:
                    f_stat = test_result[lag][0]["ssr_ftest"][0]
                    W[i, j] = f_stat
            except Exception:
                continue

    # Row-normalize so each row sums to 1 (guard against all-zero rows)
    row_sums = W.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    W = W / row_sums
    return W


def laplacian_embedding(W: np.ndarray, k: int = 8) -> np.ndarray:
    """
    Compute causal descriptors phi_i via Laplacian eigenmap of the graph.

    L = D - W (using symmetrized W so L is well-defined for eigendecomposition),
    then take the first k nontrivial eigenvectors as the embedding Phi(G) in R^{d x k}.

    Args:
        W: (D, D) adjacency matrix (possibly asymmetric/directed).
        k: embedding dimension.

    Returns:
        Phi: (D, k) array, one causal descriptor per channel.
    """
    D_dim = W.shape[0]
    W_sym = (W + W.T) / 2.0
    degree = np.diag(W_sym.sum(axis=1))
    L = degree - W_sym

    eigvals, eigvecs = np.linalg.eigh(L)
    order = np.argsort(eigvals)
    eigvecs = eigvecs[:, order]

    k_eff = min(k, D_dim - 1)
    k_eff = max(k_eff, 1)
    # Skip the trivial (near-zero eigenvalue, constant) eigenvector at index 0
    Phi = eigvecs[:, 1:1 + k_eff]

    if Phi.shape[1] < k:
        pad = np.zeros((D_dim, k - Phi.shape[1]))
        Phi = np.concatenate([Phi, pad], axis=1)

    return Phi


def blend_graphs(A_raw: np.ndarray, A_latent: np.ndarray, alpha: float = 0.8) -> np.ndarray:
    """Eq. 1: A^(t) = alpha * A_X + (1 - alpha) * A_Z^(t)."""
    return alpha * A_raw + (1 - alpha) * A_latent


def fast_correlation_proxy_graph(Z_channelwise: np.ndarray) -> np.ndarray:
    """
    Fast proxy for a 'latent causal graph' A_Z, used during periodic
    re-estimation instead of a full statistical Granger-causality test
    (which would be too slow to call every few epochs on live features).
    Computes a lagged cross-correlation matrix (lag=1), row-normalized the
    same way as the Granger graph.

    Args:
        Z_channelwise: (N, T, C) pooled latent activations with a
                        channel-like dimension C.
    Returns:
        A_Z: (C, C) row-normalized proxy adjacency matrix.
    """
    N, T, C = Z_channelwise.shape
    flat = Z_channelwise.transpose(1, 0, 2).reshape(N * T, C)
    x_t = flat[:-1]
    x_t1 = flat[1:]
    x_t = (x_t - x_t.mean(axis=0)) / (x_t.std(axis=0) + 1e-8)
    x_t1 = (x_t1 - x_t1.mean(axis=0)) / (x_t1.std(axis=0) + 1e-8)
    A_Z = np.abs(x_t.T @ x_t1) / x_t.shape[0]
    np.fill_diagonal(A_Z, 0.0)
    row_sums = A_Z.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    A_Z = A_Z / row_sums
    return A_Z


def align_causal_embeddings(phi_source: np.ndarray, phi_target: np.ndarray) -> np.ndarray:
    """
    Orthogonal Procrustes alignment of the target causal embedding onto the
    source embedding's basis (Laplacian eigenvectors are only defined up to
    sign/rotation, so raw comparison across two independently-computed
    graphs is not meaningful without this step). Channel i in source and
    channel i in target are assumed to be the same physical sensor channel,
    which anchors the alignment.

    Args:
        phi_source: (D, k)
        phi_target: (D, k)
    Returns:
        phi_target_aligned: (D, k)
    """
    from scipy.linalg import orthogonal_procrustes
    R, _ = orthogonal_procrustes(phi_target, phi_source)
    return phi_target @ R
