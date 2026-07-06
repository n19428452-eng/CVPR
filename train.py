"""
Main training loop for Causal-OT (implements the pipeline in Figure 3 /
Section 2.4, Eq. 6).

Run with synthetic data (no download needed) to sanity-check everything:
    python train.py --synthetic

Run with real data (after preparing .npy files, see README):
    python train.py --source-x path/to/source_X.npy --source-y path/to/source_y.npy \
                     --target-x path/to/target_X.npy --target-y path/to/target_y.npy
    (target_y is optional -- only used for reporting accuracy, never for training)
"""

import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

from config import CausalOTConfig
from data import TimeSeriesDomainDataset, make_synthetic_domains, load_domain_npy
from models import FeatureExtractor, Classifier
from causal_graph import (
    compute_granger_causal_graph,
    laplacian_embedding,
    align_causal_embeddings,
    blend_graphs,
    fast_correlation_proxy_graph,
)
from ot_module import sample_causal_descriptor, causal_ot_loss
from pseudo_label import pseudo_label_loss
from utils import accuracy, macro_f1, expected_calibration_error


def get_device(cfg: CausalOTConfig) -> torch.device:
    if cfg.device == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_causal_descriptors(X_s: np.ndarray, X_t: np.ndarray, cfg: CausalOTConfig):
    """
    Runs Section 2.2: computes Granger causal graphs for both domains,
    their Laplacian embeddings, and aligns them via Procrustes so that
    phi^s and phi^t live in a comparable basis (see causal_graph.py docstring).
    """
    print("[Causal-OT] Estimating Granger causal graph for SOURCE domain...")
    W_s = compute_granger_causal_graph(X_s, max_lag=cfg.max_lag,
                                        p_value_thresh=cfg.p_value_thresh)
    print("[Causal-OT] Estimating Granger causal graph for TARGET domain...")
    W_t = compute_granger_causal_graph(X_t, max_lag=cfg.max_lag,
                                        p_value_thresh=cfg.p_value_thresh)

    phi_s = laplacian_embedding(W_s, k=cfg.causal_embed_dim)
    phi_t = laplacian_embedding(W_t, k=cfg.causal_embed_dim)
    phi_t = align_causal_embeddings(phi_s, phi_t)

    return W_s, W_t, phi_s, phi_t


def train(cfg: CausalOTConfig,
          Xs: np.ndarray, ys: np.ndarray,
          Xt: np.ndarray, yt_eval: np.ndarray = None):
    """
    Xs, ys: labeled source domain (N_s, T, D), (N_s,)
    Xt: unlabeled target domain (N_t, T, D)
    yt_eval: optional target labels, used ONLY for evaluation/logging, never
             seen by the training loop.
    """
    device = get_device(cfg)
    print(f"[Causal-OT] Using device: {device}")

    cfg.num_channels = Xs.shape[2]
    cfg.seq_len = Xs.shape[1]
    cfg.num_classes = int(max(ys.max(), (yt_eval.max() if yt_eval is not None else 0)) + 1)

    # ---- Step 1: Granger causal graph construction (Section 2.2) ----
    W_s, W_t, phi_s_np, phi_t_np = build_causal_descriptors(Xs, Xt, cfg)
    phi_graph_s = torch.tensor(phi_s_np, dtype=torch.float32, device=device)
    phi_graph_t = torch.tensor(phi_t_np, dtype=torch.float32, device=device)

    # ---- Models ----
    feature_extractor = FeatureExtractor(
        num_channels=cfg.num_channels, cnn_channels=cfg.cnn_channels,
        tcn_channels=cfg.tcn_channels, lstm_hidden=cfg.lstm_hidden,
        feature_dim=cfg.feature_dim,
    ).to(device)
    classifier = Classifier(cfg.feature_dim, cfg.num_classes).to(device)

    optimizer = torch.optim.Adam(
        list(feature_extractor.parameters()) + list(classifier.parameters()),
        lr=cfg.lr, weight_decay=cfg.weight_decay,
    )

    # ---- Data loaders ----
    source_ds = TimeSeriesDomainDataset(Xs, ys)
    target_ds = TimeSeriesDomainDataset(Xt, yt_eval)  # yt only used for eval below
    source_loader = DataLoader(source_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True)
    target_loader = DataLoader(target_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True)

    # ---- Training loop (Section 2.4, Eq. 6: L_total = L_src + a*L_OT + b*L_PL) ----
    for epoch in range(cfg.num_epochs):
        feature_extractor.train()
        classifier.train()

        # Warmup: source-only training for the first `warmup_epochs`
        is_warmup = epoch < cfg.warmup_epochs

        # Periodic latent graph re-estimation + blending (Eq. 1), only after warmup
        if (not is_warmup) and (epoch - cfg.warmup_epochs) % cfg.graph_reestimate_every == 0:
            with torch.no_grad():
                feature_extractor.eval()
                Xs_t = torch.tensor(Xs, dtype=torch.float32, device=device)
                Xt_t = torch.tensor(Xt, dtype=torch.float32, device=device)
                z_channel_s = feature_extractor.cnn(Xs_t.transpose(1, 2)).transpose(1, 2).cpu().numpy()
                z_channel_t = feature_extractor.cnn(Xt_t.transpose(1, 2)).transpose(1, 2).cpu().numpy()
                A_z_s = fast_correlation_proxy_graph(z_channel_s)
                A_z_t = fast_correlation_proxy_graph(z_channel_t)
                feature_extractor.train()

            # A_z has cnn_channels dims which may differ from D; only blend
            # when dims match, otherwise skip (documented simplification).
            if A_z_s.shape == W_s.shape:
                W_s_blended = blend_graphs(W_s, A_z_s, alpha=cfg.graph_blend_alpha)
                W_t_blended = blend_graphs(W_t, A_z_t, alpha=cfg.graph_blend_alpha)
                phi_s_np = laplacian_embedding(W_s_blended, k=cfg.causal_embed_dim)
                phi_t_np = laplacian_embedding(W_t_blended, k=cfg.causal_embed_dim)
                phi_t_np = align_causal_embeddings(phi_s_np, phi_t_np)
                phi_graph_s = torch.tensor(phi_s_np, dtype=torch.float32, device=device)
                phi_graph_t = torch.tensor(phi_t_np, dtype=torch.float32, device=device)

        epoch_losses = {"src": 0.0, "ot": 0.0, "pl": 0.0, "total": 0.0}
        n_batches = min(len(source_loader), len(target_loader))
        target_iter = iter(target_loader)

        for batch_idx, (x_s, y_s) in enumerate(source_loader):
            if batch_idx >= n_batches:
                break
            try:
                target_batch = next(target_iter)
            except StopIteration:
                target_iter = iter(target_loader)
                target_batch = next(target_iter)

            if yt_eval is not None:
                x_t, _ = target_batch
            else:
                x_t = target_batch

            x_s, y_s, x_t = x_s.to(device), y_s.to(device), x_t.to(device)

            z_s = feature_extractor(x_s)
            logits_s = classifier(z_s)
            loss_src = torch.nn.functional.cross_entropy(logits_s, y_s)

            if is_warmup:
                loss_total = loss_src
                loss_ot = torch.tensor(0.0)
                loss_pl = torch.tensor(0.0)
            else:
                z_t = feature_extractor(x_t)
                logits_t = classifier(z_t)

                phi_sample_s = sample_causal_descriptor(x_s, phi_graph_s)
                phi_sample_t = sample_causal_descriptor(x_t, phi_graph_t)

                loss_ot = causal_ot_loss(
                    z_s, z_t, phi_sample_s, phi_sample_t,
                    lam=cfg.ot_lambda, eps=cfg.sinkhorn_eps,
                    max_iter=cfg.sinkhorn_max_iter,
                )
                loss_pl = pseudo_label_loss(logits_t, rho=cfg.entropy_threshold)

                loss_total = loss_src + cfg.alpha * loss_ot + cfg.beta * loss_pl

            optimizer.zero_grad()
            loss_total.backward()
            optimizer.step()

            epoch_losses["src"] += loss_src.item()
            epoch_losses["ot"] += float(loss_ot)
            epoch_losses["pl"] += float(loss_pl)
            epoch_losses["total"] += loss_total.item()

        for k in epoch_losses:
            epoch_losses[k] /= max(n_batches, 1)

        log_line = (f"Epoch {epoch+1:3d}/{cfg.num_epochs} "
                    f"| L_src {epoch_losses['src']:.4f} "
                    f"| L_OT {epoch_losses['ot']:.4f} "
                    f"| L_PL {epoch_losses['pl']:.4f} "
                    f"| L_total {epoch_losses['total']:.4f}")

        if yt_eval is not None and (epoch + 1) % 5 == 0:
            acc, f1, ece = evaluate(feature_extractor, classifier, Xt, yt_eval, cfg, device)
            log_line += f" || target acc {acc*100:.2f}% | F1 {f1*100:.2f}% | ECE {ece:.2f}"
        print(log_line)

    return feature_extractor, classifier


@torch.no_grad()
def evaluate(feature_extractor, classifier, X, y, cfg: CausalOTConfig, device):
    feature_extractor.eval()
    classifier.eval()
    X_t = torch.tensor(X, dtype=torch.float32, device=device)
    y_t = torch.tensor(y, dtype=torch.long, device=device)
    z = feature_extractor(X_t)
    logits = classifier(z)
    acc = accuracy(logits, y_t)
    f1 = macro_f1(logits, y_t, cfg.num_classes)
    ece = expected_calibration_error(logits, y_t)
    return acc, f1, ece


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic", action="store_true",
                         help="Use built-in synthetic domain-shift data (no download needed)")
    parser.add_argument("--source-x", type=str, default=None)
    parser.add_argument("--source-y", type=str, default=None)
    parser.add_argument("--target-x", type=str, default=None)
    parser.add_argument("--target-y", type=str, default=None, help="optional, eval only")
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()

    cfg = CausalOTConfig()
    if args.epochs is not None:
        cfg.num_epochs = args.epochs

    if args.synthetic or args.source_x is None:
        print("[Causal-OT] No dataset paths given -- using synthetic domain-shift data.")
        (Xs, ys), (Xt, yt) = make_synthetic_domains(
            n_samples=300, seq_len=cfg.seq_len, num_channels=cfg.num_channels,
            num_classes=cfg.num_classes,
        )
    else:
        Xs, ys = load_domain_npy(args.source_x, args.source_y)
        Xt, _ = load_domain_npy(args.target_x, args.source_y)  # placeholder, y unused for target
        yt = np.load(args.target_y) if args.target_y else None

    train(cfg, Xs, ys, Xt, yt)


if __name__ == "__main__":
    main()
