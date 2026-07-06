"""
Runs Causal-OT across a fixed list of source->target domain pairs and
prints/saves a results table in the same shape as Table 2 (WISDM) / Table 3
(UCIHAR) of the paper.

This reproduces the "Ours" row only. The other rows in the paper's tables
(TransPL, MMDA, CoDATS, SASA, RAINCOAT, SoftMax, NCP, SP, ATT, SHOT, T2PL)
are separate published baselines with their own codebases; this repo does
not reimplement them. Two honest options if you need the full comparison
table:
  1. Report your "Ours" numbers from this script alongside the paper's
     published baseline numbers (acceptable for most write-ups, as long as
     you're clear the baselines are copied from the paper, not re-run).
  2. Re-run the baselines yourself using the AdaTime benchmark repo
     (https://github.com/emadeldeen24/AdaTime), which includes several of
     these methods (CoDATS, SASA and others) under a shared evaluation
     harness -- that's also the harness the paper says it follows for
     domain-pair selection.

Usage (after converting data with convert_adatime_to_npy.py):
    python run_benchmark.py --dataset wisdm --data-root data/wisdm --epochs 100
    python run_benchmark.py --dataset ucihar --data-root data/ucihar --epochs 100

Expects, for each domain id D referenced in the pair list below:
    <data-root>/<D>/X.npy
    <data-root>/<D>/y.npy
"""

import argparse
import os
import json
import numpy as np

from config import CausalOTConfig
from data import load_domain_npy
from train import train, evaluate, get_device

# Exact source->target domain pairs from the paper's Table 2 / Table 3.
DOMAIN_PAIRS = {
    "wisdm": [(7, 18), (20, 30), (35, 31), (17, 23), (6, 19),
              (2, 11), (33, 12), (5, 26), (28, 4), (23, 32)],
    "ucihar": [(2, 11), (6, 23), (7, 13), (9, 18), (12, 16),
               (18, 27), (20, 5), (24, 8), (28, 27), (30, 20)],
}


def load_domain(data_root: str, domain_id):
    d = os.path.join(data_root, str(domain_id))
    return load_domain_npy(os.path.join(d, "X.npy"), os.path.join(d, "y.npy"))


def run(dataset: str, data_root: str, epochs: int, seed: int = 0):
    pairs = DOMAIN_PAIRS[dataset]
    cfg = CausalOTConfig()
    cfg.num_epochs = epochs

    results = {}
    for src_id, tgt_id in pairs:
        pair_name = f"{src_id}\u2192{tgt_id}"
        print(f"\n===== {dataset.upper()} : {pair_name} =====")
        Xs, ys = load_domain(data_root, src_id)
        Xt, yt = load_domain(data_root, tgt_id)

        feature_extractor, classifier = train(cfg, Xs, ys, Xt, yt)
        device = get_device(cfg)
        acc, f1, ece = evaluate(feature_extractor, classifier, Xt, yt, cfg, device)
        results[pair_name] = {"accuracy": acc * 100, "f1": f1 * 100, "ece": ece}
        print(f"  -> {pair_name}: acc={acc*100:.2f}%  F1={f1*100:.2f}%  ECE={ece:.2f}")

    accs = [v["accuracy"] for v in results.values()]
    avg_acc = float(np.mean(accs))
    results["Average"] = {"accuracy": avg_acc}

    print("\n" + "=" * 60)
    print(f"{'Pair':<12}" + "".join(f"{k:>9}" for k in results if k != "Average"))
    print(f"{'Ours':<12}" + "".join(f"{v['accuracy']:>9.1f}" for k, v in results.items() if k != "Average"))
    print(f"\nAverage accuracy over {len(pairs)} pairs: {avg_acc:.2f}%")
    print("=" * 60)

    out_path = f"benchmark_results_{dataset}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved detailed results to {out_path}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["wisdm", "ucihar"], required=True)
    parser.add_argument("--data-root", required=True,
                         help="folder containing <domain_id>/X.npy, y.npy for each domain")
    parser.add_argument("--epochs", type=int, default=100)
    args = parser.parse_args()
    run(args.dataset, args.data_root, args.epochs)
