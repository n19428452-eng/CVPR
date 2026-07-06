"""
Converts AdaTime-benchmark domain files (https://github.com/emadeldeen24/AdaTime)
into the (X.npy, y.npy) pairs expected by `data.load_domain_npy`.

AdaTime stores each "domain" (e.g. one WISDM user) as a folder containing
train.pt / test.pt, each a dict:
    {"samples": Tensor of shape (N, C, T), "labels": Tensor of shape (N,)}

Usage:
    python convert_adatime_to_npy.py \
        --pt-dir /path/to/AdaTime/data/wisdm/7 \
        --out-dir data/wisdm/7 \
        --split train

This writes data/wisdm/7/X.npy and data/wisdm/7/y.npy, already transposed to
(N, T, C) so they plug directly into train.py / data.load_domain_npy.
"""

import argparse
import os
import numpy as np
import torch


def convert(pt_dir: str, out_dir: str, split: str = "train"):
    pt_path = os.path.join(pt_dir, f"{split}.pt")
    if not os.path.exists(pt_path):
        raise FileNotFoundError(
            f"Could not find {pt_path}. Check --pt-dir points at a folder "
            f"containing train.pt/test.pt (one AdaTime domain)."
        )

    data = torch.load(pt_path, map_location="cpu")
    samples = data["samples"]
    labels = data["labels"]

    X = samples.numpy().astype(np.float32)
    y = labels.numpy().astype(np.int64)

    # AdaTime stores (N, C, T); our pipeline expects (N, T, C)
    if X.ndim == 3 and X.shape[1] < X.shape[2]:
        X = X.transpose(0, 2, 1)

    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "X.npy"), X)
    np.save(os.path.join(out_dir, "y.npy"), y)
    print(f"Wrote {X.shape} samples -> {out_dir}/X.npy, {out_dir}/y.npy")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pt-dir", required=True, help="folder with train.pt/test.pt for one domain")
    parser.add_argument("--out-dir", required=True, help="where to write X.npy / y.npy")
    parser.add_argument("--split", default="train", choices=["train", "test"])
    args = parser.parse_args()
    convert(args.pt_dir, args.out_dir, args.split)
