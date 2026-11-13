"""Export serving metadata for the trained DeepFM ranker.

The original training run saved only model weights. Online inference also
needs the exact bucket boundaries and feature offsets, plus a few per-user
aggregates that are not present in ``data/seqs.csv`` (notably user average
rating and a typical inter-event gap).

This script rebuilds those artifacts from the same ``samples.csv`` used for
training. It is intentionally streaming: the file has 4.27M rows and the host
does not have enough spare RAM for a full pandas materialisation.

Outputs (under artifacts/ranker/):
  preprocess.json     feature order, quantile edges, offsets, model config
  user_features.csv   user_id,user_n,user_avg,gap_days

Usage:
  .venv\\Scripts\\python.exe scripts\\export_ranker_meta.py
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

SPARSE = ["user_id", "movie_id", "year"]
CONT = [
    "user_n",
    "user_avg",
    "item_n",
    "item_avg",
    "days_since_prev",
    "genre_affinity",
    "genre_overlap",
]
FIELDS = SPARSE + CONT


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", default="data/samples.csv")
    ap.add_argument("--out", default="artifacts/ranker")
    ap.add_argument("--metrics", default="artifacts/ranker/deepfm_metrics.json")
    ap.add_argument("--weights", default="artifacts/ranker/deepfm.pt")
    ap.add_argument("--chunk-rows", type=int, default=250_000)
    ap.add_argument("--weights-only", action="store_true",
                    help="only refresh deepfm_weights.npz from the PyTorch checkpoint")
    return ap.parse_args()


def export_weights(args):
    """Convert the checkpoint to NPZ so serving does not need PyTorch."""
    import torch

    state = torch.load(args.weights, map_location="cpu", weights_only=True)
    arrays = {
        key: value.detach().cpu().numpy()
        for key, value in state.items()
        if value.is_floating_point()
    }
    out = os.path.join(args.out, "deepfm_weights.npz")
    np.savez_compressed(out, **arrays)
    print(
        json.dumps(
            {"weights": out, "tensors": len(arrays),
             "params": sum(v.size for v in arrays.values())},
            ensure_ascii=False,
        ),
        flush=True,
    )


def update_max(current, values):
    if len(values) == 0:
        return current
    return max(current, int(np.nanmax(values)))


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)
    if args.weights_only:
        export_weights(args)
        return

    usecols = ["user_id", "movie_id", "year", *CONT, "split"]
    # Keep only training values, matching deepfm.load_data exactly.
    values = {c: [] for c in CONT}
    profile_parts = []
    max_user = max_movie = max_year = -1
    train_rows = 0
    total_rows = 0

    for n, chunk in enumerate(
        pd.read_csv(args.samples, usecols=usecols, chunksize=args.chunk_rows), 1
    ):
        total_rows += len(chunk)
        max_user = update_max(max_user, chunk.user_id.values)
        max_movie = update_max(max_movie, chunk.movie_id.values)
        max_year = update_max(max_year, chunk.year.fillna(0).values)

        train = chunk[chunk["split"] == "train"]
        train_rows += len(train)
        for col in CONT:
            values[col].append(train[col].fillna(0).to_numpy(dtype=np.float64))

        base = chunk.groupby("user_id", sort=False)[["user_n", "user_avg"]].first()
        gap = (
            train.assign(_gap=train["days_since_prev"].fillna(0))
            .groupby("user_id", sort=False)["_gap"]
            .agg(["sum", "count"])
        )
        profile_parts.append(base.join(gap, how="outer"))
        print(
            f"[ranker-meta] chunk {n}: total={total_rows:,} train={train_rows:,}",
            flush=True,
        )

    edges = {}
    for col in CONT:
        train_values = np.concatenate(values[col])
        edges[col] = np.unique(
            np.quantile(train_values, np.linspace(0, 1, 21)[1:-1])
        ).astype(np.float64).tolist()
        print(f"[ranker-meta] {col}: edges={len(edges[col])}", flush=True)

    profiles = pd.concat(profile_parts)
    profiles = profiles.groupby(level="user_id", sort=False).agg(
        user_n=("user_n", "first"),
        user_avg=("user_avg", "first"),
        gap_sum=("sum", "sum"),
        gap_count=("count", "sum"),
    )
    profiles["gap_days"] = (
        profiles["gap_sum"] / profiles["gap_count"].replace(0, np.nan)
    ).fillna(0.0)
    profiles = profiles.reset_index()[
        ["user_id", "user_n", "user_avg", "gap_days"]
    ]
    profiles.to_csv(os.path.join(args.out, "user_features.csv"), index=False)

    with open(args.metrics, encoding="utf-8") as fh:
        metrics = json.load(fh)
    model_args = metrics.get("args", {})
    embed_dim = int(model_args.get("embed_dim", 16))
    hidden = list(model_args.get("hidden", [256, 128]))

    vocab = {
        "user_id": max_user + 1,
        "movie_id": max_movie + 1,
        "year": max_year + 2,
    }
    offsets, sizes = {}, {}
    off = 0
    for field in SPARSE:
        offsets[field], sizes[field] = off, vocab[field]
        off += vocab[field]
    for field in CONT:
        n_bins = len(edges[field]) + 1
        offsets[field], sizes[field] = off, n_bins
        off += n_bins

    metadata = {
        "version": 1,
        "feature_order": FIELDS,
        "sparse_fields": SPARSE,
        "continuous_fields": CONT,
        "edges": edges,
        "offsets": offsets,
        "field_sizes": sizes,
        "total_dim": off,
        "n_fields": len(FIELDS),
        "embed_dim": embed_dim,
        "hidden": hidden,
        "train_rows": train_rows,
        "source_rows": total_rows,
        "user_feature_rows": int(len(profiles)),
        "notes": {
            "days_since_prev": "per-user mean training gap used at serving time",
            "genre_affinity": "estimated from historical item averages in serving adapter",
        },
    }
    with open(os.path.join(args.out, "preprocess.json"), "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, ensure_ascii=False, indent=2)
    export_weights(args)

    print(
        json.dumps(
            {
                "out": args.out,
                "train_rows": train_rows,
                "users": len(profiles),
                "total_dim": off,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
