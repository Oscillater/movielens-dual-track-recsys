"""DeepFM ranking model with LR and FM baselines, trained on CPU (host).

All three models share one feature pipeline for a fair comparison:
  sparse fields   user_id, movie_id, year
  continuous      user_n, user_avg, item_n, item_avg, days_since_prev,
                  genre_affinity, genre_overlap  (quantile-bucketed per field)
  LR    first-order linear terms only
  FM    field embeddings + pairwise interactions (second order)
  DeepFM  FM wide part + MLP deep part

Label: rating >= 4. Metrics: AUC / LogLoss on val and test splits.

Usage:
  python deepfm.py --samples data/samples.csv --out artifacts/deepfm --model deepfm
"""
import argparse
import json
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

CONT = ["user_n", "user_avg", "item_n", "item_avg", "days_since_prev", "genre_affinity", "genre_overlap"]
SPARSE = ["user_id", "movie_id", "year"]


def bucketize(values, edges):
    return np.searchsorted(edges, values, side="right")


def load_data(path):
    df = pd.read_csv(path)
    train = df[df.split == "train"]
    val = df[df.split == "val"]
    test = df[df.split == "test"]
    # quantile edges per continuous field from train only
    edges = {c: np.unique(np.quantile(train[c].fillna(0).values, np.linspace(0, 1, 21)[1:-1])) for c in CONT}
    vocab = {"user_id": int(df.user_id.max()) + 1,
             "movie_id": int(df.movie_id.max()) + 1,
             "year": int(df.year.max()) + 2}
    offsets, sizes = {}, {}
    off = 0
    for f in SPARSE:
        offsets[f], sizes[f] = off, vocab[f]
        off += vocab[f]
    for c in CONT:
        n_bins = len(edges[c]) + 1
        offsets[c], sizes[c] = off, n_bins
        off += n_bins
    return train, val, test, edges, offsets, sizes, off


def frame_to_matrix(df, edges, offsets):
    cols = []
    for f in SPARSE:
        v = df[f].fillna(0).values.astype(np.int64)
        cols.append(v + offsets[f])
    for c in CONT:
        v = bucketize(df[c].fillna(0).values, edges[c]).astype(np.int64)
        cols.append(v + offsets[c])
    X = np.stack(cols, axis=1)
    y = df.label.values.astype(np.float32)
    return X, y


class Ranker(nn.Module):
    """LR / FM / DeepFM in one parameterization."""

    def __init__(self, n_fields, field_sizes, embed_dim, hidden, mode):
        super().__init__()
        self.mode = mode
        self.n_fields = n_fields
        total = sum(field_sizes)
        self.linear = nn.Embedding(total, 1)
        nn.init.zeros_(self.linear.weight)
        if mode in ("fm", "deepfm"):
            self.emb = nn.Embedding(total, embed_dim)
            nn.init.normal_(self.emb.weight, std=0.01)
        if mode == "deepfm":
            layers, d = [], n_fields * embed_dim
            for h in hidden:
                layers += [nn.Linear(d, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(0.2)]
                d = h
            layers.append(nn.Linear(d, 1))
            self.mlp = nn.Sequential(*layers)

    def forward(self, x):  # x: [B, F] int64
        first = self.linear(x).squeeze(-1).sum(-1)             # [B] first-order over fields
        if self.mode == "lr":
            return first
        v = self.emb(x)                                        # [B, F, E]
        sum_sq = v.sum(1) ** 2
        sq_sum = (v ** 2).sum(1)
        second = 0.5 * (sum_sq - sq_sum).sum(-1)               # [B]
        if self.mode == "fm":
            return first + second
        deep = self.mlp(v.flatten(1)).squeeze(-1)              # [B]
        return first + second + deep


def auc(y, p):
    order = np.argsort(p)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(p) + 1)
    pos = y.sum()
    return (ranks[y == 1].sum() - pos * (pos + 1) / 2) / (pos * (len(y) - pos))


def run_epoch(model, X, y, opt, args, device):
    model.train()
    total, n = 0.0, 0
    idx = np.random.permutation(len(y))
    for s in range(0, len(y), args.batch):
        b = idx[s:s + args.batch]
        xb = torch.from_numpy(X[b]).to(device)
        yb = torch.from_numpy(y[b]).to(device)
        logit = model(xb)
        loss = nn.functional.binary_cross_entropy_with_logits(logit, yb)
        opt.zero_grad()
        loss.backward()
        opt.step()
        total += loss.item() * len(b)
        n += len(b)
    return total / n


def evaluate(model, X, y, device, bs=8192):
    model.eval()
    preds = []
    with torch.no_grad():
        for s in range(0, len(y), bs):
            xb = torch.from_numpy(X[s:s + bs]).to(device)
            preds.append(torch.sigmoid(model(xb)).cpu().numpy())
    p = np.concatenate(preds)
    ll = -np.mean(y * np.log(p + 1e-9) + (1 - y) * np.log(1 - p + 1e-9))
    return {"AUC": round(float(auc(y, p)), 4), "LogLoss": round(float(ll), 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", choices=["lr", "fm", "deepfm"], default="deepfm")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--embed-dim", type=int, default=16)
    ap.add_argument("--hidden", type=int, nargs="+", default=[256, 128])
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cpu"
    torch.set_num_threads(max(1, os_cpu() - 2))

    train, val, test, edges, offsets, sizes, total_dim = load_data(args.samples)
    Xtr, ytr = frame_to_matrix(train, edges, offsets)
    Xva, yva = frame_to_matrix(val, edges, offsets)
    Xte, yte = frame_to_matrix(test, edges, offsets)
    print(f"train {Xtr.shape} pos_rate {ytr.mean():.3f}")

    model = Ranker(len(SPARSE) + len(CONT), list(sizes.values()), args.embed_dim, args.hidden, args.model)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    history = []
    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        loss = run_epoch(model, Xtr, ytr, opt, args, device)
        va = evaluate(model, Xva, yva, device)
        history.append({"epoch": ep, "train_loss": round(loss, 4), "val": va, "minutes": round((time.time() - t0) / 60, 1)})
        print(json.dumps(history[-1]))

    te = evaluate(model, Xte, yte, device)
    import os
    os.makedirs(args.out, exist_ok=True)
    torch.save(model.state_dict(), f"{args.out}/{args.model}.pt")
    with open(f"{args.out}/{args.model}_metrics.json", "w", encoding="utf-8") as fh:
        json.dump({"args": vars(args), "history": history, "test": te}, fh, ensure_ascii=False, indent=2)
    print("TEST", json.dumps(te))


def os_cpu():
    import os
    return os.cpu_count() or 4


if __name__ == "__main__":
    main()
