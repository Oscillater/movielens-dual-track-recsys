"""Unified recall evaluation across channels on the shared leave-last-out test set.

Channels: pop | itemcf | als | sasrec (genrec slot added later).
Uniform protocol for every channel:
  - history  = time-ordered TRAIN-split sequence (dws_user_seq)
  - target   = the single held-out test item (dws_gt, split=test)
  - ranking  = over all valid items, with train-seen items EXCLUDED
  - metrics  = Recall@K (single target => equals HR@K), NDCG@K, coverage,
               intra-list genre diversity, mean scoring latency per user

Usage:
  python evaluate.py --data data --sasrec artifacts/sasrec --out experiments --k 50
"""
import argparse
import json
import os
import time
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix


def load_inputs(data_dir):
    seqs = {}
    with open(os.path.join(data_dir, "seqs.csv"), encoding="utf-8") as fh:
        next(fh)
        for line in fh:
            uid, _, rest = line.rstrip("\n").partition(",")
            if rest:
                seqs[int(uid)] = [int(x) for x in rest.split()]
    test_gt = {}
    with open(os.path.join(data_dir, "gt.csv"), encoding="utf-8") as fh:
        next(fh)
        for line in fh:
            uid, mid, split = line.rstrip("\n").split(",")
            if split == "test":
                test_gt[int(uid)] = int(mid)
    # item_stats.csv: movie_id,n_ratings,avg_rating,title,year — titles may
    # contain unquoted commas, so parse only the first three numeric columns
    rows = []
    with open(os.path.join(data_dir, "item_stats.csv"), encoding="utf-8") as fh:
        next(fh)
        for line in fh:
            p = line.rstrip("\n").split(",")
            if len(p) >= 3:
                rows.append((int(p[0]), int(p[1]), float(p[2])))
    items = pd.DataFrame(rows, columns=["movie_id", "n_ratings", "avg_rating"])
    genres_map = {}
    with open(os.path.join(data_dir, "movies.csv"), encoding="utf-8") as fh:
        next(fh)
        for line in fh:
            parts = line.rstrip("\n").split(",")
            mid = int(parts[0])
            g = parts[-1] if len(parts) >= 4 else ""
            # genres column may itself contain no comma; title may contain commas
            # csv layout: movie_id,title,year,genres -> genres is last field
            genres_map[mid] = set(g.split("|")) - {""} if g else set()
    return seqs, test_gt, items, genres_map


class Channel:
    name = "base"

    def topk(self, users, k):
        raise NotImplementedError


class PopChannel(Channel):
    name = "pop"

    def __init__(self, items):
        self.top500 = items.sort_values("n_ratings", ascending=False).movie_id.head(500).tolist()

    def topk(self, users, k, seen_map, valid):
        out = []
        for u in users:
            s = seen_map[u]
            recs = []
            for i in self.top500:
                if i not in s:
                    recs.append(i)
                    if len(recs) >= k:
                        break
            out.append(recs)
        return out


class ItemCFChannel(Channel):
    name = "itemcf"

    def __init__(self, data_dir, items):
        co = pd.read_csv(os.path.join(data_dir, "cooccur.csv"))
        n_map = dict(zip(items.movie_id.astype(int), items.n_ratings.astype(float)))
        rows, cols, vals = [], [], []
        for i, j, c in co.itertuples(index=False):
            sim = c / np.sqrt(n_map.get(int(i), 1) * n_map.get(int(j), 1))
            rows.extend([int(i), int(j)])
            cols.extend([int(j), int(i)])
            vals.extend([sim, sim])
        n_items = int(items.movie_id.max()) + 1
        self.S = csr_matrix((vals, (rows, cols)), shape=(n_items, n_items))

    def topk(self, users, k, seen_map, valid):
        out = []
        t0 = time.perf_counter()
        for u in users:
            hist = [i for i in seen_map[u] if i < self.S.shape[0]]
            if not hist:
                out.append([])
                continue
            scores = np.asarray(self.S[hist].sum(axis=0)).ravel()
            for i in seen_map[u]:
                if i < len(scores):
                    scores[i] = -np.inf
            idx = np.argpartition(-scores, k)[:k] if len(scores) > k else np.argsort(-scores)[:k]
            idx = idx[np.argsort(-scores[idx])]
            out.append([int(i) for i in idx if scores[i] > -np.inf][:k])
        self.latency = (time.perf_counter() - t0) / len(users)
        return out


class ALSChannel(Channel):
    name = "als"

    def __init__(self, data_dir, valid_items):
        U = pd.read_csv(os.path.join(data_dir, "user_factors.csv"))
        I = pd.read_csv(os.path.join(data_dir, "item_factors.csv"))
        fcols = [c for c in U.columns if c != "user_id"]
        self.u_ids = U.user_id.values
        self.U = U[fcols].values
        fcols_i = [c for c in I.columns if c != "movie_id"]
        self.i_ids = I.movie_id.values.astype(int)
        self.I = I[fcols_i].values
        self.u_row = {u: r for r, u in enumerate(self.u_ids)}
        self.i_row = {i: r for r, i in enumerate(self.i_ids)}

    def topk(self, users, k, seen_map, valid):
        out = []
        t0 = time.perf_counter()
        for u in users:
            if u not in self.u_row:
                out.append([])
                continue
            v = self.U[self.u_row[u]]
            scores = self.I @ v
            for i in seen_map[u]:
                if i in self.i_row:
                    scores[self.i_row[i]] = -np.inf
            idx = np.argpartition(-scores, k)[:k]
            idx = idx[np.argsort(-scores[idx])]
            out.append([int(self.i_ids[r]) for r in idx][:k])
        self.latency = (time.perf_counter() - t0) / len(users)
        return out


class SASRecChannel(Channel):
    name = "sasrec"

    def __init__(self, data_dir, sasrec_dir, maxlen=50):
        import torch
        from sasrec import SASRec
        self.torch = torch
        ids = np.loadtxt(os.path.join(sasrec_dir, "item_ids.txt"), dtype=np.int64)
        emb = np.load(os.path.join(sasrec_dir, "item_emb.npy"))
        self.item_ids = ids
        state = torch.load(os.path.join(sasrec_dir, "sasrec.pt"), map_location="cpu")
        d = emb.shape[1]
        self.model = SASRec(state["item_emb.weight"].shape[0], d=d)
        self.model.load_state_dict(state)
        self.model.eval()
        self.torch.set_num_threads(max(1, os.cpu_count() - 2))
        self.maxlen = maxlen

    def topk(self, users, k, seen_map, valid):
        torch = self.torch
        row_of = {int(i): r for r, i in enumerate(self.item_ids)}
        out = []
        t0 = time.perf_counter()
        B = 1024
        for s in range(0, len(users), B):
            chunk = users[s:s + B]
            hist_b, mask_rows = [], []
            for u in chunk:
                h = [i for i in seen_map[u] if i in row_of][-self.maxlen:]
                pad = self.maxlen - len(h)
                hist_b.append([0] * pad + h)
            with torch.no_grad():
                x = torch.tensor(hist_b, dtype=torch.long)
                hidden = self.model(x)[:, -1, :]
                emb = self.model.item_emb(self.item_ids_t) if hasattr(self, "item_ids_t") else None
            scores = (hidden @ self.model.item_emb.weight.T).numpy()
            for row, u in enumerate(chunk):
                sc = scores[row]
                for i in seen_map[u]:
                    if i < len(sc):
                        sc[i] = -np.inf
                idx = np.argpartition(-sc, k)[:k] if len(sc) > k else np.argsort(-sc)[:k]
                idx = idx[np.argsort(-sc[idx])]
                out.append([int(i) for i in idx if sc[i] > -np.inf][:k])
        self.latency = (time.perf_counter() - t0) / len(users)
        return out


def genrec_placeholder(users, k, seen_map, valid):
    return [[] for _ in users]


def metrics_for(users, topk_lists, test_gt, genres_map, all_valid):
    hits10 = hits50 = ndcg10 = ndcg50 = 0
    recommended = set()
    diversity = []
    for u, recs in zip(users, topk_lists):
        tgt = test_gt[u]
        if tgt in recs[:10]:
            hits10 += 1
            ndcg10 += 1.0 / np.log2(recs[:10].index(tgt) + 2)
        if tgt in recs[:50]:
            hits50 += 1
            ndcg50 += 1.0 / np.log2(recs[:50].index(tgt) + 2)
        recommended.update(recs)
        gs = set()
        for i in recs:
            gs |= genres_map.get(i, set())
        diversity.append(len(gs))
    n = len(users)
    return {
        "n_users": n,
        "Recall@10": round(hits10 / n, 4),
        "Recall@50": round(hits50 / n, 4),
        "NDCG@10": round(float(ndcg10 / n), 4),
        "NDCG@50": round(float(ndcg50 / n), 4),
        "coverage": round(len(recommended) / len(all_valid), 4),
        "genre_diversity@50": round(float(np.mean(diversity)), 2),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--sasrec", default="artifacts/sasrec")
    ap.add_argument("--out", default="experiments")
    ap.add_argument("--k", type=int, default=50)
    ap.add_argument("--eval-users", type=int, default=0, help="0 = full test set")
    ap.add_argument("--channels", nargs="+", default=["pop", "itemcf", "als", "sasrec"])
    args = ap.parse_args()

    seqs, test_gt, items, genres_map = load_inputs(args.data)
    users_all = sorted(u for u in test_gt if len(seqs.get(u, [])) >= 1)
    users = users_all if not args.eval_users else users_all[: args.eval_users]
    seen_map = {u: set(int(i) for i in seqs[u]) for u in users}
    valid = set(items.movie_id.astype(int))

    os.makedirs(args.out, exist_ok=True)
    out_path = os.path.join(args.out, "recall_results.json")
    results = {}
    if os.path.exists(out_path):
        with open(out_path, encoding="utf-8") as fh:
            results = json.load(fh)
    for name in args.channels:
        t0 = time.perf_counter()
        if name == "pop":
            ch = PopChannel(items)
            recs = ch.topk(users, args.k, seen_map, valid)
            ch.latency = (time.perf_counter() - t0) / len(users)
        elif name == "itemcf":
            ch = ItemCFChannel(args.data, items)
            recs = ch.topk(users, args.k, seen_map, valid)
        elif name == "als":
            ch = ALSChannel(args.data, valid)
            recs = ch.topk(users, args.k, seen_map, valid)
        elif name == "sasrec":
            ch = SASRecChannel(args.data, args.sasrec)
            recs = ch.topk(users, args.k, seen_map, valid)
        elif name == "genrec":
            recs = genrec_placeholder(users, args.k, seen_map, valid)
            ch = type("C", (), {"latency": 0.0, "name": "genrec"})()
        else:
            raise SystemExit(f"unknown channel {name}")
        m = metrics_for(users, recs, test_gt, genres_map, valid)
        m["latency_ms_per_user"] = round(ch.latency * 1000, 2)
        results[name] = m
        print(name, json.dumps(m), flush=True)
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(results, fh, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
