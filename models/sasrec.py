"""SASRec sequence recall baseline, trained on CPU (host).

Protocol notes (must match the generative branch for a fair comparison):
  - History = time-ordered TRAIN-split sequence per user (dws_user_seq).
  - Targets = leave-last-out items from dws_gt (val for model selection,
    test for the final reported numbers).
  - Evaluation = full ranking over all 62,424 items (no negative sampling).

Usage:
  python sasrec.py --seqs data/seqs.csv --gt data/gt.csv --out artifacts/sasrec \
      --epochs 10 --positions-per-epoch 500000
"""
import argparse
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------- data
def load_seqs(path):
    """user_id -> np.int32 array (memory-lean; 24.7M ints as python lists ~ 700MB)."""
    seqs = {}
    with open(path, encoding="utf-8") as fh:
        next(fh)  # header
        for line in fh:
            uid, _, rest = line.rstrip("\n").partition(",")
            if rest:
                seqs[int(uid)] = np.fromstring(rest, dtype=np.int32, sep=" ")
    return seqs


def load_gt(path):
    gt = {"val": {}, "test": {}}
    with open(path, encoding="utf-8") as fh:
        next(fh)
        for line in fh:
            uid, mid, split = line.rstrip("\n").split(",")
            gt[split][int(uid)] = int(mid)
    return gt


# ---------------------------------------------------------------- model
class SASRec(nn.Module):
    def __init__(self, n_items, d=64, layers=2, heads=2, maxlen=50, dropout=0.2):
        super().__init__()
        self.maxlen = maxlen
        self.item_emb = nn.Embedding(n_items + 1, d, padding_idx=0)
        self.pos_emb = nn.Embedding(maxlen, d)
        self.emb_drop = nn.Dropout(dropout)
        self.attn_ln = nn.ModuleList([nn.LayerNorm(d) for _ in range(layers)])
        self.attn = nn.ModuleList(
            [nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True) for _ in range(layers)]
        )
        self.ffn_ln = nn.ModuleList([nn.LayerNorm(d) for _ in range(layers)])
        self.ffn = nn.ModuleList(
            [nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Dropout(dropout), nn.Linear(4 * d, d)) for _ in range(layers)]
        )
        self.last_ln = nn.LayerNorm(d)
        nn.init.normal_(self.item_emb.weight, std=0.02)
        with torch.no_grad():
            self.item_emb.weight[0].zero_()

    def forward(self, seq):  # seq: [B, L] left-padded, 0 = pad
        B, L = seq.shape
        pos = torch.arange(L, device=seq.device).unsqueeze(0).expand(B, L)
        x = self.emb_drop(self.item_emb(seq) * (self.item_emb.embedding_dim ** 0.5) + self.pos_emb(pos))
        causal = torch.triu(torch.ones(L, L, dtype=torch.bool, device=seq.device), diagonal=1)
        pad_mask = seq == 0
        for ln_a, attn, ln_f, ffn in zip(self.attn_ln, self.attn, self.ffn_ln, self.ffn):
            q = ln_a(x)
            a, _ = attn(q, x, x, attn_mask=causal, key_padding_mask=pad_mask, need_weights=False)
            x = x + a
            x = x + ffn(ln_f(x))
        return self.last_ln(x)  # [B, L, d]


def make_batch(seqs, users, ts, args, device):
    seq_b, pos_b = [], []
    for user, t in zip(users, ts):
        hist = seqs[user][:t][-args.maxlen:]
        pad = args.maxlen - len(hist)
        seq_b.append([0] * pad + hist.tolist())
        pos_b.append(int(seqs[user][t]))
    return (
        torch.tensor(seq_b, dtype=torch.long, device=device),
        torch.tensor(pos_b, dtype=torch.long, device=device),
    )


def full_rank_scores(model, hist_batch, item_ids_t, device):
    """hist_batch: [B, L] -> scores [B, n_valid_items] at the last position."""
    model.eval()
    with torch.no_grad():
        h = model(hist_batch.to(device))[:, -1, :]          # [B, d]
        emb = model.item_emb(item_ids_t)                    # [V, d]
        return h @ emb.T                                    # [B, V]


def evaluate(model, seqs, targets, item_ids_t, device, args):
    users = [u for u, tgt in targets.items() if len(seqs.get(u, [])) >= 1]
    if args.eval_users and len(users) > args.eval_users:
        rng = random.Random(args.seed)
        users = rng.sample(users, args.eval_users)
    hr10 = hr50 = ndcg10 = ndcg50 = 0.0
    B = 1024
    for s in range(0, len(users), B):
        chunk = users[s:s + B]
        hist_b = []
        for u in chunk:
            h = seqs[u][-args.maxlen:]
            hist_b.append([0] * (args.maxlen - len(h)) + h.tolist())
        scores = full_rank_scores(model, torch.tensor(hist_b, dtype=torch.long), item_ids_t, device)
        top = torch.topk(scores, 50, dim=1).indices.tolist()
        for row, u in enumerate(chunk):
            tgt = targets[u]
            rank_list = [item_ids_t[k].item() for k in top[row]]
            if tgt in rank_list[:10]:
                hr10 += 1
                ndcg10 += 1.0 / np.log2(rank_list[:10].index(tgt) + 2)
            if tgt in rank_list[:50]:
                hr50 += 1
                ndcg50 += 1.0 / np.log2(rank_list[:50].index(tgt) + 2)
    n = len(users)
    return {"n_users": n, "HR@10": hr10 / n, "HR@50": hr50 / n, "NDCG@10": ndcg10 / n, "NDCG@50": ndcg50 / n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seqs", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--positions-per-epoch", type=int, default=500_000)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--heads", type=int, default=2)
    ap.add_argument("--maxlen", type=int, default=50)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--negs", type=int, default=1024)
    ap.add_argument("--eval-users", type=int, default=50_000)
    ap.add_argument("--init-from", default=None, help="continue training from a checkpoint")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    device = "cpu"
    torch.set_num_threads(max(1, os.cpu_count() - 2))

    seqs = load_seqs(args.seqs)
    gt = load_gt(args.gt)
    args.n_items = max(max(int(s.max()) for s in seqs.values() if s.size) + 1,
                       max(gt["test"].values()) + 1)
    # position sampling without materialising a pool: users weighted by history
    # length, target index t ~ U{1..len-1}; fully vectorised
    user_arr = np.array([u for u, s in seqs.items() if s.size >= 2], dtype=np.int64)
    lens = np.array([seqs[u].size for u in user_arr], dtype=np.int64)
    cum = np.cumsum(lens)
    rng_np = np.random.default_rng(args.seed)

    def sample_positions(n):
        uidx = np.minimum(np.searchsorted(cum, rng_np.random(n) * cum[-1], side="right"),
                          len(user_arr) - 1)
        ts = 1 + (rng_np.random(n) * (lens[uidx] - 1)).astype(np.int64)
        return user_arr[uidx], np.minimum(ts, lens[uidx] - 1)

    item_ids = np.array(sorted({int(i) for s in seqs.values() if s.size for i in np.unique(s)} | set(gt["test"].values())))
    item_ids_t = torch.tensor(item_ids, dtype=torch.long)

    model = SASRec(args.n_items, args.d, args.layers, args.heads, args.maxlen, args.dropout)
    if args.init_from and os.path.exists(args.init_from):
        model.load_state_dict(torch.load(args.init_from, map_location="cpu"))
        print("resumed from", args.init_from)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    os.makedirs(args.out, exist_ok=True)
    history, best_ndcg, best_state = [], -1.0, None
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        n_pos = min(args.positions_per_epoch, int(cum[-1]))
        users_b, ts_b = sample_positions(n_pos)
        total, hits = 0.0, 0
        for s in range(0, n_pos, args.batch):
            seq_b, pos_b = make_batch(seqs, users_b[s:s + args.batch], ts_b[s:s + args.batch], args, device)
            h = model(seq_b)[:, -1, :]                                        # [B, d]
            logits = h @ model.item_emb.weight.T                              # [B, n_items+1]
            pos_logits = logits.gather(1, pos_b.unsqueeze(1)).squeeze(1)      # [B]
            neg_idx = torch.randint(1, args.n_items, (pos_b.shape[0], args.negs), device=device)
            neg_logits = logits.gather(1, neg_idx)                            # [B, K]
            loss = (
                -torch.log(torch.sigmoid(pos_logits) + 1e-9).mean()
                - torch.log(1 - torch.sigmoid(neg_logits) + 1e-9).mean()
            )
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * pos_b.shape[0]
            hits += (pos_logits.unsqueeze(1) > neg_logits).all(dim=1).sum().item()
        val_m = evaluate(model, seqs, gt["val"], item_ids_t, device, args)
        history.append({"epoch": epoch, "loss": total / n_pos, "acc": hits / n_pos,
                        "val": val_m, "minutes": round((time.time() - t0) / 60, 1)})
        print(json.dumps(history[-1], ensure_ascii=False))
        if val_m["NDCG@10"] > best_ndcg:
            best_ndcg = val_m["NDCG@10"]
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    test_m = evaluate(model, seqs, gt["test"], item_ids_t, device, args)
    emb = model.item_emb.weight.detach().numpy()[item_ids]  # [V, d] aligned with item_ids
    np.save(os.path.join(args.out, "item_emb.npy"), emb)
    with open(os.path.join(args.out, "item_ids.txt"), "w") as fh:
        fh.write("\n".join(map(str, item_ids.tolist())))
    torch.save(best_state, os.path.join(args.out, "sasrec.pt"))
    result = {"args": vars(args), "history": history, "val_best_ndcg10": best_ndcg, "test": test_m}
    with open(os.path.join(args.out, "metrics.json"), "w", encoding="utf-8") as fh:
        json.dump(result, fh, ensure_ascii=False, indent=2)
    print("TEST", json.dumps(test_m))


if __name__ == "__main__":
    main()
