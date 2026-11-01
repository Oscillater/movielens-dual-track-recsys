"""Generative recommendation (TIGER / OneRec style, CPU-light adaptation).

Replaces MiniOneRec's LLM SFT with a from-scratch decoder-only Transformer
that autoregressively generates the next item's semantic ID; constrained
beam search over the SID trie guarantees only real items can be generated.

  train:    python genrec.py train --seqs data/seqs.csv --sid artifacts/sid/sid.csv \
                --stats artifacts/sid/sid_stats.json --out artifacts/genrec
  evaluate: python genrec.py eval --... --split test --users 5000

Protocol matches sasrec.py / evaluate.py exactly (same sequences, same
leave-last-out targets, full item space via constrained decoding).
"""
import argparse
import json
import math
import os
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


# ------------------------------------------------------------------ data
class SidSpace:
    def __init__(self, sid_csv, stats_json):
        sid = pd.read_csv(sid_csv)
        with open(stats_json, encoding="utf-8") as fh:
            stats = json.load(fh)
        self.vocab_sizes = stats["vocab_sizes"]              # per level
        self.offsets = [int(stats["codes_per_level"]) * m for m in range(stats["levels"])] + [
            stats["levels"] * stats["codes_per_level"]
        ]
        self.PAD = sum(self.vocab_sizes)                     # last id = pad
        self.item2sid = {
            int(r.movie_id): [int(r.t0), int(r.t1), int(r.t2), int(r.t3)]
            for r in sid.itertuples()
        }
        self.sid2item = {tuple(v): k for k, v in self.item2sid.items()}
        # trie over valid SID token paths (variable depth 3 or 4)
        self.trie = {}
        for toks in self.item2sid.values():
            node = self.trie
            for depth, t in enumerate(toks):
                terminal = depth == len(toks) - 1
                node = node.setdefault(t, {})
                if terminal:
                    node["#"] = True

    def tokens(self, movie_id):
        return self.item2sid[movie_id]


def build_token_seqs(seqs_csv, space):
    """user -> token sequence of the train-split history."""
    seqs = {}
    with open(seqs_csv, encoding="utf-8") as fh:
        next(fh)
        for line in fh:
            uid, _, rest = line.rstrip("\n").partition(",")
            if not rest:
                continue
            toks = []
            for mid in map(int, rest.split()):
                if mid in space.item2sid:
                    toks.extend(space.item2sid[mid])
            seqs[int(uid)] = np.array(toks, dtype=np.int64)
    return seqs


def load_gt(path, split):
    out = {}
    with open(path, encoding="utf-8") as fh:
        next(fh)
        for line in fh:
            uid, mid, sp = line.rstrip("\n").split(",")
            if sp == split:
                out[int(uid)] = int(mid)
    return out


# ------------------------------------------------------------------ model
class Block(nn.Module):
    def __init__(self, d, heads, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(d)
        self.ffn = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Dropout(dropout), nn.Linear(4 * d, d))

    def forward(self, x, causal):
        h = self.ln1(x)
        a, _ = self.attn(h, x, x, attn_mask=causal, need_weights=False)
        x = x + a
        return x + self.ffn(self.ln2(x))


class GenRec(nn.Module):
    def __init__(self, vocab, d=128, layers=4, heads=4, maxlen=72, dropout=0.1):
        super().__init__()
        self.maxlen = maxlen
        self.tok_emb = nn.Embedding(vocab + 1, d, padding_idx=vocab)
        self.pos_emb = nn.Embedding(maxlen, d)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([Block(d, heads, dropout) for _ in range(layers)])
        self.ln = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab + 1, bias=False)
        self.head.weight = self.tok_emb.weight  # weight tying

    def forward(self, x):
        B, L = x.shape
        pos = torch.arange(L, device=x.device).unsqueeze(0).expand(B, L)
        h = self.drop(self.tok_emb(x) + self.pos_emb(pos))
        causal = torch.triu(torch.ones(L, L, dtype=torch.bool, device=x.device), diagonal=1)
        for blk in self.blocks:
            h = blk(h, causal)
        return self.head(self.ln(h))                            # [B, L, V+1]


def lr_schedule(step, warmup, total):
    if step < warmup:
        return step / max(1, warmup)
    return max(0.1, 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, total - warmup))))


def train(args, space, seqs, val_gt):
    device = "cpu"
    torch.set_num_threads(max(1, os.cpu_count() - 2))
    torch.manual_seed(args.seed)
    vocab = space.PAD
    model = GenRec(vocab, args.d, args.layers, args.heads, args.maxlen, args.dropout)
    opt = torch.optim.AdamW(model.parameters(), lr=1.0, betas=(0.9, 0.95), weight_decay=0.1)
    users = np.array([u for u, s in seqs.items() if s.size >= 6])
    lens = np.array([seqs[u].size for u in users])
    cum = np.cumsum(lens)
    steps_per_epoch = max(1, args.users_per_epoch // args.batch)
    total_steps = steps_per_epoch * args.epochs
    step = 0
    os.makedirs(args.out, exist_ok=True)
    best = (-1.0, None)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        tot, toks = 0.0, 0
        for _ in range(steps_per_epoch):
            pick = np.searchsorted(cum, np.random.rand(args.batch) * cum[-1], side="right")
            pick = np.minimum(pick, len(users) - 1)
            batch, tgt = [], []
            for u in users[pick]:
                s = seqs[u][-args.maxlen - 1:]
                x = s[:-1]
                y = s[1:]
                pad = args.maxlen - len(x)
                batch.append([vocab] * pad + x.tolist())
                tgt.append([vocab] * pad + y.tolist())
            x = torch.tensor(batch, dtype=torch.long)
            y = torch.tensor(tgt, dtype=torch.long)
            logits = model(x)
            mask = y != vocab
            loss = nn.functional.cross_entropy(logits.reshape(-1, vocab + 1), y.reshape(-1),
                                               ignore_index=vocab)
            for g in opt.param_groups:
                g["lr"] = args.lr * lr_schedule(step, total_steps // 10, total_steps)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item() * mask.sum().item()
            toks += mask.sum().item()
            step += 1
        hr = quick_val(model, seqs, val_gt, space, users_sample=1500, beam=8, device=device)
        rec = {"epoch": epoch, "tok_loss": round(tot / max(1, toks), 4), "val_HR@10": round(hr, 4),
               "minutes": round((time.time() - t0) / 60, 1)}
        print(json.dumps(rec), flush=True)
        if hr > best[0]:
            best = (hr, {k: v.detach().clone() for k, v in model.state_dict().items()})
            torch.save(best[1], os.path.join(args.out, "genrec.pt"))
    with open(os.path.join(args.out, "train_done.json"), "w") as fh:
        json.dump({"best_val_HR@10": best[0], "epochs": args.epochs}, fh)


@torch.no_grad()
def quick_val(model, seqs, gt, space, users_sample, beam, device):
    users = [u for u, t in gt.items() if seqs.get(u, np.array([])).size >= 6]
    random.Random(0).shuffle(users)
    users = users[:users_sample]
    hits = 0
    for u in users:
        recs = generate(model, seqs[u], space, beam, 10, exclude=set(int(i) for i in seqs[u]))
        if gt[u] in recs:
            hits += 1
    return hits / max(1, len(users))


@torch.no_grad()
def generate(model, hist_tokens, space, beam, topk, device="cpu", exclude=None):
    """Constrained beam search: every beam carries its trie node and advances
    token-by-token; paths completing at a '#' terminal become candidates.
    TIGER-style: top-K items = top-K distinct completed SIDs."""
    model.eval()
    ctx = hist_tokens[-model.maxlen:].tolist()
    if not ctx:
        return []
    beams = [(0.0, [], space.trie)]        # (cum_logprob, generated_tokens, trie_node)
    done = []
    for _ in range(4):                     # SID depth is 3 or 4 tokens
        if not beams:
            break
        cand = []
        for score, toks, node in beams:
            x = torch.tensor([(ctx + toks)[-model.maxlen:]], dtype=torch.long)
            logp = torch.log_softmax(model(x)[0, -1], dim=-1)
            for t in torch.topk(logp, beam * 3).indices.tolist():
                if t in node and t != space.PAD:
                    child = node[t]
                    ns = score + float(logp[t])
                    if "#" in child:
                        done.append((ns, toks + [t]))
                    else:
                        cand.append((ns, toks + [t], child))
        if not cand:
            break
        cand.sort(key=lambda p: -p[0])
        beams = cand[:beam]
    done.sort(key=lambda p: -p[0])
    out, seen_sids = [], set()
    for _, toks in done:
        sid = tuple(toks)
        if sid in seen_sids:
            continue
        seen_sids.add(sid)
        mid = space.sid2item.get(sid)
        if mid is not None and (exclude is None or mid not in exclude):
            out.append(mid)
        if len(out) >= topk:
            break
    return out


def args_maxlen(model):
    return model.maxlen


def evaluate_full(args, space, seqs, gt):
    users = [u for u, t in gt.items() if seqs.get(u, np.array([])).size >= 6]
    random.Random(args.seed).shuffle(users)
    if args.users:
        users = users[: args.users]
    model = GenRec(space.PAD, args.d, args.layers, args.heads, args.maxlen, args.dropout)
    model.load_state_dict(torch.load(os.path.join(args.out, "genrec.pt"), map_location="cpu"))
    torch.set_num_threads(max(1, os.cpu_count() - 2))
    hits10 = hits50 = ndcg10 = ndcg50 = 0
    t0 = time.time()
    recs_map = {}
    for u in users:
        recs = generate(model, seqs[u], space, beam=args.beam, topk=50,
                        exclude=set(int(i) for i in seqs[u]))
        recs_map[u] = recs
        tgt = gt[u]
        if tgt in recs[:10]:
            hits10 += 1
            ndcg10 += 1.0 / np.log2(recs[:10].index(tgt) + 2)
        if tgt in recs[:50]:
            hits50 += 1
            ndcg50 += 1.0 / np.log2(recs[:50].index(tgt) + 2)
    n = len(users)
    result = {
        "n_users": n,
        "Recall@10": round(hits10 / n, 4),
        "Recall@50": round(hits50 / n, 4),
        "NDCG@10": round(float(ndcg10 / n), 4),
        "NDCG@50": round(float(ndcg50 / n), 4),
        "latency_ms_per_user": round((time.time() - t0) / n * 1000, 1),
    }
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "genrec_test_metrics.json"), "w", encoding="utf-8") as fh:
        json.dump(result, fh, ensure_ascii=False, indent=2)
    print(json.dumps(result))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["train", "eval"])
    ap.add_argument("--seqs", default="data/seqs.csv")
    ap.add_argument("--gt", default="data/gt.csv")
    ap.add_argument("--sid", default="artifacts/sid/sid.csv")
    ap.add_argument("--stats", default="artifacts/sid/sid_stats.json")
    ap.add_argument("--out", default="artifacts/genrec")
    ap.add_argument("--split", default="test")
    ap.add_argument("--users", type=int, default=5000)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--users-per-epoch", type=int, default=30000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--maxlen", type=int, default=72)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--beam", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    space = SidSpace(args.sid, args.stats)
    seqs = build_token_seqs(args.seqs, space)
    if args.mode == "train":
        val_gt = load_gt(args.gt, "val")
        train(args, space, seqs, val_gt)
    else:
        gt = load_gt(args.gt, args.split)
        evaluate_full(args, space, seqs, gt)


if __name__ == "__main__":
    main()
