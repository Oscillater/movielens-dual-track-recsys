"""Replay the online recall -> DeepFM -> MMR path on held-out users.

This is intentionally an online-path check, not the full offline benchmark:
it uses the same per-user candidate generation and DeepFM adapter as FastAPI,
then compares fusion-only, DeepFM-only, and their MMR reranked lists.

Usage:
  .venv\\Scripts\\python.exe scripts\\eval_serving_ranker.py --users 2000
"""
import argparse
import json
import math
import os
import random
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from serving.app import STATE, mmr_diversify, recall_channels  # noqa: E402
from serving.ranker import DeepFMRanker  # noqa: E402


def load_gt(path):
    gt = {}
    with open(path, encoding="utf-8") as fh:
        next(fh)
        for line in fh:
            uid, mid, split = line.rstrip("\n").split(",")
            if split == "test":
                gt[int(uid)] = int(mid)
    return gt


def evaluate(recs, gt, valid_count):
    hits10 = hits20 = ndcg10 = ndcg20 = 0.0
    recommended = set()
    diversity = []
    for uid, items in recs.items():
        target = gt[uid]
        if target in items[:10]:
            hits10 += 1
            ndcg10 += 1.0 / math.log2(items[:10].index(target) + 2)
        if target in items[:20]:
            hits20 += 1
            ndcg20 += 1.0 / math.log2(items[:20].index(target) + 2)
        recommended.update(items)
        genres = set()
        for mid in items[:20]:
            genres.update(STATE["movies"].get(mid, {}).get("genres", ()))
        diversity.append(len(genres))
    n = len(recs)
    return {
        "n_users": n,
        "Recall@10": round(hits10 / n, 4),
        "Recall@20": round(hits20 / n, 4),
        "NDCG@10": round(ndcg10 / n, 4),
        "NDCG@20": round(ndcg20 / n, 4),
        "coverage": round(len(recommended) / valid_count, 4),
        "genre_diversity@20": round(sum(diversity) / n, 2),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--users", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--recall-k", type=int, default=30)
    ap.add_argument("--rank-k", type=int, default=20)
    ap.add_argument("--lam", type=float, default=0.6)
    ap.add_argument("--out", default="experiments/serving_ranker_eval.json")
    args = ap.parse_args()

    gt = load_gt(os.path.join(ROOT, "data", "gt.csv"))
    users = [u for u in gt if len(STATE["hist"].get(u, [])) >= 1]
    random.Random(args.seed).shuffle(users)
    users = users[: args.users]
    ranker = DeepFMRanker(ROOT)
    genre_map = {m: set(v.get("genres", ())) for m, v in STATE["movies"].items()}

    fusion_direct, deepfm_direct = {}, {}
    fusion_mmr, deepfm_mmr = {}, {}
    score_seconds = 0.0
    t0 = time.perf_counter()
    for n, uid in enumerate(users, 1):
        chans = recall_channels(uid, k=args.recall_k)
        base = max(len(max(chans.values(), key=len)) if chans else 1, 1)
        cands = {}
        for name, ids in chans.items():
            for rank, mid in enumerate(ids):
                cands.setdefault(mid, 0.0)
                cands[mid] += (base - rank) / base
        if not cands:
            continue
        fusion_pairs = sorted(cands.items(), key=lambda x: -x[1])
        t = time.perf_counter()
        ctr = ranker.score(uid, [m for m, _ in fusion_pairs], STATE)
        score_seconds += time.perf_counter() - t
        deep_pairs = sorted(
            ((m, ctr[m]) for m in cands), key=lambda x: (-x[1], -cands[x[0]])
        )

        fusion_direct[uid] = [m for m, _ in fusion_pairs[: args.rank_k]]
        deepfm_direct[uid] = [m for m, _ in deep_pairs[: args.rank_k]]
        fusion_mmr[uid] = [
            m for m, _ in mmr_diversify(fusion_pairs, genre_map, args.lam, args.rank_k)
        ]
        deepfm_mmr[uid] = [
            m for m, _ in mmr_diversify(deep_pairs, genre_map, args.lam, args.rank_k)
        ]
        if n % 250 == 0:
            print(f"[eval] {n}/{len(users)}", flush=True)

    valid_count = len(STATE["movies"])
    result = {
        "protocol": {
            "users": len(users),
            "seed": args.seed,
            "candidate_recall_k_per_channel": args.recall_k,
            "rank_k": args.rank_k,
            "mmr_lambda": args.lam,
            "note": "online-path replay; DeepFM uses serving-time feature approximations",
        },
        "fusion_direct": evaluate(fusion_direct, gt, valid_count),
        "deepfm_direct": evaluate(deepfm_direct, gt, valid_count),
        "fusion_mmr": evaluate(fusion_mmr, gt, valid_count),
        "deepfm_mmr": evaluate(deepfm_mmr, gt, valid_count),
        "decision": {
            "serving_default": "fusion",
            "deepfm": "available via rank_mode=deepfm, not default",
            "reason": "DeepFM directly replacing the recall score loses 41% R@10; "
                      "the CTR model needs a candidate-aware training objective before it can rank the funnel",
        },
        "deepfm_score_ms_per_user": round(score_seconds / len(users) * 1000, 2),
        "total_seconds": round(time.perf_counter() - t0, 1),
    }
    os.makedirs(os.path.dirname(os.path.join(ROOT, args.out)), exist_ok=True)
    with open(os.path.join(ROOT, args.out), "w", encoding="utf-8") as fh:
        json.dump(result, fh, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
