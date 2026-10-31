"""Build semantic IDs (SID) for every movie: content embedding -> RQ-KMeans.

Pipeline (all CPU, mirrors MiniOneRec's CPU SID builders):
  1. text      = title + genres + top genome tags
  2. embed     = sentence-transformers all-MiniLM-L6-v2 (384d), L2-normalised
  3. reduce    = PCA to --pq-d (default 32)
  4. quantise  = faiss ResidualQuantizer (RQ-KMeans), M levels x 256 codes
  5. disambiguate = TIGER-style: collided items get an extra token (0,1,2,...)
                   offset into the 4th level of the vocabulary

Outputs (artifacts/sid/):
  sid.csv            movie_id, sid (tokens joined by '-')
  sid_stats.json     codebook usage / collision diagnostics
"""
import argparse
import json
import os

import faiss
import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--content", default="data/movie_content.csv")
ap.add_argument("--out", default="artifacts/sid")
ap.add_argument("--pq-d", type=int, default=32)
ap.add_argument("--levels", type=int, default=3)
ap.add_argument("--nbits", type=int, default=8)
args = ap.parse_args()

os.makedirs(args.out, exist_ok=True)
df = pd.read_csv(args.content)
df["text"] = (
    df.title.fillna("") + " | " + df.genres.fillna("").str.replace("|", " ", regex=False)
    + " | " + df.top_tags.fillna("")
)

from sentence_transformers import SentenceTransformer  # noqa: E402

model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
emb = model.encode(df.text.tolist(), batch_size=256, show_progress_bar=True, normalize_embeddings=True)
emb = emb.astype(np.float32)

# PCA to pq-d
pca = faiss.PCAMatrix(emb.shape[1], args.pq_d)
pca.train(emb)
x = pca.apply(emb)
faiss.normalize_L2(x)

rq = faiss.ResidualQuantizer(args.pq_d, args.levels, args.nbits)
rq.beam_size = 16
rq.train(x)
codes = rq.compute_codes(x)  # [N, M] int
codes = np.asarray(codes).astype(np.int64)

# collision disambiguation: group by full SID tuple, append 4th token for dupes
key = [tuple(row) for row in codes]
groups = {}
for idx, k in enumerate(key):
    groups.setdefault(k, []).append(idx)

per_level_usage = [int(len(np.unique(codes[:, m]))) for m in range(args.levels)]
dup_groups = sum(1 for g in groups.values() if len(g) > 1)
dup_items = sum(len(g) for g in groups.values() if len(g) > 1)

sids = np.empty((len(df), args.levels + 1), dtype=np.int64)
sids[:, : args.levels] = codes
fourth = np.zeros(len(df), dtype=np.int64)
for k, members in groups.items():
    if len(members) > 1:
        for rank, idx in enumerate(sorted(members)):
            fourth[idx] = rank + 1
sids[:, args.levels] = fourth

out = pd.DataFrame({"movie_id": df.movie_id.values})
for m in range(args.levels + 1):
    out[f"t{m}"] = sids[:, m] + m * (2 ** args.nbits)   # per-level vocabulary offset
out["sid"] = out[[f"t{m}" for m in range(args.levels + 1)]].astype(str).agg("-".join, axis=1)
out.to_csv(os.path.join(args.out, "sid.csv"), index=False)

stats = {
    "n_movies": int(len(df)),
    "levels": args.levels,
    "codes_per_level": 2 ** args.nbits,
    "per_level_codebook_usage": per_level_usage,
    "collided_groups": int(dup_groups),
    "collided_items": int(dup_items),
    "collision_rate": round(dup_items / len(df), 4),
    "vocab_sizes": [2 ** args.nbits] * args.levels + [int(fourth.max()) + 1],
}
with open(os.path.join(args.out, "sid_stats.json"), "w", encoding="utf-8") as fh:
    json.dump(stats, fh, ensure_ascii=False, indent=2)
print(json.dumps(stats))
