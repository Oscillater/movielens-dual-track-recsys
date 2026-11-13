"""Online serving: multi-channel recall -> (optional DeepFM rank) -> MMR.

Run on the host (artifacts live in project/):
  .venv\Scripts\python -m uvicorn serving.app:app --port 8000

Channels available depend on artifacts present at startup; missing ones are
skipped and reported by /health.
"""
import json
import os
import re
import sys
import threading
import time
from collections import defaultdict

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "models"))

from fastapi import FastAPI, Query  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402

from serving.ranker import DeepFMRanker, artifacts_available  # noqa: E402

app = FastAPI(title="Movie Search & Recommendation (Spark/Hive/SparkML + GenRec)")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

STATE = {}
_RANKER_LOCK = threading.Lock()
_RANKER = {"loaded": False, "object": None, "error": None}


def get_ranker():
    """Lazily load the NumPy DeepFM adapter on the first DeepFM request."""
    if _RANKER["loaded"]:
        return _RANKER["object"]
    with _RANKER_LOCK:
        if _RANKER["loaded"]:
            return _RANKER["object"]
        try:
            _RANKER["object"] = DeepFMRanker(ROOT)
            print("[serving] DeepFM ranker loaded", flush=True)
        except Exception as exc:  # noqa: BLE001 - service must fall back to fusion
            _RANKER["error"] = f"{type(exc).__name__}: {exc}"
            print(f"[serving] DeepFM unavailable, using fusion scores: {exc}", flush=True)
        _RANKER["loaded"] = True
        return _RANKER["object"]


def parse_movies(path, content_path=None):
    """movies.csv layout: movie_id,title,year,genres (Spark-quoted CSV).

    ``content_path`` (movie_content.csv) adds the genome ``top_tags`` column so
    that /search can match tags as well as titles and genres.
    """
    import csv
    rows = []
    with open(path, encoding="utf-8", newline="") as fh:
        for r in csv.reader(fh, escapechar="\\"):
            if len(r) < 4 or r[0] == "movie_id":
                continue
            title = r[1].strip()
            # Spark CSV quoting leaves symmetric wrapping quotes on comma-titles
            if len(title) >= 2 and title[0] == '"' and title[-1] == '"':
                title = title[1:-1].strip()
            try:
                mid = int(r[0])
            except ValueError:
                continue
            year = r[2] or (re.search(r"\((\d{4})\)", title) or [None, ""])[1]
            rows.append({
                "movie_id": mid,
                "title": title,
                "year": year,
                "genres": [g for g in r[3].split("|") if g],
                "top_tags": "",
            })
    movies = {r["movie_id"]: r for r in rows}
    if content_path and os.path.exists(content_path):
        # Spark CSV escapes embedded quotes with a backslash; the csv module
        # handles it as long as the escapechar matches.
        with open(content_path, encoding="utf-8", newline="") as fh:
            for r in csv.reader(fh, escapechar="\\"):
                if len(r) < 5 or r[0] == "movie_id":
                    continue
                try:
                    mid = int(r[0])
                except ValueError:
                    continue
                if mid in movies:
                    movies[mid]["top_tags"] = r[4].strip()
    return movies


def load_state():
    data = os.path.join(ROOT, "data")
    st = {"movies": parse_movies(
        os.path.join(data, "movies.csv"),
        os.path.join(data, "movie_content.csv"),
    )}
    # per-request lookups are precomputed: rebuilding them for 62k movies on
    # every call dominated latency (~150ms) without changing any result.
    st["genre_map"] = {m: set(v["genres"]) for m, v in st["movies"].items()}
    # pre-lowercased search haystacks keep /search at a few ms per query
    st["search_docs"] = [
        (mid, (v["title"] + " " + " ".join(v["genres"]) + " " + v.get("top_tags", "")).lower())
        for mid, v in st["movies"].items()
    ]
    # item popularity from item_stats (first three numeric columns)
    rows = []
    with open(os.path.join(data, "item_stats.csv"), encoding="utf-8") as fh:
        next(fh)
        for line in fh:
            p = line.split(",")
            if len(p) >= 3:
                rows.append((int(p[0]), int(p[1]), float(p[2])))
    pop = pd.DataFrame(rows, columns=["movie_id", "n", "avg"]).sort_values("n", ascending=False)
    st["pop_order"] = pop.movie_id.values
    st["pop_list"] = [int(i) for i in pop.movie_id.values]
    st["pop_n"] = dict(zip(pop.movie_id.astype(int), pop.n))
    st["pop_avg"] = dict(zip(pop.movie_id.astype(int), pop.avg))

    # ItemCF adjacency (symmetric, sim = co / sqrt(n_i n_j))
    co = pd.read_csv(os.path.join(data, "cooccur.csv"))
    adj = defaultdict(list)
    for i, j, c in co.itertuples(index=False):
        s = c / np.sqrt(st["pop_n"].get(int(i), 1) * st["pop_n"].get(int(j), 1))
        adj[int(i)].append((int(j), s))
        adj[int(j)].append((int(i), s))
    for k in adj:
        adj[k] = sorted(adj[k], key=lambda x: -x[1])[:100]
    st["itemcf"] = adj

    # user history (for exclusions + reasons)
    st["hist"] = {}
    with open(os.path.join(data, "seqs.csv"), encoding="utf-8") as fh:
        next(fh)
        for line in fh:
            uid, _, rest = line.partition(",")
            if rest:
                st["hist"][int(uid)] = [int(x) for x in rest.split()]

    # ALS factors
    uf_path = os.path.join(data, "user_factors.csv")
    if os.path.exists(uf_path):
        uf = pd.read_csv(uf_path)
        itf = pd.read_csv(os.path.join(data, "item_factors.csv"))
        fcols = [c for c in uf.columns if c != "user_id"]
        st["als_u"] = {int(u): v for u, v in zip(uf.user_id, uf[fcols].values)}
        st["als_i_ids"] = itf.movie_id.values.astype(int)
        st["als_I"] = itf[[c for c in itf.columns if c != "movie_id"]].values

    # SASRec embeddings (final-layer item vectors, full-rank recall) — v2 = 续训最优检查点
    emb_path = os.path.join(ROOT, "artifacts", "sasrec_v2", "item_emb.npy")
    if os.path.exists(emb_path):
        st["sasrec_ids"] = np.loadtxt(os.path.join(ROOT, "artifacts", "sasrec_v2", "item_ids.txt"), dtype=np.int64)
        st["sasrec_emb"] = np.load(emb_path)
        st["sasrec_pos"] = {int(i): r for r, i in enumerate(st["sasrec_ids"])}

    st["genrec_metrics"] = None
    gm = os.path.join(ROOT, "artifacts", "genrec", "genrec_test_metrics.json")
    if os.path.exists(gm):
        with open(gm, encoding="utf-8") as fh:
            st["genrec_metrics"] = json.load(fh)
    return st


def recall_channels(user_id, k=30):
    st = STATE
    hist = st["hist"].get(user_id, [])
    seen = set(hist)
    channels = {}

    # popularity
    pop = [i for i in st["pop_list"] if i not in seen][:k]
    channels["热门"] = pop

    # ItemCF
    scores = defaultdict(float)
    for i in hist:
        for j, s in st["itemcf"].get(i, ()):  # co-visitation similarity
            if j not in seen:
                scores[j] += s
    itemcf = [i for i, _ in sorted(scores.items(), key=lambda x: -x[1])[:k]]
    channels["协同过滤"] = itemcf

    # ALS
    if "als_u" in st and user_id in st["als_u"]:
        v = st["als_u"][user_id]
        sc = st["als_I"] @ v
        sc[np.isin(st["als_i_ids"], list(seen))] = -np.inf
        idx = np.argpartition(-sc, k)[:k]
        idx = idx[np.argsort(-sc[idx])]
        channels["矩阵分解"] = [int(st["als_i_ids"][r]) for r in idx if sc[r] > -np.inf][:k]

    # SASRec vector recall
    if "sasrec_emb" in st:
        last = hist[-50:]
        if last:
            ids, emb = st["sasrec_ids"], st["sasrec_emb"]
            pos = st["sasrec_pos"]
            # mean of last-5 item embeddings as the user query vector
            rows = [pos[i] for i in last[-5:] if i in pos]
            if rows:
                q = emb[rows].mean(axis=0)
                sc = emb @ q
                sc[np.isin(ids, list(seen))] = -np.inf
                idx = np.argpartition(-sc, k)[:k]
                idx = idx[np.argsort(-sc[idx])]
                channels["序列召回"] = [int(ids[r]) for r in idx if sc[r] > -np.inf][:k]
    return channels


def mmr_diversify(cands, genres_map, lam=0.6, k=20):
    """MMR with genre-overlap diversity; cands: list of (movie_id, base_score)."""
    selected, remaining = [], list(cands)
    while remaining and len(selected) < k:
        best, best_val = None, -1e9
        for mid, base in remaining:
            gs = set(genres_map.get(mid, set()))
            div = 0.0
            if selected:
                overlap = [len(gs & set(genres_map.get(m, set()))) / max(1, len(gs)) for m, _ in selected]
                div = max(overlap)
            val = lam * base - (1 - lam) * div
            if val > best_val:
                best, best_val = (mid, base), val
        selected.append(best)
        remaining.remove(best)
    return selected


@app.get("/health")
def health():
    channels = ["热门"]
    if "itemcf" in STATE:
        channels.append("协同过滤")
    if "als_u" in STATE:
        channels.append("矩阵分解")
    if "sasrec_emb" in STATE:
        channels.append("序列召回")
    return {
        "channels": channels,
        "genrec_metrics": STATE.get("genrec_metrics"),
        "ranker": {
            "name": "DeepFM",
            "artifact_available": artifacts_available(ROOT),
            "loaded": _RANKER["loaded"],
            "active": _RANKER["object"] is not None,
            "error": _RANKER["error"],
        },
    }


@app.get("/recommend")
def recommend(
    user_id: int,
    k: int = 12,
    lam: float = 0.6,
    rank_mode: str = Query("fusion", pattern="^(fusion|deepfm)$"),
):
    """rank_mode=fusion keeps RRF rank order; deepfm enables the CTR reranker.

    DeepFM is off by default because the online replay showed a relevant
    regression: R@10 0.0745 -> 0.0440 on 2,000 held-out users. It remains
    available for demonstration and further calibration.
    """
    t0 = time.time()
    chans = recall_channels(user_id, k=30)
    base = max(len(max(chans.values(), key=len)) if chans else 1, 1)
    cands = {}
    for name, ids in chans.items():
        for rank, mid in enumerate(ids):
            cands.setdefault(mid, {"score": 0.0, "from": []})
            cands[mid]["score"] += (base - rank) / base
            cands[mid]["from"].append(name)
    recall_scores = {m: c["score"] for m, c in cands.items()}
    pairs = sorted(recall_scores.items(), key=lambda x: -x[1])
    ctr_scores = {}
    if rank_mode == "deepfm" and pairs:
        ranker = get_ranker()
        if ranker is not None:
            try:
                ctr_scores = ranker.score(user_id, [m for m, _ in pairs], STATE)
            except Exception as exc:  # noqa: BLE001 - never break recommendations
                print(f"[serving] DeepFM scoring failed, using fusion scores: {exc}", flush=True)
    if ctr_scores:
        # DeepFM is the relevance score entering MMR; the recall rank score is
        # retained only as tie-break / explainability metadata.
        pairs = sorted(
            ((m, ctr_scores[m]) for m, _ in pairs),
            key=lambda x: (-x[1], -recall_scores[x[0]]),
        )

    final = mmr_diversify(pairs, STATE["genre_map"], lam=lam, k=k)
    out = []
    for mid, score in final:
        mv = STATE["movies"].get(mid, {})
        out.append({
            "movie_id": mid, "title": mv.get("title"), "year": mv.get("year"),
            "genres": mv.get("genres"), "score": round(score, 4),
            "recall_score": round(recall_scores[mid], 4),
            "ctr": round(ctr_scores[mid], 4) if mid in ctr_scores else None,
            "channels": cands[mid]["from"],
            "n_ratings": STATE["pop_n"].get(mid),
            "avg_rating": STATE["pop_avg"].get(mid),
        })
    return {
        "user_id": user_id,
        "requested_ranker": rank_mode,
        "ranker": "deepfm" if ctr_scores else "fusion",
        "ranker_note": (
            "DeepFM is optional: online replay showed a relevance regression"
            if rank_mode == "deepfm" else
            "fusion rank order (DeepFM available via rank_mode=deepfm)"
        ),
        "latency_ms": round((time.time() - t0) * 1000, 1),
        "items": out,
    }


# 中文类型同义词 → MovieLens 英文类型（语料全英文，中文查询先展开）
ZH_GENRES = {"科幻": "sci-fi", "爱情": "romance", "喜剧": "comedy", "动作": "action",
             "动画": "animation", "恐怖": "horror", "惊悚": "thriller", "剧情": "drama",
             "战争": "war", "奇幻": "fantasy", "悬疑": "mystery", "犯罪": "crime",
             "冒险": "adventure", "纪录": "documentary", "音乐": "musical",
             "西部": "western", "儿童": "children", "黑色电影": "film-noir"}


@app.get("/search")
def search(q: str, k: int = 20):
    """Lexical search over title + genres + genome tags (no embedding index).

    Chinese genre words are expanded to their English equivalents first, then
    every whitespace-separated token must appear in the movie's haystack.
    Results are ordered by rating count so the page shows recognisable titles.
    """
    ql = q.lower().strip()
    for zh, en in ZH_GENRES.items():
        if zh in ql:
            ql = ql.replace(zh, en)
    tokens = [t for t in ql.split() if t]
    if not tokens:
        return {"query": q, "items": []}
    out = []
    single = tokens[0] if len(tokens) == 1 else None
    for mid, hay in STATE["search_docs"]:
        if (single in hay) if single is not None else all(t in hay for t in tokens):
            mv = STATE["movies"][mid]
            out.append({"movie_id": mid, "title": mv["title"], "year": mv["year"],
                        "genres": mv["genres"], "n_ratings": STATE["pop_n"].get(mid),
                        "avg_rating": STATE["pop_avg"].get(mid)})
    out.sort(key=lambda r: -(r["n_ratings"] or 0))
    return {"query": q, "items": out[:k]}


@app.get("/history/{user_id}")
def history(user_id: int, k: int = 10):
    hist = STATE["hist"].get(user_id, [])[-k:][::-1]
    return {"user_id": user_id,
            "items": [{"movie_id": m, "title": STATE["movies"].get(m, {}).get("title"),
                       "genres": STATE["movies"].get(m, {}).get("genres")} for m in hist]}


@app.get("/metrics")
def metrics():
    out = {}
    p = os.path.join(ROOT, "experiments", "recall_results.json")
    if os.path.exists(p):
        with open(p, encoding="utf-8") as fh:
            out["recall"] = json.load(fh)
    for name in ("lr", "fm", "deepfm"):
        p = os.path.join(ROOT, "artifacts", "ranker", f"{name}_metrics.json")
        if os.path.exists(p):
            with open(p, encoding="utf-8") as fh:
                out[name] = json.load(fh).get("test")
    if STATE.get("genrec_metrics"):
        out["genrec"] = STATE["genrec_metrics"]
    return out


web_dir = os.path.join(ROOT, "web")
if os.path.isdir(web_dir):
    app.mount("/", StaticFiles(directory=web_dir, html=True), name="web")


STATE = load_state()
