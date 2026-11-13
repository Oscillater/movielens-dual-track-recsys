"""DeepFM serving adapter: candidate features -> CTR scores.

The adapter reproduces the small DeepFM forward pass in NumPy, using weights
exported from the PyTorch checkpoint. A 6.4M-parameter model does not justify
loading the whole torch runtime into the demo service; on this machine that
saves roughly 10s of cold-start time and hundreds of MB of RSS.

The online feature values are reconstructed from artifacts available to the
serving process:

  user_n / user_avg   exact train-split aggregates exported from samples.csv
  item_n / item_avg   exact DWS aggregates already loaded by the app
  year                movie metadata
  days_since_prev     per-user mean training gap (candidate future time is
                      unknowable at serving time)
  genre_affinity      estimated from historical item averages grouped by genre
  genre_overlap       user's top-5 genres by history count vs candidate genres

The approximations are explicit because the original training pipeline used
rating-level aggregates that are not part of ``data/seqs.csv``.
"""
import json
import os
from collections import Counter

import numpy as np

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


def artifact_paths(root):
    base = os.path.join(root, "artifacts", "ranker")
    return {
        "weights": os.path.join(base, "deepfm_weights.npz"),
        "preprocess": os.path.join(base, "preprocess.json"),
        "users": os.path.join(base, "user_features.csv"),
    }


def artifacts_available(root):
    paths = artifact_paths(root)
    return all(os.path.exists(p) for p in paths.values())


class DeepFMRanker:
    def __init__(self, root):
        paths = artifact_paths(root)
        missing = [name for name, path in paths.items() if not os.path.exists(path)]
        if missing:
            raise FileNotFoundError(f"DeepFM artifacts missing: {missing}")
        with open(paths["preprocess"], encoding="utf-8") as fh:
            self.preprocess = json.load(fh)

        weights = np.load(paths["weights"])
        self.linear = weights["linear.weight"][:, 0]
        self.emb = weights["emb.weight"]
        self.hidden = self.preprocess["hidden"]
        self.mlp = []
        for i in range(len(self.hidden)):
            base = i * 4
            self.mlp.append(
                {
                    "w": weights[f"mlp.{base}.weight"],
                    "b": weights[f"mlp.{base}.bias"],
                    "gamma": weights[f"mlp.{base + 1}.weight"],
                    "beta": weights[f"mlp.{base + 1}.bias"],
                    "mean": weights[f"mlp.{base + 1}.running_mean"],
                    "var": weights[f"mlp.{base + 1}.running_var"],
                }
            )
        final_base = len(self.hidden) * 4
        self.final_w = weights[f"mlp.{final_base}.weight"]
        self.final_b = weights[f"mlp.{final_base}.bias"]
        if self.linear.shape[0] != self.preprocess["total_dim"]:
            raise ValueError("DeepFM weights and preprocess metadata do not match")

        import pandas as pd

        users = pd.read_csv(paths["users"])
        user_max = int(users.user_id.max()) if len(users) else 0
        self.user_n = np.zeros(user_max + 1, dtype=np.float32)
        self.user_avg = np.full(user_max + 1, np.nan, dtype=np.float32)
        self.user_gap = np.zeros(user_max + 1, dtype=np.float32)
        idx = users.user_id.to_numpy(dtype=np.int64)
        self.user_n[idx] = users.user_n.to_numpy(dtype=np.float32)
        self.user_avg[idx] = users.user_avg.to_numpy(dtype=np.float32)
        self.user_gap[idx] = users.gap_days.fillna(0).to_numpy(dtype=np.float32)
        self.user_avg_global = float(
            np.nanmean(self.user_avg) if np.isfinite(self.user_avg).any() else 3.5
        )

    def _user_basics(self, user_id, hist):
        if 0 <= user_id < len(self.user_n) and np.isfinite(self.user_avg[user_id]):
            return (
                float(self.user_n[user_id]),
                float(self.user_avg[user_id]),
                float(self.user_gap[user_id]),
            )
        return float(len(hist)), self.user_avg_global, 0.0

    def _profile(self, user_id, hist, state):
        """Derive compact genre features from the user's training history."""
        user_n, user_avg, gap_days = self._user_basics(user_id, hist)
        movies = state["movies"]
        item_avg = state.get("pop_avg", {})
        genre_n = Counter()
        genre_sum = Counter()
        for mid in hist:
            mv = movies.get(int(mid))
            if not mv:
                continue
            avg = float(item_avg.get(int(mid), user_avg) or user_avg)
            for genre in mv.get("genres", ()):
                genre_n[genre] += 1
                genre_sum[genre] += avg
        top5 = {g for g, _ in genre_n.most_common(5)}
        genre_avg = {
            g: (genre_sum[g] / genre_n[g] if genre_n[g] else user_avg)
            for g in genre_n
        }
        return {
            "user_n": user_n,
            "user_avg": user_avg,
            "gap_days": gap_days,
            "genre_n": genre_n,
            "genre_avg": genre_avg,
            "top5": top5,
        }

    @staticmethod
    def _year(movie):
        try:
            return int(float(movie.get("year") or 0))
        except (TypeError, ValueError):
            return 0

    @staticmethod
    def _safe_id(value, size):
        value = int(value)
        return value if 0 <= value < size else 0

    def _feature_matrix(self, user_id, movie_ids, state):
        hist = state["hist"].get(user_id, [])
        profile = self._profile(user_id, hist, state)
        movies = state["movies"]
        item_n = state.get("pop_n", {})
        item_avg = state.get("pop_avg", {})
        sizes = self.preprocess["field_sizes"]
        offsets = self.preprocess["offsets"]
        edges = self.preprocess["edges"]

        rows = []
        for mid in movie_ids:
            mid = int(mid)
            mv = movies.get(mid, {})
            genres = list(mv.get("genres", ()))
            gs = [g for g in genres if g in profile["genre_n"]]
            if gs:
                wsum = sum(float(profile["genre_n"][g]) for g in gs)
                affinity = (
                    sum(float(profile["genre_avg"][g]) * float(profile["genre_n"][g]) for g in gs)
                    / wsum
                )
            else:
                affinity = profile["user_avg"]
            row = {
                "user_id": self._safe_id(user_id, sizes["user_id"]),
                "movie_id": self._safe_id(mid, sizes["movie_id"]),
                "year": max(0, min(self._year(mv), sizes["year"] - 1)),
                "user_n": profile["user_n"],
                "user_avg": profile["user_avg"],
                "item_n": float(item_n.get(mid, 0) or 0),
                "item_avg": float(item_avg.get(mid, 0) or profile["user_avg"]),
                "days_since_prev": profile["gap_days"],
                "genre_affinity": affinity,
                "genre_overlap": float(sum(g in profile["top5"] for g in genres)),
            }
            rows.append(row)

        X = np.empty((len(rows), len(self.preprocess["feature_order"])), dtype=np.int64)
        for col, field in enumerate(self.preprocess["feature_order"]):
            if field in SPARSE:
                vals = np.array([r[field] for r in rows], dtype=np.int64)
            else:
                raw = np.array([r[field] for r in rows], dtype=np.float64)
                vals = np.searchsorted(
                    np.asarray(edges[field], dtype=np.float64), raw, side="right"
                ).astype(np.int64)
            X[:, col] = vals + offsets[field]
        return X

    def _forward(self, X):
        """NumPy equivalent of models.deepfm.Ranker(mode='deepfm')."""
        first = self.linear[X].sum(axis=1)
        v = self.emb[X]  # [B, fields, dim]
        second = 0.5 * ((v.sum(axis=1) ** 2) - (v * v).sum(axis=1)).sum(axis=1)

        h = v.reshape(len(X), -1).astype(np.float32)
        for layer in self.mlp:
            h = h @ layer["w"].T + layer["b"]
            h = (
                (h - layer["mean"])
                / np.sqrt(layer["var"] + 1e-5)
                * layer["gamma"]
                + layer["beta"]
            )
            h = np.maximum(h, 0.0)
        deep = (h @ self.final_w.T + self.final_b)[:, 0]
        logits = first + second + deep
        return 1.0 / (1.0 + np.exp(-np.clip(logits, -30.0, 30.0)))

    def score(self, user_id, movie_ids, state):
        """Return {movie_id: probability}. Empty input stays cheap."""
        movie_ids = [int(i) for i in movie_ids]
        if not movie_ids:
            return {}
        X = self._feature_matrix(int(user_id), movie_ids, state)
        probs = self._forward(X)
        return {mid: float(p) for mid, p in zip(movie_ids, probs)}
