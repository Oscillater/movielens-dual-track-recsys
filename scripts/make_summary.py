"""Assemble experiments/summary.md from all result artifacts (idempotent)."""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "experiments", "summary.md")

lines = ["# 实验结果汇总（自动生成）", ""]

rec_path = os.path.join(ROOT, "experiments", "recall_results.json")
if os.path.exists(rec_path):
    with open(rec_path, encoding="utf-8") as fh:
        rec = json.load(fh)
    zh = {"pop": "热门", "itemcf": "ItemCF", "als": "ALS(隐式)", "sasrec": "SASRec",
          "genrec": "GenRec(生成式)", "fusion": "多路融合(RRF)"}
    lines += ["## 召回通道统一评测（测试集 = 每用户 leave-last-out，排除已看）", "",
              "| 通道 | Recall@10 | Recall@50 | NDCG@10 | NDCG@50 | 覆盖率 | 类型多样性@50 | 延迟ms/人 |",
              "|---|---|---|---|---|---|---|---|"]
    order = ["pop", "itemcf", "als", "sasrec", "genrec", "fusion"]
    gen_p = os.path.join(ROOT, "artifacts", "genrec", "genrec_test_metrics.json")
    if "genrec" not in rec and os.path.exists(gen_p):
        with open(gen_p, encoding="utf-8") as fh:
            g = json.load(fh)
        rec["genrec"] = {
            "Recall@10": g["Recall@10"], "Recall@50": g["Recall@50"],
            "NDCG@10": g["NDCG@10"], "NDCG@50": g["NDCG@50"],
            "coverage": "-", "genre_diversity@50": "-",
            "latency_ms_per_user": g["latency_ms_per_user"],
        }
    for ch in order + [c for c in rec if c not in order]:
        if ch not in rec:
            continue
        v = rec[ch]
        lines.append(
            f"| {zh.get(ch, ch)} | {v['Recall@10']} | {v['Recall@50']} | {v['NDCG@10']} "
            f"| {v['NDCG@50']} | {v['coverage']} | {v['genre_diversity@50']} | {v['latency_ms_per_user']} |")
    lines.append("")

ranker = []
for name in ("lr", "fm", "deepfm"):
    p = os.path.join(ROOT, "artifacts", "ranker", f"{name}_metrics.json")
    if os.path.exists(p):
        with open(p, encoding="utf-8") as fh:
            m = json.load(fh)
        ranker.append((name.upper(), m["test"]["AUC"], m["test"]["LogLoss"], m.get("best_val_auc")))
if ranker:
    lines += ["## 精排 CTR 模型（DeepFM / FM / LR，同一特征管线）", "",
              "| 模型 | Test AUC | Test LogLoss | Best Val AUC |", "|---|---|---|---|"]
    for n, a, ll, bva in ranker:
        lines.append(f"| {n} | {a} | {ll} | {round(bva, 4) if bva else '-'} |")
    lines.append("")

gbt_p = os.path.join(ROOT, "artifacts", "gbt_metrics.json")
gbt_hdfs = os.path.join(ROOT, "artifacts", "gbt_metrics_hdfs.json")
for p in (gbt_p, gbt_hdfs):
    if os.path.exists(p):
        with open(p, encoding="utf-8") as fh:
            g = json.load(fh)
        lines += ["## 回归预测（电影热度，时间切分防泄漏）", "",
                  f"- 训练/测试: {g.get('n_train')} / {g.get('n_test')}",
                  f"- GBTRegressor: RMSE {g['gbt']['RMSE']} / MAE {g['gbt']['MAE']} / R² {g['gbt']['R2']}",
                  f"- LinearRegression: RMSE {g['linear']['RMSE']} / MAE {g['linear']['MAE']} / R² {g['linear']['R2']}", ""]
        break

sc_path = os.path.join(ROOT, "artifacts", "scaling.json")
if os.path.exists(sc_path):
    with open(sc_path, encoding="utf-8") as fh:
        sc = json.load(fh)
    lines += ["## Spark 数据量扩展性（固定工作负载）", "",
              "| 数据量 | 运行时间(s) |", "|---|---|"]
    for r in sc:
        lines.append(f"| {int(r['fraction']*100)}% | {r['runtime_s']} |")
    lines.append("")

sid_p = os.path.join(ROOT, "artifacts", "sid", "sid_stats.json")
if os.path.exists(sid_p):
    with open(sid_p, encoding="utf-8") as fh:
        s = json.load(fh)
    lines += ["## 语义 ID 构建（RQ-KMeans）", "",
              f"- 电影数 {s['n_movies']}，{s['levels']} 层 × {s['codes_per_level']} 码字",
              f"- 每层码字使用率: {[round(u / s['codes_per_level'] * 100, 1) for u in s['per_level_codebook_usage']]}%",
              f"- 碰撞率: {s['collision_rate']}", ""]

with open(OUT, "w", encoding="utf-8") as fh:
    fh.write("\n".join(lines))
print("wrote", OUT)
