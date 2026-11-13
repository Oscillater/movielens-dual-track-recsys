# 基于 Spark 的电影搜索与推荐系统

课程大作业（大数据架构的数据处理和分析系统），独立完成。

- 实验数字（自动生成）：[experiments/summary.md](experiments/summary.md)
- 从零技术教程：[docs/tech-tutorial.md](docs/tech-tutorial.md)
- 汇报 slides：[report/slides.pdf](report/slides.pdf)

## 一句话简介

在自建三节点 Hadoop/Hive/Spark 集群上处理 MovieLens 25M（2500 万条评分），实现完整的判别式推荐漏斗：ALS / ItemCF / SASRec / 热门多路召回 → 排名融合 → MMR 多样性重排；同时实现受 TIGER / OneRec 启发的生成式推荐支线（RQ-KMeans 语义 ID + 小型 Transformer + trie 约束解码）。两条路线在同一套指标与数据切分下离线对比（生成式因 CPU 解码成本按 2 万用户抽样），并提供 FastAPI + ECharts 的在线搜推服务。

两条容易误解的工程事实：

- **召回是全量精确排序，没有 FAISS 近似索引**：对 6.2 万候选做 NumPy 矩阵内积 / 共现查表；FAISS 只用于语义 ID 的残余量化（RQ-KMeans）。
- **在线默认排序是融合分，不是 DeepFM**：融合分按通道长度归一化累加排名，便于与 MMR 的相似度项同量纲。DeepFM 精排保留为 `rank_mode=deepfm` 可选分支；2,000 用户在线路径重放显示直接替换融合排序会使 R@10 从 0.0745 掉到 0.0440，因此默认关闭。

## 关键结果

| 项目 | 结果 |
|---|---|
| 评测协议 | 162,541 名测试用户，leave-last-out，排除已看，62,423 部候选全量排序 |
| 最优单通道 | ALS(隐式) Recall@50 **0.2405**，覆盖率 0.1535 |
| 最优融合 | R@10 **0.0739** / NDCG@10 **0.0371** / R@50 0.2239 |
| SASRec v2 | Recall@50 **0.1513**（较 v1 0.1379 提升 9.7%） |
| 消融 | 去掉 ALS：R@50 −20.6%、覆盖率 −54.0%；去掉 ItemCF：R@50 −6.1% |
| 精排 CTR | LR AUC 0.7573 ≥ DeepFM 0.7552 > FM 0.7463（同一特征管线） |
| 回归（热度） | GBT R² **0.8917** / RMSE 0.5175；线性 R² 0.6551 |
| 语义 ID | 62,423 部电影，3 层 × 256 码字，三层使用率 100%，碰撞率 3.35% |
| 生成式 | GenRec（20,000 用户抽样）Recall@50 0.0046，150 ms/人 |
| 在线重放 | 融合 R@10 0.0745；融合+MMR 0.0760（多样性 14.13→15.22）；DeepFM 0.0440 |
| 服务延迟 | 本机 500 次请求：`/recommend` p50 31 ms / p95 62 ms；`/search` p50 26 ms |

> 生成式通道受 CPU 约束解码成本限制只抽样 2 万名用户，与判别式通道的全量数字不可直接比较；summary 表中已标注人数。

## 复现

### 依赖

- 集群：VMware 虚拟机内 Docker 三节点，Hadoop 3.3.6 / Hive 4.0.1 / Spark 3.5.9 on YARN
- 宿主机：Python 3.11（`.venv`），numpy / pandas / torch / faiss-cpu / fastapi / uvicorn

### 数据与离线流程

1. 下载 MovieLens 25M 并上传 HDFS（`~/data/ml-25m.zip`）
2. `spark/etl.py`：ODS → DWD（leave-last-out 切分）→ DWS
3. `spark/als.py`（显式 + 隐式因子）→ `spark/samples.py`（精排样本、内容文本）→ `spark/gbt.py` → `spark/export.py`
4. `models/sasrec.py`、`models/deepfm.py`（LR/FM/DeepFM 共用管线）、`sid/build_sid.py`、`sid/genrec.py`
5. `models/evaluate.py` 产出 `experiments/recall_results.json`，再执行 `scripts/make_summary.py` 生成 `experiments/summary.md`

`data/` 与 `artifacts/` 体积大且可由上述脚本重建，未纳入版本库。

### 在线服务

```bash
.venv/Scripts/python -m uvicorn serving.app:app --port 8000
```

浏览器打开 http://127.0.0.1:8000。接口：

- `/recommend?user_id=1&k=12&rank_mode=fusion|deepfm`：多路召回 → 融合/可选精排 → MMR
- `/search?q=科幻`：标题 + 类型 + genome 标签的词典检索（中文类型词自动展开为英文）
- `/history/{user_id}`、`/metrics`、`/health`

### 精排上线评估

```bash
.venv/Scripts/python scripts/eval_serving_ranker.py --users 2000
```

结果写入 `experiments/serving_ranker_eval.json`，用于决定默认排序策略。

## 目录结构

```
project/
  spark/        # Hive 建表、ETL、ALS/ItemCF、样本导出、GBT、扩展性实验
  sid/          # 内容编码、RQ-KMeans 语义 ID、生成式模型与约束解码
  models/       # SASRec、DeepFM/LR/FM、统一评测框架
  serving/      # FastAPI：多路召回 + 排名融合 +（可选）DeepFM + MMR
  web/          # 单页前端（原生 HTML + ECharts CDN）
  scripts/      # 汇总生成、在线路径重放、过夜接力
  experiments/  # 评测结果 JSON/Markdown 与系统截图
  report/       # Beamer 源文件、编译好的 PDF（页面预览本地生成、不入库）
  docs/         # 从零技术教程
  data/         # 不入库：数据集与训练样本
  artifacts/    # 不入库：模型权重与指标
```

## 已定决策（为什么是现在这样）

| 决策 | 结论 | 理由 |
|---|---|---|
| 数据集 | MovieLens 25M | 宿主机无独显、D 盘空间紧张；解压约 530MB，CPU 训得动、HDFS/Hive 装得下，且自带 genres/tags 文本可做内容特征与搜索 |
| 总体形态 | 判别式漏斗 + 生成式分支对比 | 工作量占 50 分需要全链路；"判别式 vs 生成式"是 2025 年工业界核心议题，对比实验本身就是论文级叙事 |
| 生成式路线 | RQ-KMeans 语义 ID + 从零训练 decoder-only 小 Transformer + 约束解码 | 借 MiniOneRec 的 CPU 可跑部件与评测协议，避开其 4–8 张 A100 的 SFT/RL 阶段；这正是 TIGER 原始设计，社区复现（phonism/genrec、RQ-VAE-Recommender）验证过可行性 |
| 深度学习硬件 | 宿主机 CPU（i5-12500H 12 核） | 无独立显卡（Iris Xe 核显），样本量控制在 ~500 万行、embedding 8–16 维，训练分钟到小时级 |
| 前端 | FastAPI + 单页静态 HTML + ECharts CDN | 不写 Vue/npm 工程链，半天联调；评委看 PDF 截图无法区分，省下的时间投给消融实验 |
| 虚拟机 | 内存 6→8GB，Docker 化集群沿用实验二/三 | Hadoop 3.3.6 三节点 + Hive 4.0.1 + MySQL 8.0.43 已就绪，是"自建 Linux 大数据环境"加分项的实体 |

## 参考资源

- 课程参考：Datawhale fun-rec（判别式漏斗全链路模板）https://datawhalechina.github.io/fun-rec/
- 生成式推荐：MiniOneRec https://github.com/AkaliKong/MiniOneRec ｜ 论文 https://arxiv.org/abs/2510.24431
- TIGER（NeurIPS 2023，生成式检索原始设计）https://arxiv.org/abs/2305.05065
- 社区复现：phonism/genrec、EdoardoBotta/RQ-VAE-Recommender
- 数据集：MovieLens 25M https://grouplens.org/datasets/movielens/25m/

## 本机环境事实（2026-10-31 实测）

- 宿主机：i5-12500H（12 核 16 线程）/ 16GB / 无独显 / D 盘清理后可用 32.3GB
- 虚拟机：VMware `ubuntu-bigdata`（Ubuntu 24.04.5，2 vCPU / 8GB，IP 192.168.246.128），内部 Docker 跑 Hadoop/HBase/Hive/MySQL
- 关键约束：深度模型只能 CPU 训练；数据集工作集控制在 1–2GB；虚拟机磁盘增长会消耗 D 盘空间

## 交付状态

- 已完成：集群环境、数仓、全部模型、统一评测与消融、FastAPI + 网页、PPT/PDF、系统截图、技术教程
- 待用户提供：代码仓库远端地址（GitHub/Gitee）
- 说明：模型权重与数据集不入库，按上方"数据与离线流程"重建；如需直接下载权重可在仓库 Release 中附带 `artifacts/` 压缩包
