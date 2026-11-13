# 基于 Spark 的电影搜索与推荐系统

课程项目：大数据架构的数据处理与分析系统（独立完成）

- 实验结果汇总：[experiments/summary.md](experiments/summary.md)
- 项目汇报材料：[report/slides.pdf](report/slides.pdf)

## 摘要

本项目在自建三节点 Hadoop/Hive/Spark 集群上基于 MovieLens 25M 数据集（2500 万条评分记录）构建端到端电影搜索与推荐系统。系统包含两条技术路线：（1）**判别式推荐管线**，涵盖多路召回（ALS、ItemCF、SASRec 与热门召回）、排名融合、精排以及最大边缘相关（MMR）多样性重排；（2）**生成式推荐支线**，借鉴 TIGER 与 OneRec 范式，采用残差量化 K-Means（RQ-KMeans）构建层次化语义 ID，并基于小型因果 Transformer 与前缀树（Trie）约束束搜索实现自回归候选生成。两条路线在统一的留一法（leave-last-out）数据切分与全量候选集上开展离线对比评估，并通过 FastAPI 与 ECharts 提供在线检索与推荐服务。

### 方法与实现说明

- **全量精确召回**：离线评测与在线召回均对 62,423 部候选电影执行精确打分（NumPy 矩阵内积或共现表检索），未引入近似最近邻（ANN）索引；FAISS 仅用于语义 ID 构建阶段的残差向量量化（RQ-KMeans）。
- **在线排序策略**：在线服务默认采用通道归一化倒数排名融合（Normalized RRF），使召回分与 MMR 多样性惩罚项保持同量纲；DeepFM 精排作为可选分支（`rank_mode=deepfm`）提供。2,000 名用户的在线管线重放表明，直接以 pointwise CTR 预估替换多路融合排序会导致 Recall@10 由 0.0745 降至 0.0440，故默认关闭。

## 核心实验结果

| 评测维度 | 实验配置与指标结果 |
|---|---|
| 评测协议 | 162,541 名测试用户，留一法（leave-last-out）切分，过滤历史交互，62,423 部候选电影全量排序 |
| 最优单路召回 | 隐式反馈 ALS：Recall@50 = **0.2405**，目录覆盖率 = 0.1535 |
| 多路召回融合 | 加权 RRF：Recall@10 = **0.0739**，NDCG@10 = **0.0371**，Recall@50 = 0.2239 |
| 序列召回迭代 | SASRec v2：Recall@50 = **0.1513**（较 v1 基线 0.1379 相对提升 9.7%） |
| 通道消融分析 | 移除 ALS：Recall@50 下降 20.6%，覆盖率下降 54.0%；移除 ItemCF：Recall@50 下降 6.1% |
| 精排 CTR 预估 | 同一特征管线下：LR AUC = 0.7573，DeepFM AUC = 0.7552，FM AUC = 0.7463 |
| 热度回归预测 | GBT：$R^2 = \mathbf{0.8917}$，RMSE = 0.5175；线性回归：$R^2 = 0.6551$ |
| 语义 ID 量化 | 62,423 部电影，$3 \times 256$ 层次码本，各层码本利用率 100%，叶节点碰撞率 3.35% |
| 生成式推荐 | GenRec（20,000 名测试用户子集）：Recall@50 = 0.0046，平均解码延迟 150 ms/用户 |
| 在线策略重放 | 融合排序 R@10 = 0.0745；融合 + MMR R@10 = 0.0760（类型多样性 14.13 → 15.22）；DeepFM R@10 = 0.0440 |
| 在线服务延迟 | 本机 500 次请求基准测试：`/recommend` p50 = 31 ms、p95 = 62 ms；`/search` p50 = 26 ms |

> 注：受限于 CPU 端自回归束搜索的计算开销，生成式推荐通道在 20,000 名随机抽样用户上完成评估，其指标不宜与全量测试集（162,541 名用户）上的判别式通道直接等同对比。

## 实验复现指南

### 运行环境与依赖

- **分布式集群**：基于 VMware 虚拟机的三节点 Docker 集群，运行 Hadoop 3.3.6、Hive 4.0.1 与 Spark 3.5.9（YARN 模式）。
- **宿主机环境**：Python 3.11 与 [uv](https://docs.astral.sh/uv/) 包管理器；依赖版本锁定于 `pyproject.toml` 与 `uv.lock`。

```bash
uv sync  # 创建虚拟环境并安装锁定依赖
```

分布式作业通过集群内置的 `spark-submit` 提交，宿主机环境无需安装 PySpark。

### 离线数据处理与模型训练

1. 下载 MovieLens 25M 数据集并上传至 HDFS（`~/data/ml-25m.zip`）。
2. 数据仓库构建（`spark/etl.py`）：完成 ODS 原始层加载、DWD 明细层清洗与留一法切分，以及 DWS 汇总指标计算。
3. 分布式特征与模型计算：依次执行 `spark/als.py`（显式与隐式矩阵分解）、`spark/samples.py`（精排样本构造与内容特征提取）、`spark/gbt.py`（热度回归）与 `spark/export.py`（特征与模型产物导出）。
4. 深度模型与生成式支线训练：依次执行 `models/sasrec.py`、`models/deepfm.py`（含 LR/FM/DeepFM 对照）、`sid/build_sid.py`（内容编码与 RQ-KMeans 量化）与 `sid/genrec.py`（自回归生成模型）。
5. 统一离线评测：运行 `models/evaluate.py` 生成 `experiments/recall_results.json`，随后执行 `scripts/make_summary.py` 导出 `experiments/summary.md`。

注：`data/` 与 `artifacts/` 目录体积较大，可通过上述脚本完整重建，故未纳入版本控制。

### 在线检索与推荐服务

启动 FastAPI 服务端：

```bash
uv run python -m uvicorn serving.app:app --port 8000
```

服务启动后可通过浏览器访问 `http://127.0.0.1:8000`。核心接口如下：

- `/recommend?user_id=1&k=12&rank_mode=fusion|deepfm`：执行多路召回、排名融合（或精排）与 MMR 重排。
- `/search?q=科幻`：基于标题、类型与 Genome 标签的倒排检索（支持中文类型词向英文词项的自动映射）。
- `/history/{user_id}`、`/metrics`、`/health`：用户历史交互查询、系统评测指标看板与服务健康检查。

### 在线排序策略离线重放

```bash
uv run python scripts/eval_serving_ranker.py --users 2000
```

评测结果输出至 `experiments/serving_ranker_eval.json`，用于对比不同在线排序与重排策略的端到端表现。

## 目录结构

```text
project/
  spark/        # Hive 建表、ETL 管线、ALS/ItemCF、样本构造、GBT 回归与扩展性实验
  sid/          # 多模态文本编码、RQ-KMeans 语义 ID 构建、生成式模型与约束解码
  models/       # SASRec 序列召回、LR/FM/DeepFM 精排模型与统一离线评测框架
  serving/      # FastAPI 在线服务：多路召回、排名融合、可选精排与 MMR 多样性重排
  web/          # 前端可视化界面（原生 HTML + ECharts）
  scripts/      # 实验表格生成、在线策略重放与批处理脚本
  experiments/  # 离线与在线评测结果（JSON/Markdown）及系统运行截图
  report/       # Beamer 演示文稿源码与编译产物
  data/         # 数据集与中间样本（不纳入版本控制）
  artifacts/    # 模型权重与导出特征（不纳入版本控制）
```

## 系统设计依据与技术选型

| 维度 | 选型方案 | 设计依据 |
|---|---|---|
| 数据集 | MovieLens 25M | 兼顾数据规模（2500 万交互）与单机存储约束（解压后约 530 MB），包含丰富的类型与 Genome 标签文本，足以支撑协同过滤、序列建模、内容量化与全文检索。 |
| 架构设计 | 判别式多级漏斗 + 生成式推荐对照 | 完整覆盖工业界经典多级搜推链路，并在相同实验基准下引入前沿生成式检索范式，便于开展跨范式性能与效率对比。 |
| 生成式方案 | RQ-KMeans 语义 ID + 轻量级因果 Transformer + 约束束搜索 | 遵循 TIGER 与 MiniOneRec 的核心检索架构，保留离散语义量化与前缀树约束解码机制，同时规避大规模强化学习阶段的高昂算力需求。 |
| 训练算力配置 | 宿主机 CPU（Intel i5-12500H，12 核 16 线程） | 针对无独立 GPU 的硬件约束，将精排训练样本控制在 500 万规模、嵌入维度设为 8–16 维，使各深度模型可在数十分钟至数小时内收敛。 |
| 前端展示 | FastAPI + 静态 HTML + ECharts | 采用轻量级前后端集成方案，支持实时搜推交互与离线实验指标的可视化分析。 |
| 大数据集群 | 三节点 Docker 集群（8 GB 虚拟机内存） | 复用并扩容实验二/三的容器化集群（Hadoop 3.3.6 / Hive 4.0.1 / MySQL 8.0.43），承载分布式数仓 ETL 与 Spark MLlib 训练任务。 |

## 参考文献与开源资源

- 判别式推荐参考架构：Datawhale Fun-Rec（<https://datawhalechina.github.io/fun-rec/>）
- 生成式推荐框架：MiniOneRec（<https://github.com/AkaliKong/MiniOneRec>，论文：<https://arxiv.org/abs/2510.24431>）
- 生成式检索基础模型：Rajput et al., *Recommender Systems with Generative Retrieval (TIGER)*, NeurIPS 2023（<https://arxiv.org/abs/2305.05065>）
- 相关开源复现：`phonism/genrec`、`EdoardoBotta/RQ-VAE-Recommender`
- 实验数据集：GroupLens MovieLens 25M（<https://grouplens.org/datasets/movielens/25m/>）

