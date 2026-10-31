# 基于 Spark 的电影搜索与推荐系统

课程大作业（大数据架构的数据处理和分析系统），独立完成。施工清单见 [PLAN.md](PLAN.md)。

## 一句话简介

在自建 Hadoop/Hive/Spark 集群上处理 MovieLens 2500 万条评分，实现工业级判别式推荐漏斗（ALS / ItemCF / SASRec 多路召回 → FAISS 检索 → DeepFM 精排 → MMR 重排）与受 OneRec/TIGER 启发的生成式推荐（语义 ID + 小型 Transformer），在统一协议下系统对比两条技术路线，并提供 FastAPI + ECharts 的在线搜推服务。

## 已定决策（为什么是现在这样）

| 决策 | 结论 | 理由 |
|---|---|---|
| 数据集 | MovieLens 25M | 宿主机无独显、D 盘空间紧张；600–700MB 体量 CPU 训得动、HDFS/Hive 装得下，且自带 genres/tags 文本可做内容特征与搜索 |
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

- 宿主机：i5-12500H（12 核 16 线程）/ 16GB / 无独显 / D 盘剩余 5.2GB（清理安装包后约 12.5GB）
- 虚拟机：VMware `ubuntu-bigdata`（Ubuntu 24.04.5，2 vCPU / 6GB，IP 192.168.246.128，开机后需确认），内部 Docker 跑 Hadoop/HBase/Hive/MySQL
- 关键约束：深度模型只能 CPU 训练；数据集工作集控制在 1–2GB；虚拟机磁盘增长会消耗 D 盘空间

## 规划的代码目录（随进度创建）

```
project/
  spark/        # Hive 建表 SQL、ETL、ALS、ItemCF、样本导出
  sid/          # 内容编码、RQ-KMeans、碰撞处理
  models/       # sasrec.py deepfm.py genrec.py + train_*.py
  serving/      # FastAPI、FAISS 索引加载、约束解码
  web/          # index.html + ECharts
  experiments/  # 统一评测、消融、数据量曲线
```
