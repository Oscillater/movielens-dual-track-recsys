"""Spark MLlib regression: movie popularity forecasting (course "回归预测" module).

Task framing (temporal, leakage-free):
  features = per-movie behaviour in the EARLY window  (2015-01 .. 2017-12)
  target   = log1p(rating count) in the LATER window  (2018-01 .. 2019-11)

Models: GBTRegressor vs LinearRegression (baseline). Metrics: RMSE / MAE / R2.

Run inside master container:
  spark-submit --master yarn --deploy-mode client --num-executors 1 \
      --executor-memory 2g --executor-cores 2 --driver-memory 1g /opt/etl/gbt.py
"""
import json

from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import GBTRegressor, LinearRegression
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

EARLY_END = 1514764800        # 2018-01-01 UTC
spark = (
    SparkSession.builder.appName("ml25m-gbt")
    .enableHiveSupport()
    .getOrCreate()
)
spark.sql("USE movie")
r = spark.table("movie.dwd_ratings").select("movie_id", "rating", "rate_ts")

early = r.filter(F.col("rate_ts") < EARLY_END)
later = r.filter(F.col("rate_ts") >= EARLY_END)

feat = (
    early.groupBy("movie_id")
    .agg(
        F.count("*").alias("f_n_early"),
        F.avg("rating").alias("f_avg_early"),
        F.stddev("rating").alias("f_std_early"),
        F.min("rate_ts").alias("f_first_ts"),
        F.max("rate_ts").alias("f_last_ts"),
        F.countDistinct("rating").alias("f_rating_levels"),
    )
    .withColumn("f_span_days", (F.col("f_last_ts") - F.col("f_first_ts")) / 86400.0)
    .withColumn("f_recency_days", (F.lit(EARLY_END) - F.col("f_last_ts")) / 86400.0)
)
target = later.groupBy("movie_id").agg(F.count("*").alias("y_count"))
data = (
    feat.join(target, "movie_id")
    .withColumn("label", F.log1p(F.col("y_count")))
    .na.fill({"f_std_early": 0.0})
)

cols = ["f_n_early", "f_avg_early", "f_std_early", "f_span_days", "f_recency_days", "f_rating_levels"]
assembler = VectorAssembler(inputCols=cols, outputCol="features")
train, test = data.randomSplit([0.8, 0.2], seed=42)

results = {}
for name, model in [
    ("gbt", GBTRegressor(featuresCol="features", labelCol="label", maxIter=40, maxDepth=5, seed=42)),
    ("linear", LinearRegression(featuresCol="features", labelCol="label", maxIter=50)),
]:
    pipe = Pipeline(stages=[assembler, model]).fit(train)
    pred = pipe.transform(test)
    ev = RegressionEvaluator(labelCol="label", predictionCol="prediction")
    m = {
        "RMSE": round(ev.setMetricName("rmse").evaluate(pred), 4),
        "MAE": round(ev.setMetricName("mae").evaluate(pred), 4),
        "R2": round(ev.setMetricName("r2").evaluate(pred), 4),
    }
    results[name] = m
    print(name, json.dumps(m))

import pyspark.sql.functions as F  # noqa: E402
n_train, n_test = train.count(), test.count()
spark.sparkContext.parallelize([json.dumps({"n_train": n_train, "n_test": n_test, **results})]).coalesce(1).saveAsTextFile("/project/gbt_metrics")
spark.stop()
