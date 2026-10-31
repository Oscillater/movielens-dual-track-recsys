"""ALS collaborative filtering on the unified train split (Spark MLlib).

Run inside master container after etl.py dws:
  spark-submit --master yarn --deploy-mode client --num-executors 1 \
      --executor-memory 2g --executor-cores 2 --driver-memory 1g \
      /opt/etl/als.py [rank] [regParam] [maxIter]

Outputs (HDFS, /project/als/):
  user_factors.csv  user_id, f0..f{rank-1}
  item_factors.csv  movie_id, f0..f{rank-1}
  als_metrics.json  RMSE on val split (explicit feedback)
"""
import json
import sys

from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

RANK = int(sys.argv[1]) if len(sys.argv) > 1 else 64
REG = float(sys.argv[2]) if len(sys.argv) > 2 else 0.1
MAX_ITER = int(sys.argv[3]) if len(sys.argv) > 3 else 12

spark = (
    SparkSession.builder.appName("ml25m-als")
    .enableHiveSupport()
    .config("spark.sql.shuffle.partitions", "64")
    .getOrCreate()
)
spark.sql("USE movie")
train = spark.table("movie.dwd_ratings").filter("split = 'train'")
val = spark.table("movie.dwd_ratings").filter("split = 'val'")

als = ALS(
    userCol="user_id", itemCol="movie_id", ratingCol="rating",
    rank=RANK, regParam=REG, maxIter=MAX_ITER,
    coldStartStrategy="drop", nonnegative=False, seed=42,
)
model = als.fit(train)

preds = model.transform(val)
evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")
rmse = evaluator.evaluate(preds)
print(f"ALS_RMSE val={rmse:.4f} rank={RANK} reg={REG} iter={MAX_ITER}")

uf = model.userFactors.select(
    F.col("id").alias("user_id"),
    *[F.col("features").getItem(i).alias(f"f{i}") for i in range(RANK)],
)
iff = model.itemFactors.select(
    F.col("id").alias("movie_id"),
    *[F.col("features").getItem(i).alias(f"f{i}") for i in range(RANK)],
)
uf.coalesce(1).write.mode("overwrite").option("header", True).csv("/project/als/user_factors")
iff.coalesce(1).write.mode("overwrite").option("header", True).csv("/project/als/item_factors")

metrics = {"rmse_val": rmse, "rank": RANK, "reg": REG, "max_iter": MAX_ITER}
spark.sparkContext.parallelize([json.dumps(metrics)]).coalesce(1).saveAsTextFile("/project/als/metrics")
spark.stop()
