"""Data-scaling experiment: fixed workload over growing input fractions.

Workload (same shape as the ETL core): parse ratings -> per-user aggregation
-> join movie metadata -> per-genre aggregation, materialised with count().
Timing covers the full Spark job per fraction.

Run inside master container:
  spark-submit --master yarn --deploy-mode client --num-executors 1 \
      --executor-memory 2g --executor-cores 2 --driver-memory 1g /opt/etl/scaling.py
"""
import json
import time

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

spark = (
    SparkSession.builder.appName("ml25m-scaling")
    .enableHiveSupport()
    .config("spark.sql.shuffle.partitions", "64")
    .getOrCreate()
)
spark.sql("USE movie")

ratings = spark.table("movie.dwd_ratings").filter("split = 'train'")
movies = spark.table("movie.dwd_movies")

results = []
for frac in [0.25, 0.5, 1.0]:
    sample = ratings.sample(fraction=frac, seed=42)
    t0 = time.time()
    n = (
        sample.join(movies.select("movie_id", "genres"), "movie_id")
        .select("user_id", "rating", F.explode("genres").alias("genre"))
        .groupBy("genre")
        .agg(F.count("*").alias("n"), F.avg("rating").alias("avg"))
        .count()
    )
    dt = round(time.time() - t0, 1)
    results.append({"fraction": frac, "runtime_s": dt, "genre_rows": n})
    print(json.dumps(results[-1]), flush=True)

spark.sparkContext.parallelize([json.dumps(results)]).coalesce(1).saveAsTextFile("/project/scaling")
spark.stop()
