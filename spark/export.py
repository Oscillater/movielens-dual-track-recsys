"""Export host-side training artifacts from Hive tables to single-part CSVs on HDFS.

Run inside master container after etl.py dws:
  spark-submit --master yarn --deploy-mode client --num-executors 1 \
      --executor-memory 2g --executor-cores 2 --driver-memory 1g /opt/etl/export.py

Outputs under /project/export/<name>/ as one CSV part (header included).
Then fetch to the VM with:
  for t in seqs gt movies item_stats cooccur; do
    hdfs dfs -getmerge /project/export/$t /home/bigdata/data/export/$t.csv
  done
"""
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

spark = (
    SparkSession.builder.appName("ml25m-export")
    .enableHiveSupport()
    .getOrCreate()
)
spark.sql("USE movie")

OUT = "/project/export"


def save(df, name):
    df.coalesce(1).write.mode("overwrite").option("header", True).csv(f"{OUT}/{name}")
    print(f"exported {name}")


# user sequences: user_id, "id id id ..." (time-ordered train history)
seqs = spark.table("movie.dws_user_seq").select(
    "user_id",
    F.concat_ws(" ", F.col("seq")).alias("seq"),
)
save(seqs, "seqs")

# evaluation ground truth: user, item, split (val | test)
save(spark.table("movie.dws_gt").select("user_id", "movie_id", "split"), "gt")

# movie metadata for SID content, web demo and features
save(
    spark.table("movie.dwd_movies").select(
        "movie_id", "title", "year", F.concat_ws("|", "genres").alias("genres"),
    ),
    "movies",
)

# item stats (train) used as ranking features and popularity lists
save(
    spark.table("movie.dws_item_stats").select(
        "movie_id", "n_ratings", "avg_rating", "title", "year",
    ),
    "item_stats",
)

# ItemCF co-occurrence pairs (i < j) with co-rating user counts
save(spark.table("movie.dws_cooccur"), "cooccur")

spark.stop()
