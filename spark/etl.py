"""MovieLens 25M -> Hive layered warehouse (ODS -> DWD -> DWS).

Run inside master container:
  spark-submit --master yarn --deploy-mode client --num-executors 1 \
      --executor-memory 2g --executor-cores 2 --driver-memory 1g \
      /opt/etl/etl.py <stage> [POP_MIN] [HIST_CAP]

Stages: ods | dwd | dws | verify

Layering:
  ODS  raw single-column text tables over /project/ods/* (byte-exact source)
  DWD  parsed parquet tables; dwd_ratings carries the unified train/val/test
       split (per user: last event -> test, 2nd last -> val, rest -> train)
  DWS  aggregate tables built from TRAIN split only (no label leakage):
       user/item stats, ItemCF co-occurrence, user sequences, eval ground truth
"""
import re
import sys

from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F

DB = "movie"
POP_MIN = int(sys.argv[2]) if len(sys.argv) > 2 else 200
HIST_CAP = int(sys.argv[3]) if len(sys.argv) > 3 else 30


def get_spark(stage):
    return (
        SparkSession.builder.appName(f"ml25m-etl-{stage}")
        .enableHiveSupport()
        .config("spark.sql.shuffle.partitions", "64")
        .config("spark.sql.warehouse.dir", "/user/hive/warehouse")
        .getOrCreate()
    )


def stage_ods(spark):
    """Raw text tables: ODS keeps the source byte-exact, one line per row."""
    spark.sql(f"CREATE DATABASE IF NOT EXISTS {DB}")
    for name in [
        "ratings", "movies", "tags", "links", "genome_scores", "genome_tags",
    ]:
        spark.sql(
            f"DROP TABLE IF EXISTS {DB}.ods_{name}"
        )
        spark.sql(
            f"""
            CREATE EXTERNAL TABLE {DB}.ods_{name} (line STRING)
            ROW FORMAT DELIMITED FIELDS TERMINATED BY '\001'
            STORED AS TEXTFILE
            LOCATION '/project/ods/{name}'
            """
        )
        spark.sql(f"SELECT count(*) AS n_{name} FROM {DB}.ods_{name}").show()


def stage_dwd(spark):
    spark.sql(f"CREATE DATABASE IF NOT EXISTS {DB}")
    spark.sql(f"USE {DB}")

    # ---- dwd_ratings: parse + unified leave-last-out split ----
    raw = spark.table(f"{DB}.ods_ratings").filter(F.col("line") != "userId,movieId,rating,timestamp")
    parsed = raw.select(
        F.split(F.col("line"), ",").getItem(0).cast("int").alias("user_id"),
        F.split(F.col("line"), ",").getItem(1).cast("int").alias("movie_id"),
        F.split(F.col("line"), ",").getItem(2).cast("float").alias("rating"),
        F.split(F.col("line"), ",").getItem(3).cast("long").alias("rate_ts"),
    ).filter(F.col("user_id").isNotNull())
    w = Window.partitionBy("user_id").orderBy(F.col("rate_ts").desc(), F.col("movie_id").desc())
    dwd = parsed.withColumn("rn", F.row_number().over(w)).withColumn(
        "split",
        F.when(F.col("rn") == 1, "test").when(F.col("rn") == 2, "val").otherwise("train"),
    ).drop("rn")
    dwd.write.mode("overwrite").partitionBy("split").format("parquet").saveAsTable(f"{DB}.dwd_ratings")

    # ---- dwd_movies: title/year/genres; title may contain commas ----
    mv = spark.table(f"{DB}.ods_movies").filter(F.col("line") != "movieId,title,genres")
    parts = mv.select(
        F.regexp_extract(F.col("line"), r"^(\d+),", 1).cast("int").alias("movie_id"),
        F.regexp_extract(F.col("line"), r"^\d+,(.+),[^,]*$", 1).alias("title"),
        F.regexp_extract(F.col("line"), r",([^,]*)$", 1).alias("genres_raw"),
    ).filter(F.col("movie_id").isNotNull())
    dwd_movies = parts.select(
        "movie_id",
        "title",
        F.regexp_extract(F.col("title"), r"\((\d{4})\)\s*$", 1).cast("int").alias("year"),
        F.filter(
            F.split(F.col("genres_raw"), r"\|"),
            lambda g: g != "(no genres listed)",
        ).alias("genres"),
    )
    dwd_movies.write.mode("overwrite").format("parquet").saveAsTable(f"{DB}.dwd_movies")

    # ---- dwd_tags: tag text may contain commas -> regex ----
    tg = spark.table(f"{DB}.ods_tags").filter(F.col("line") != "userId,movieId,tag,timestamp")
    dwd_tags = tg.select(
        F.regexp_extract(F.col("line"), r"^(\d+),", 1).cast("int").alias("user_id"),
        F.regexp_extract(F.col("line"), r"^\d+,\d+,(.*),\d+\s*$", 1).alias("tag"),
        F.regexp_extract(F.col("line"), r",(\d+)\s*$", 1).cast("long").alias("tag_ts"),
    ).filter(F.col("user_id").isNotNull())
    dwd_tags.write.mode("overwrite").format("parquet").saveAsTable(f"{DB}.dwd_tags")


def stage_dws(spark, pop_min=POP_MIN, hist_cap=HIST_CAP):
    spark.sql(f"USE {DB}")
    train = spark.table(f"{DB}.dwd_ratings").filter(F.col("split") == "train")

    # ---- dws_user_stats ----
    (train.groupBy("user_id")
     .agg(F.count("*").alias("n_ratings"), F.round(F.avg("rating"), 3).alias("avg_rating"),
          F.min("rate_ts").alias("first_ts"), F.max("rate_ts").alias("last_ts"))
     .write.mode("overwrite").format("parquet").saveAsTable(f"{DB}.dws_user_stats"))

    # ---- dws_item_stats (train counts + movie meta) ----
    (train.groupBy("movie_id")
     .agg(F.count("*").alias("n_ratings"), F.round(F.avg("rating"), 3).alias("avg_rating"))
     .join(spark.table(f"{DB}.dwd_movies"), "movie_id", "left")
     .write.mode("overwrite").format("parquet").saveAsTable(f"{DB}.dws_item_stats"))

    # ---- dws_gt: evaluation ground truth (val + test) ----
    (spark.table(f"{DB}.dwd_ratings").filter(F.col("split") != "train")
     .select("user_id", "movie_id", "rating", "split")
     .write.mode("overwrite").format("parquet").saveAsTable(f"{DB}.dws_gt"))

    # ---- dws_user_seq: time-ordered train history per user ----
    seqs = (train
            .groupBy("user_id")
            .agg(F.sort_array(F.collect_list(F.struct("rate_ts", "movie_id"))).alias("events")))
    seqs = seqs.select("user_id", F.transform("events", lambda e: e["movie_id"]).alias("seq"))
    seqs.write.mode("overwrite").format("parquet").saveAsTable(f"{DB}.dws_user_seq")

    # ---- dws_cooccur: ItemCF pairs, contribution-limited ----
    pop = train.groupBy("movie_id").count().filter(F.col("count") >= pop_min).select("movie_id")
    hist = (train.join(pop, "movie_id")
            .select("user_id", "movie_id", "rate_ts")
            .withColumn("rk", F.row_number().over(
                Window.partitionBy("user_id").orderBy(F.col("rate_ts").desc(), F.col("movie_id").desc())))
            .filter(F.col("rk") <= hist_cap).drop("rk"))
    pairs = (hist.groupBy("user_id")
             .agg(F.collect_set("movie_id").alias("items")))
    # emit ordered pairs i<j from each user's item set
    co = pairs.rdd.flatMap(
        lambda row: ((min(a, b), max(a, b)) for idx, a in enumerate(row["items"]) for b in row["items"][idx + 1:])
    ).toDF(["i", "j"]).groupBy("i", "j").count().filter(F.col("count") >= 5)
    top = co.withColumn("rk", F.row_number().over(
        Window.partitionBy("i").orderBy(F.col("count").desc(), F.col("j"))))
    top.filter(F.col("rk") <= 50).select("i", "j", "count").write.mode("overwrite").format("parquet").saveAsTable(f"{DB}.dws_cooccur")


def stage_verify(spark):
    spark.sql(f"USE {DB}")
    tables = [
        "ods_ratings", "ods_movies", "ods_tags", "ods_links",
        "ods_genome_scores", "ods_genome_tags",
        "dwd_ratings", "dwd_movies", "dwd_tags",
        "dws_user_stats", "dws_item_stats", "dws_gt", "dws_user_seq", "dws_cooccur",
    ]
    for t in tables:
        try:
            n = spark.table(f"{DB}.{t}").count()
            print(f"[verify] {DB}.{t}: {n}")
        except Exception as exc:  # noqa: BLE001 - verify must keep going
            print(f"[verify] {DB}.{t}: FAILED {exc}")
    print("[verify] split distribution:")
    spark.table(f"{DB}.dwd_ratings").groupBy("split").count().orderBy("split").show()
    print("[verify] sample seq:")
    spark.table(f"{DB}.dws_user_seq").limit(2).show(truncate=120)


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "verify"
    spark = get_spark(stage)
    if stage == "ods":
        stage_ods(spark)
    elif stage == "dwd":
        stage_dwd(spark)
    elif stage == "dws":
        stage_dws(spark)
    elif stage == "verify":
        stage_verify(spark)
    else:
        raise SystemExit(f"unknown stage: {stage}")
    spark.stop()
