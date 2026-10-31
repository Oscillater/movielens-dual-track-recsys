"""Assemble DeepFM/LR/FM training samples and SID content texts on Spark.

Run inside master container after etl.py dws:
  spark-submit --master yarn --deploy-mode client --num-executors 1 \
      --executor-memory 2g --executor-cores 2 --driver-memory 1g \
      /opt/etl/samples.py [TRAIN_FRACTION]

Outputs (HDFS):
  /project/export/samples.csv   label + features + split (train subsampled,
                                val/test in full)
  /project/export/movie_content.csv  movie_id,title,year,genres,top_tags
    (content text base for RQ-KMeans semantic IDs)
"""
import sys

from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F

TRAIN_FRACTION = float(sys.argv[1]) if len(sys.argv) > 1 else 0.16

spark = (
    SparkSession.builder.appName("ml25m-samples")
    .enableHiveSupport()
    .config("spark.sql.shuffle.partitions", "64")
    .getOrCreate()
)
spark.sql("USE movie")

ratings = spark.table("movie.dwd_ratings")
train = ratings.filter("split = 'train'")
users = spark.table("movie.dws_user_stats")
items = spark.table("movie.dws_item_stats")
movies = spark.table("movie.dwd_movies")

# ---- user x genre affinity from train (mean rating & count per genre) ----
profile = (
    train.join(movies.select("movie_id", "genres"), "movie_id")
    .filter(F.size("genres") > 0)
    .select("user_id", F.explode("genres").alias("genre"), "rating")
    .groupBy("user_id", "genre")
    .agg(F.count("*").alias("g_n"), F.avg("rating").alias("g_avg"))
)
# user's top-5 genres by rating count (for overlap feature)
top5 = (profile.withColumn("rk", F.row_number().over(
    Window.partitionBy("user_id").orderBy(F.col("g_n").desc(), F.col("genre"))))
    .filter("rk <= 5")
    .groupBy("user_id")
    .agg(F.collect_set("genre").alias("user_top_genres")))

# ---- temporal feature: days since user's previous rating ----
lagged = train.select("user_id", "movie_id", "rating", "rate_ts", "split").withColumn(
    "prev_ts", F.lag("rate_ts").over(Window.partitionBy("user_id").orderBy("rate_ts"))
)

df = (
    ratings.join(users, "user_id")
    .join(items.select("movie_id",
                       F.col("n_ratings").alias("item_n"),
                       F.col("avg_rating").alias("item_avg"),
                       "year"), "movie_id")
    .join(lagged.select("user_id", "movie_id", "rate_ts", "prev_ts"), ["user_id", "movie_id", "rate_ts"], "left")
    .withColumn("days_since_prev", (F.col("rate_ts") - F.col("prev_ts")) / 86400.0)
)

# genre affinity: join exploded (user, genre, g_avg) with movie genres, average per row
aff = (ratings.join(movies.select("movie_id", "genres"), "movie_id")
       .filter(F.size("genres") > 0)
       .select("user_id", "movie_id", "rate_ts", F.explode("genres").alias("genre"))
       .join(profile.select("user_id", "genre", "g_avg", "g_n"), ["user_id", "genre"], "left"))
aff_agg = aff.groupBy("user_id", "movie_id", "rate_ts").agg(
    F.sum(F.coalesce(F.col("g_avg") * F.col("g_n"), F.lit(0.0))).alias("aff_wsum"),
    F.sum(F.coalesce(F.col("g_n"), F.lit(0))).alias("aff_wcnt"),
)
df = (df.join(aff_agg, ["user_id", "movie_id", "rate_ts"], "left")
      .withColumn("genre_affinity",
                  F.when(F.col("aff_wcnt") > 0, F.col("aff_wsum") / F.col("aff_wcnt"))
                  .otherwise(F.col("avg_rating")))
      .join(top5, "user_id", "left")
      .join(movies.select("movie_id", F.col("genres").alias("m_genres")), "movie_id")
      .withColumn("genre_overlap", F.size(F.array_intersect(F.coalesce("user_top_genres", F.array()), "m_genres")))
      .withColumn("label", (F.col("rating") >= 4.0).cast("int")))

samples = df.select(
    "label", "user_id", "movie_id",
    F.col("n_ratings").alias("user_n"), F.col("avg_rating").alias("user_avg"),
    "item_n", "item_avg", "year",
    F.round("days_since_prev", 3).alias("days_since_prev"),
    F.round("genre_affinity", 3).alias("genre_affinity"),
    "genre_overlap", "split",
)

train_out = samples.filter("split = 'train'").sample(fraction=TRAIN_FRACTION, seed=42)
val_out = samples.filter("split = 'val'")
test_out = samples.filter("split = 'test'")
train_out.unionByName(val_out).unionByName(test_out).coalesce(1).write.mode(
    "overwrite").option("header", True).csv("/project/export/samples")
print("samples exported")

# ---- movie content text for SID building ----
top_tags = (spark.table("movie.ods_genome_scores")
            .filter(F.col("line") != "movieId,tagId,relevance")
            .select(F.split("line", ",").getItem(0).cast("int").alias("movie_id"),
                    F.split("line", ",").getItem(1).cast("int").alias("tag_id"),
                    F.split("line", ",").getItem(2).cast("float").alias("relevance"))
            .filter(F.col("movie_id").isNotNull()))
tag_names = (spark.table("movie.ods_genome_tags")
             .filter(F.col("line") != "tagId,tag")
             .select(F.split("line", ",").getItem(0).cast("int").alias("tag_id"),
                     F.split("line", ",").getItem(1).alias("tag")))
content = (top_tags.withColumn("rk", F.row_number().over(
    Window.partitionBy("movie_id").orderBy(F.col("relevance").desc())))
    .filter("rk <= 10")
    .join(tag_names, "tag_id")
    .groupBy("movie_id").agg(F.concat_ws(" ", F.sort_array(F.collect_list("tag"))).alias("top_tags")))
content_out = (movies.select("movie_id", "title", "year",
                             F.concat_ws("|", "genres").alias("genres"))
               .join(content, "movie_id", "left")
               .na.fill({"top_tags": ""}))
content_out.coalesce(1).write.mode("overwrite").option("header", True).csv(
    "/project/export/movie_content")
print("movie_content exported")
spark.stop()
