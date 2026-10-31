-- Hive mirror tables over Spark-written parquet paths.
-- Spark Catalog (embedded Derby in master:/opt/etl/derby) is the pipeline's
-- source of truth; these external tables expose the same data to beeline for
-- interactive queries and screenshots. Run inside hive container via beeline:
--   docker exec -it hive /usr/local/hive/bin/beeline -u jdbc:hive2://localhost:10000/default -f /opt/etl/hive_mirror.sql

CREATE DATABASE IF NOT EXISTS movie;

CREATE EXTERNAL TABLE IF NOT EXISTS movie.dwd_ratings (
  user_id INT, movie_id INT, rating FLOAT, rate_ts BIGINT
) PARTITIONED BY (split STRING)
STORED AS PARQUET
LOCATION '/user/hive/warehouse/movie.db/dwd_ratings';
MSCK REPAIR TABLE movie.dwd_ratings;

CREATE EXTERNAL TABLE IF NOT EXISTS movie.dwd_movies (
  movie_id INT, title STRING, year INT, genres ARRAY<STRING>
) STORED AS PARQUET
LOCATION '/user/hive/warehouse/movie.db/dwd_movies';

CREATE EXTERNAL TABLE IF NOT EXISTS movie.dwd_tags (
  user_id INT, tag STRING, tag_ts BIGINT
) STORED AS PARQUET
LOCATION '/user/hive/warehouse/movie.db/dwd_tags';

CREATE EXTERNAL TABLE IF NOT EXISTS movie.dws_user_stats (
  user_id INT, n_ratings BIGINT, avg_rating FLOAT, first_ts BIGINT, last_ts BIGINT
) STORED AS PARQUET
LOCATION '/user/hive/warehouse/movie.db/dws_user_stats';

CREATE EXTERNAL TABLE IF NOT EXISTS movie.dws_item_stats (
  movie_id INT, n_ratings BIGINT, avg_rating FLOAT,
  title STRING, year INT, genres ARRAY<STRING>
) STORED AS PARQUET
LOCATION '/user/hive/warehouse/movie.db/dws_item_stats';

CREATE EXTERNAL TABLE IF NOT EXISTS movie.dws_gt (
  user_id INT, movie_id INT, rating FLOAT, split STRING
) STORED AS PARQUET
LOCATION '/user/hive/warehouse/movie.db/dws_gt';

CREATE EXTERNAL TABLE IF NOT EXISTS movie.dws_user_seq (
  user_id INT, seq ARRAY<INT>
) STORED AS PARQUET
LOCATION '/user/hive/warehouse/movie.db/dws_user_seq';

CREATE EXTERNAL TABLE IF NOT EXISTS movie.dws_cooccur (
  i INT, j INT, count BIGINT
) STORED AS PARQUET
LOCATION '/user/hive/warehouse/movie.db/dws_cooccur';

-- sanity
SELECT split, count(*) AS n FROM movie.dwd_ratings GROUP BY split ORDER BY split;
SELECT count(*) AS n_movies FROM movie.dwd_movies;
SELECT title, year, genres FROM movie.dws_item_stats ORDER BY n_ratings DESC LIMIT 5;
