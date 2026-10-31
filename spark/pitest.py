from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("PySparkYarnSmoke").getOrCreate()
sc = spark.sparkContext

n_df = spark.range(0, 1000000).count()
n_rdd = sc.parallelize(range(1000), 4).map(lambda x: x * 2).sum()

print("PYSPARK_YARN_OK dataframe_count=%d rdd_sum=%d" % (n_df, n_rdd))
spark.stop()
