from pyspark.sql import SparkSession

spark = (
    SparkSession.builder.appName("hive-metastore-check")
    .enableHiveSupport()
    .getOrCreate()
)
dbs = [r[0] for r in spark.sql("show databases").collect()]
print("METASTORE_DATABASES:", dbs)
if "test" in dbs:
    n = spark.table("test.wordcount").count()
    print("METASTORE_SHARED_OK test.wordcount rows =", n)
spark.stop()
