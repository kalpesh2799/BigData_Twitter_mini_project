from pyspark import SparkContext, SparkConf,sql
from pyspark.ml import Pipeline
from pyspark.ml.feature import Tokenizer, StopWordsRemover
from pyspark.sql.functions import concat_ws, regexp_replace, lower, col, trim, size, udf
from nltk.corpus import stopwords
from pyspark.sql.types import ArrayType, StringType

stops = stopwords.words('english')
master = 'local'
appName = 'PySpark_Dataframe Operations'

config = SparkConf().setAppName(appName).setMaster(master)
sc = SparkContext(conf=config)
# You will need to create the sqlContext
ss = sql.SparkSession(sc)

if sc:
    print(sc.appName)
else:
    print('Could not initialise pyspark session')

Data = ss.readStream.format('socket').option('host', 'localhost').option('port', 1313).load()

# def tostr(df):
#     return  concat_ws(",", df)
#
# def saver(df, epoch_id):
#     tokenizer = Tokenizer(inputCol="value", outputCol="words")
#     df_tokenized = tokenizer.transform(df)
#     df_tokenized.show()
#     df_tokenized=df_tokenized.withColumn("cstweets",tostr(df_tokenized.words)).drop("words").drop("value")
#     df_tokenized.show()
#     if df_tokenized.count()>0:
#         df_tokenized.write.mode("overwrite").option('header', True).csv("hdfs://localhost:9000/user/stage/")
#
#
# qy = Data.writeStream.foreachBatch(saver).start()
# qy.awaitTermination()

df_select_clean = (Data.withColumn("tweet_text", regexp_replace("value", r"[@#&][A-Za-z0-9_-]+", " "))
                   .withColumn("tweet_text", regexp_replace("tweet_text", r"\w+:\/\/\S+", " "))
                   .withColumn("tweet_text", regexp_replace("tweet_text", r"[^A-Za-z]", " "))
                   .withColumn("tweet_text", regexp_replace("tweet_text", r"\s+", " "))
                   .withColumn("tweet_text", lower(col("tweet_text")))
                   .withColumn("tweet_text", trim(col("tweet_text")))
                   ).drop("value")

#df_select_clean=df_select_clean.withColumn("tweet_text", trim(col("tweet_text"))).dropna("any")

# top5_udf = udf(lambda l: print(l.sort(key=len)[0:5]), returnType=ArrayType(StringType()))
top5_udf = udf(lambda l: (sorted(l,key=len,reverse=True))[0:5], returnType=ArrayType(StringType()))
to_str_udf =udf(lambda l : ",".join(l), returnType=StringType())


tokenizer = Tokenizer(inputCol="tweet_text", outputCol="tokens")
stopword_remover = StopWordsRemover(inputCol="tokens", outputCol="remove_stop").setStopWords(stops)

df_tokenized = tokenizer.transform(df_select_clean).drop("tweet_text")
df_tokenized = df_tokenized.withColumn("tokens_length", size("tokens")).filter("tokens_length>1").drop("tokens_length")
df_rmstop = stopword_remover.transform(df_tokenized).drop("tokens")
df_top_5=df_rmstop.withColumn("top_5",top5_udf(col("remove_stop"))).drop("remove_stop")
to_str = df_top_5.withColumn("tweets",to_str_udf(col("top_5"))).drop("top_5")

# qy=to_str.writeStream.outputMode('append').format('console').start()
# qy.awaitTermination()


qy=to_str.writeStream.outputMode('append').format("csv")\
    .option("path","hdfs://localhost:9000/user/stage_trial3/")\
    .option("checkpointLocation", "hdfs://localhost:9000/user/chk/")\
    .start()
qy.awaitTermination()

