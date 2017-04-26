from pyspark import SparkContext, SparkConf
from pyspark.sql import SQLContext
from pyspark.sql.functions import udf
from pyspark.sql.types import DoubleType
from accessment_model import mySim
import pandas as pd
import datetime
import time

def new_sim(target_sentence):
    return udf(lambda c: mySim(target_sentence, c), DoubleType())

time_start = datetime.datetime.now()
timeStart = time.clock()

conf = SparkConf().setMaster("local[*]").setAppName('rapid_assessment').set('spark.executor.memory', '2g')
sc = SparkContext(conf=conf)
spark = SQLContext(sc)

df = spark.read.csv('bhutan_sentence.csv', header=True)

#udf_f = udf(mySim, DoubleType())

goals = pd.read_csv('SDG_Goal.csv')
goals.columns = ['goal', 'text']

health = goals.iloc[10:19, :]



print(df.withColumn("similarity", new_sim(health.iloc[0, 1])('sentence')).show())


print('Spark takes {:.4f} seconds to calculate the similarity'.format(time.clock()-timeStart))