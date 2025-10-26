#****************************************************************************
# (C) Cloudera, Inc. 2020-2025
#  All rights reserved.
#
#  Applicable Open Source License: GNU Affero General Public License v3.0
#
#  NOTE: Cloudera open source products are modular software products
#  made up of hundreds of individual components, each of which was
#  individually copyrighted.  Each Cloudera open source product is a
#  collective work under U.S. Copyright Law. Your license to use the
#  collective work is as provided in your written agreement with
#  Cloudera.  Used apart from the collective work, this file is
#  licensed for your use pursuant to the open source license
#  identified above.
#
#  This code is provided to you pursuant a written agreement with
#  (i) Cloudera, Inc. or (ii) a third-party authorized to distribute
#  this code. If you do not have a written agreement with Cloudera nor
#  with an authorized and properly licensed third party, you do not
#  have any rights to access nor to use this code.
#
#  Absent a written agreement with Cloudera, Inc. (“Cloudera”) to the
#  contrary, A) CLOUDERA PROVIDES THIS CODE TO YOU WITHOUT WARRANTIES OF ANY
#  KIND; (B) CLOUDERA DISCLAIMS ANY AND ALL EXPRESS AND IMPLIED
#  WARRANTIES WITH RESPECT TO THIS CODE, INCLUDING BUT NOT LIMITED TO
#  IMPLIED WARRANTIES OF TITLE, NON-INFRINGEMENT, MERCHANTABILITY AND
#  FITNESS FOR A PARTICULAR PURPOSE; (C) CLOUDERA IS NOT LIABLE TO YOU,
#  AND WILL NOT DEFEND, INDEMNIFY, NOR HOLD YOU HARMLESS FOR ANY CLAIMS
#  ARISING FROM OR RELATED TO THE CODE; AND (D)WITH RESPECT TO YOUR EXERCISE
#  OF ANY RIGHTS GRANTED TO YOU FOR THE CODE, CLOUDERA IS NOT LIABLE FOR ANY
#  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, PUNITIVE OR
#  CONSEQUENTIAL DAMAGES INCLUDING, BUT NOT LIMITED TO, DAMAGES
#  RELATED TO LOST REVENUE, LOST PROFITS, LOSS OF INCOME, LOSS OF
#  BUSINESS ADVANTAGE OR UNAVAILABILITY, OR LOSS OR CORRUPTION OF
#  DATA.
#
# #  Author(s): Paul de Fusco
#***************************************************************************/

import sparknlp
import pyspark.sql.functions as F

spark = SparkSession.builder \
    .appName("Spark NLP") \
    .master("local[*]") \
    .config("spark.driver.memory", "16G") \
    .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
    .config("spark.kryoserializer.buffer.max", "2000M") \
    .config("spark.driver.maxResultSize", "0") \
    .config("spark.jars.packages", "com.johnsnowlabs.nlp:spark-nlp_2.12:6.2.0") \
    .getOrCreate()

news_df = spark.read\
                .option("header", "true")\
                .csv("/app/mount/news_category_test.csv")\
                .withColumnRenamed("description", "text")

news_df.show(truncate=50)

news_df.take(3)

entities = ['parent firm', 'economy', 'amino acids']

with open ('entities.txt', 'w') as f:
    for i in entities:
        f.write(i+'\n')

documentAssembler = DocumentAssembler() \
    .setInputCol("text") \
    .setOutputCol("document")

tokenizer = Tokenizer() \
    .setInputCols(["document"]) \
    .setOutputCol("token")

entity_extractor = TextMatcher() \
                      .setInputCols(["document",'token'])\
                      .setOutputCol("entities")\
                      .setEntities("entities.txt")\
                      .setCaseSensitive(False)\
                      .setEntityValue('entities')

nlpPipeline = Pipeline(stages=[documentAssembler,
                               tokenizer,
                               entity_extractor])

result = nlpPipeline.fit(news_df).transform(news_df.limit(10))

result.select('entities.result').take(3)

chunk_embeddings = ChunkEmbeddings() \
                      .setInputCols(["entities", "embeddings"]) \
                      .setOutputCol("chunk_embeddings") \
                      .setPoolingStrategy("AVERAGE")

glove_embeddings = WordEmbeddingsModel.pretrained('glove_100d')\
    .setInputCols(["document", "token"])\
    .setOutputCol("embeddings")

nlpPipeline = Pipeline(stages=[documentAssembler,
                               tokenizer,
                               entity_extractor,
                               glove_embeddings,
                               chunk_embeddings])

result = nlpPipeline.fit(news_df).transform(news_df.limit(10))

result_df = result.select(F.explode(F.arrays_zip(result.entities.result,
                                                 result.chunk_embeddings.embeddings)).alias("cols")) \
                  .select(F.expr("cols['0']").alias("entities"),
                          F.expr("cols['1']").alias("chunk_embeddings"))

result_df.show(truncate=100)
