# Spark NLP in CDE

A Cloudera Data Engineering (CDE) Session is an interactive short-lived development environment for running Spark commands to help you iterate upon and build your Spark workloads.

You can use CDE Sessions in CDE Virtual Clusters of type "All Purpose - Tier 2". The following commands illustrate a basic Document Chunking example with Spark NLP.

### Requirements

* A CDE 1.22 Service in Public or Private Cloud (AWS, Azure, OCP, Cloudera ECS OK)
* A CDE Virtual Cluster of type "All Purpose" with Spark 3.3.
* A working installation of the CDE CLI on your local machine.

### Step by Step Commands

```
% cde resource create --name sparknlp-env \
                      --type python-env

% cde resource create --name text-data

% cde resource upload --name sparknlp-env \
                      --local-path requirements.txt

% cde resource upload --name text-data \
                      --local-path data/news_category_test.csv

% cde resource upload --name text-data \
                      --local-path test.py

% cde session create --name cde-spark-nlp \
                     --type pyspark
                     --python-env-resource-name sparknlp-env
                     --mount-1-resource text-data

% cde session interact --name cde-spark-nlp
```

Enter the following commands in the Spark Shell:

```
import pyspark.sql.functions as F

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
```
