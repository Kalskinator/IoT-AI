import os

# Set security and hostname environment variables
os.environ["SPARK_LOCAL_HOSTNAME"] = "localhost"
os.environ["_JAVA_OPTIONS"] = "-Djava.security.manager=allow"

from pyspark.sql import SparkSession
from pyspark.sql.functions import isnull, when, count, col
from pyspark.ml.feature import StringIndexer, VectorAssembler, OneHotEncoder
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql import functions as F
from pyspark.sql import types as T
from pyspark.ml import Pipeline

# Create SparkSession with additional configurations
spark = (
    SparkSession.builder.appName("Lab5")
    .config("spark.driver.host", "localhost")
    .config("spark.driver.bindAddress", "127.0.0.1")
    .config("spark.sql.shuffle.partitions", "10")
    .config("spark.default.parallelism", "10")
    .master("local[*]")
    .getOrCreate()
)


def without_pipelines():
    df = spark.read.format("csv").option("header", True).load("Lab5/Data/train.csv")

    # df.show(5)
    # cast survived pclass age and fare to float
    dataset = df.select(
        col("Survived").cast("float"),
        col("Pclass").cast("float"),
        col("Sex"),
        col("Age").cast("float"),
        col("Fare").cast("float"),
        col("Embarked"),
    )

    # null_counts = dataset.select([count(when(isnull(c), c)).alias(c) for c in dataset.columns]).show()

    dataset = dataset.replace("?", None).dropna(how="any")

    # null_counts = dataset.select([count(when(isnull(c), c)).alias(c) for c in dataset.columns]).show()

    dataset = (
        StringIndexer(inputCol="Sex", outputCol="Gender", handleInvalid="keep")
        .fit(dataset)
        .transform(dataset)
    )

    dataset = (
        StringIndexer(inputCol="Embarked", outputCol="Boarded", handleInvalid="keep")
        .fit(dataset)
        .transform(dataset)
    )

    dataset = dataset.drop("Embarked", "Sex")

    required_features = ["Pclass", "Gender", "Age", "Fare", "Boarded"]

    assembler = VectorAssembler(inputCols=required_features, outputCol="features")
    transformed_data = assembler.transform(dataset)

    transformed_data.show(3)

    # split the data into training and testing
    train_data, test_data = transformed_data.randomSplit([0.8, 0.2], seed=42)
    print("Number of training samples: ", train_data.count())
    print("Number of testing samples: ", test_data.count())

    rf = RandomForestClassifier(
        numTrees=10, maxDepth=5, labelCol="Survived", featuresCol="features"
    )

    model = rf.fit(train_data)

    predictions = model.transform(test_data)

    evaluator = MulticlassClassificationEvaluator(
        labelCol="Survived", predictionCol="prediction", metricName="accuracy"
    )

    accuracy = evaluator.evaluate(predictions)
    print(f"Training Accuracy: {accuracy}")


def with_pipelines():
    # Load and prepare initial dataset
    df = spark.read.format("csv").option("header", True).load("Lab5/Data/train.csv")

    # Initial data selection without transformations
    dataset = df.select(
        col("Survived").cast("float"),
        col("Pclass").cast("float"),
        col("Sex"),
        col("Age").cast("float"),
        col("Fare").cast("float"),
        col("Embarked"),
    )

    dataset = dataset.replace("?", None).dropna(how="any")

    train_data, test_data = dataset.randomSplit([0.8, 0.2], seed=42)

    sex_indexer = StringIndexer(inputCol="Sex", outputCol="Gender", handleInvalid="keep")
    embarked_indexer = StringIndexer(inputCol="Embarked", outputCol="Boarded", handleInvalid="keep")

    assembler = VectorAssembler(
        inputCols=["Pclass", "Gender", "Age", "Fare", "Boarded"], outputCol="features"
    )

    rf_model = RandomForestClassifier(
        numTrees=10, maxDepth=5, labelCol="Survived", featuresCol="features"
    )

    # Create and fit pipeline
    pipeline = Pipeline(stages=[sex_indexer, embarked_indexer, assembler, rf_model])
    final_pipeline = pipeline.fit(train_data)

    # Make predictions
    predictions = final_pipeline.transform(test_data)
    predictions.show(5, truncate=False)


def main():
    pass


if __name__ == "__main__":
    # without_pipelines()
    with_pipelines()
