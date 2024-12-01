from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

def load_and_clean_data(spark, file_path, delimiter=";"):
    """
    Load and clean the CSV data from S3.
    """
    # Read CSV file from S3
    data = spark.read.csv(file_path, header=True, inferSchema=True, sep=delimiter)

    # Clean column names by replacing spaces and removing quotes
    for col_name in data.columns:
        clean_name = col_name.replace(' ', '_').replace('"', '').strip()
        data = data.withColumnRenamed(col_name, clean_name)

    return data

def main():
    # Initialize Spark session with S3 configuration
    spark = SparkSession.builder \
        .appName("WineQualityTraining_v1") \
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem") \
        .config("spark.hadoop.fs.s3a.aws.credentials.provider", "com.amazonaws.auth.DefaultAWSCredentialsProviderChain") \
        .getOrCreate()

    # S3 bucket and file paths
    bucket = "s3a://aws-logs-802267675729-us-east-1/elasticmapreduce/j-K52JDHWQUO37/"
    training_data_path = f"{bucket}TrainingDataset.csv"
    validation_data_path = f"{bucket}ValidationDataset.csv"
    model_output_path = f"{bucket}models/logistic_regression_model"

    # Load and clean the training and validation datasets
    train_data = load_and_clean_data(spark, training_data_path)
    validation_data = load_and_clean_data(spark, validation_data_path)

    # Define feature columns and target column
    feature_columns = [
        "fixed_acidity", "volatile_acidity", "citric_acid", "residual_sugar",
        "chlorides", "free_sulfur_dioxide", "total_sulfur_dioxide", "density",
        "pH", "sulphates", "alcohol"
    ]
    target_column = "quality"

    # Assemble feature columns into a single vector column
    assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
    train_data = assembler.transform(train_data).select("features", target_column)
    validation_data = assembler.transform(validation_data).select("features", target_column)

    # Train the Logistic Regression model
    lr = LogisticRegression(featuresCol="features", labelCol=target_column, maxIter=10)
    model = lr.fit(train_data)

    # Evaluate the model on the validation dataset
    predictions = model.transform(validation_data)
    evaluator = MulticlassClassificationEvaluator(
        labelCol=target_column,
        predictionCol="prediction",
        metricName="f1"
    )
    f1_score = evaluator.evaluate(predictions)
    print(f"Validation F1 Score: {f1_score}")

    # Save the trained model to S3
    model.write().overwrite().save(model_output_path)
    print(f"Model saved to: {model_output_path}")

    # Stop the Spark session
    spark.stop()

if __name__ == "__main__":
    main()
