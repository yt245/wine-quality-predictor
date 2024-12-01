from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import RandomForestClassificationModel
from pyspark.sql.functions import col
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import sys

def load_and_clean_data(spark, file_path, delimiter=";"):
    """
    Load and clean the CSV data.
    """
    # Read CSV file from the local path
    data = spark.read.csv(file_path, header=True, inferSchema=True, sep=delimiter)

    # Clean column names by replacing spaces and removing quotes
    for col_name in data.columns:
        clean_name = col_name.replace(' ', '_').replace('"', '').strip()
        data = data.withColumnRenamed(col_name, clean_name)

    return data

def main():
    # Initialize Spark session with S3 configuration
    spark = SparkSession.builder \
        .appName("WineQualityPrediction_RFC") \
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem") \
        .config("spark.hadoop.fs.s3a.aws.credentials.provider", "com.amazonaws.auth.DefaultAWSCredentialsProviderChain") \
        .getOrCreate()

    # S3 paths for model and prediction output
    model_path = "s3a://aws-logs-802267675729-us-east-1/elasticmapreduce/j-K52JDHWQUO37/models/random_forest_model"
    output_csv_path = "s3a://aws-logs-802267675729-us-east-1/elasticmapreduce/j-K52JDHWQUO37/predictions/output"
    
    # Local path for TestDataset
    input_csv_path = "/home/ubuntu/TestDataset.csv"  # This is the local file

    # Load and clean the input dataset from local disk
    data = load_and_clean_data(spark, input_csv_path)

    # Define feature columns
    feature_columns = [
        "fixed_acidity", "volatile_acidity", "citric_acid", "residual_sugar",
        "chlorides", "free_sulfur_dioxide", "total_sulfur_dioxide", "density",
        "pH", "sulphates", "alcohol"
    ]

    # Assemble feature columns into a single vector column
    assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
    data = assembler.transform(data)

    # Load the pre-trained RandomForest model from S3
    model = RandomForestClassificationModel.load(model_path)

    # Make predictions using the trained model
    predictions = model.transform(data)

    # Select necessary columns, including 'features' column
    predictions = predictions.withColumn("features_str", col("features").cast("string"))
    output_columns = ["prediction", "quality", "features_str"]

    # Print a sample of predictions
    print("Sample rows after predictions:")
    predictions.select(output_columns).show(5)

    # Evaluate the model using F1 score
    evaluator = MulticlassClassificationEvaluator(
        labelCol="quality", predictionCol="prediction", metricName="f1"
    )
    f1_score = evaluator.evaluate(predictions)
    print(f"F1 Score: {f1_score}")

    # Write predictions to S3
    predictions.select(output_columns).write.csv(output_csv_path, header=True, mode="overwrite")

    print("Prediction output written successfully.")

    # Stop the Spark session
    spark.stop()

if __name__ == "__main__":
    main()
