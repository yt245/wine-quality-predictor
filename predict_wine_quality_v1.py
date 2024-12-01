from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegressionModel
from pyspark.sql.functions import col
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import sys

def load_and_clean_data(spark, file_path, delimiter=";"):
    """
    Load and clean the CSV data from a local path.
    """
    # Read CSV file from local path
    data = spark.read.csv(file_path, header=True, inferSchema=True, sep=delimiter)

    # Clean column names by replacing spaces and removing quotes
    for col_name in data.columns:
        clean_name = col_name.replace(' ', '_').replace('"', '').strip()
        data = data.withColumnRenamed(col_name, clean_name)

    return data


def main():
    # Initialize Spark session with S3 configuration
    spark = SparkSession.builder \
        .appName("WineQualityPrediction_v1") \
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem") \
        .config("spark.hadoop.fs.s3a.aws.credentials.provider", "com.amazonaws.auth.DefaultAWSCredentialsProviderChain") \
        .config("spark.hadoop.fs.s3a.access.key", "YOUR_ACCESS_KEY") \
        .config("spark.hadoop.fs.s3a.secret.key", "YOUR_SECRET_KEY") \
        .getOrCreate()

    # Input and Output paths
    input_csv_path = sys.argv[1]  # Local path for TestDataset.csv, e.g., /home/ubuntu/TestDataset.csv
    model_path = sys.argv[2]      # Path to the model on S3 (should be s3a://path/to/model)
    output_path = sys.argv[3]     # Path to write predictions

    # Load and clean the input dataset
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

    # Load the pre-trained model from S3 using the s3a scheme
    model = LogisticRegressionModel.load(model_path)

    # Make predictions
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

    # Write predictions to the specified output path
    predictions.select(output_columns).write.csv(output_path, header=True, mode="overwrite")

    print("Prediction output written successfully.")

    # Stop the Spark session
    spark.stop()


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: spark-submit predict_wine_quality_v1.py <input_csv_path> <model_s3_path> <output_csv_path>")
        sys.exit(-1)

    main()
