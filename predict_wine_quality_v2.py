from pyspark.sql import SparkSession
from pyspark.ml.classification import LogisticRegressionModel
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql.functions import col


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
        .appName("WineQualityPrediction_v2") \
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem") \
        .config("spark.hadoop.fs.s3a.aws.credentials.provider", "com.amazonaws.auth.DefaultAWSCredentialsProviderChain") \
        .getOrCreate()

    # S3 bucket and file paths
    bucket = "s3a://aws-logs-802267675729-us-east-1/elasticmapreduce/j-K52JDHWQUO37/"
    validation_data_path = f"{bucket}ValidationDataset.csv"
    model_output_path = f"{bucket}models/logistic_regression_model"
    output_path = f"{bucket}predictions/output"

    # Load and clean the validation dataset
    validation_data = load_and_clean_data(spark, validation_data_path)

    # Define the feature columns
    feature_columns = [
        "fixed_acidity", "volatile_acidity", "citric_acid", "residual_sugar",
        "chlorides", "free_sulfur_dioxide", "total_sulfur_dioxide", "density",
        "pH", "sulphates", "alcohol"
    ]

    # Assemble the feature columns into a single features vector
    assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
    validation_data = assembler.transform(validation_data)

    # Load the pre-trained model
    print(f"Loading model from: {model_output_path}")
    model = LogisticRegressionModel.load(model_output_path)

    # Make predictions
    predictions = model.transform(validation_data)

    # Drop the 'features' column before saving if it's not needed in the output
    predictions = predictions.drop("features")

    # Define the output columns
    output_columns = ["prediction", "quality"]

    # Write predictions to the specified output path
    print(f"Writing predictions to: {output_path}")
    predictions.select(*output_columns).write.csv(output_path, header=True, mode="overwrite")

    print("Prediction output written successfully.")

    # Stop the Spark session
    spark.stop()


if __name__ == "__main__":
    main()
