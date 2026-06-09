

import os

from dotenv import load_dotenv
from pyspark.sql import functions as F
from pyspark.sql import SparkSession
from pyspark.sql.types import DoubleType, StructField, StructType


def build_schema() -> StructType:
    fields = [
        StructField("unit", DoubleType(), True),
        StructField("cycle", DoubleType(), True),
    ]
    fields.extend([StructField(f"sensor_{i}", DoubleType(), True) for i in range(1, 25)])
    fields.extend(
        [
            StructField("RUL", DoubleType(), True),
            StructField("failure", DoubleType(), True),
        ]
    )
    return StructType(fields)


def run_kafka_stream_reader():
    load_dotenv()
    bootstrap_servers = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
    topic = os.getenv("KAFKA_TOPIC", "turbofan-fd001")
    failure_threshold = float(os.getenv("FAILURE_THRESHOLD", "30"))
    output_path = os.getenv("SPARK_OUTPUT_PATH", "outputs/stream_predictions")
    checkpoint_console = os.getenv("SPARK_CHECKPOINT_CONSOLE", "outputs/checkpoints/console")
    checkpoint_parquet = os.getenv("SPARK_CHECKPOINT_PARQUET", "outputs/checkpoints/parquet")

    spark = (
        SparkSession.builder
        .appName("PredictiveMaintenanceKafkaSpark")
        .getOrCreate()
    )

    df = (
        spark.readStream
        .format("kafka")
        .option("kafka.bootstrap.servers", bootstrap_servers)
        .option("subscribe", topic)
        .option("startingOffsets", "earliest")
        .load()
    )

    values_df = df.selectExpr("CAST(value AS STRING) AS message")

    schema = build_schema()

    parsed_df = (
        values_df
        .select(F.from_json(F.col("message"), schema).alias("data"))
        .select("data.*")
    )

    scored_df = parsed_df.withColumn(
        "predicted_failure",
        F.when(F.col("RUL") < F.lit(failure_threshold), F.lit(1.0)).otherwise(F.lit(0.0)),
    )

    result_df = scored_df.select(
        "unit",
        "cycle",
        "RUL",
        "failure",
        "predicted_failure",
    )

    console_query = (
        result_df.writeStream
        .format("console")
        .option("truncate", False)
        .option("checkpointLocation", checkpoint_console)
        .start()
    )

    parquet_query = (
        result_df.writeStream
        .format("parquet")
        .option("path", output_path)
        .option("checkpointLocation", checkpoint_parquet)
        .outputMode("append")
        .start()
    )

    print(f"Streaming from topic '{topic}' on '{bootstrap_servers}'")
    print(f"Writing stream output to: {output_path}")
    print("Columns: unit, cycle, RUL, failure, predicted_failure")

    console_query.awaitTermination()
    parquet_query.awaitTermination()


if __name__ == "__main__":
    run_kafka_stream_reader()
