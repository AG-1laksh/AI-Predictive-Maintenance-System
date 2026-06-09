

import json
import os

import pandas as pd
from dotenv import load_dotenv
from kafka import KafkaProducer


def build_producer(bootstrap_servers: str) -> KafkaProducer:
    return KafkaProducer(
        bootstrap_servers=bootstrap_servers,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
    )


def send_csv_rows_to_kafka(csv_path: str, topic: str):
    load_dotenv()
    bootstrap_servers = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")

    producer = build_producer(bootstrap_servers)
    df = pd.read_csv(csv_path)

    for _, row in df.iterrows():
        producer.send(topic, row.to_dict())

    producer.flush()
    producer.close()

    print(f"Sent {len(df)} rows to topic '{topic}'.")
    print("DataFrame head:")
    print(df.head())


if __name__ == "__main__":
    load_dotenv()
    topic_name = os.getenv("KAFKA_TOPIC", "turbofan-fd001")
    source_csv = os.getenv("KAFKA_SOURCE_CSV", "outputs/train_FD001_clean.csv")
    send_csv_rows_to_kafka(source_csv, topic_name)
