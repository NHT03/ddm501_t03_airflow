"""
Khởi tạo bucket S3 trên MinIO cho MLflow Artifact Store
"""
import os
import sys
import time
import boto3
from botocore.exceptions import ClientError, EndpointConnectionError

BUCKET = os.environ.get("MLFLOW_BUCKET", "mlflow")
ENDPOINT = os.environ.get("MLFLOW_S3_ENDPOINT_URL", "http://minio:9000")


def main() -> None:
    s3 = boto3.client("s3", endpoint_url=ENDPOINT)

    for attempt in range(1, 31):
        try:
            s3.list_buckets()
            break
        except (EndpointConnectionError, ClientError) as exc:
            print(f"Waiting for MinIO at {ENDPOINT} ({attempt}/30): {type(exc).__name__}")
            time.sleep(2)
    else:
        sys.exit(f"{ENDPOINT} never answered")

    existing = [b["Name"] for b in s3.list_buckets()["Buckets"]]
    if BUCKET in existing:
        print(f"Bucket '{BUCKET}' already exists.")
        return
    s3.create_bucket(Bucket=BUCKET)
    print(f"Successfully created bucket '{BUCKET}'.")


if __name__ == "__main__":
    main()
