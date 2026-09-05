import os
from pathlib import Path

import boto3
from botocore.exceptions import BotoCoreError, ClientError


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACTS = [
    ROOT / "models" / "fake_news_pipeline.joblib",
    ROOT / "reports" / "metrics.json",
]


def _client():
    endpoint_url = os.getenv("AWS_ENDPOINT_URL")
    return boto3.client(
        "s3",
        endpoint_url=endpoint_url,
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID", "test"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY", "test"),
        region_name=os.getenv("AWS_DEFAULT_REGION", "us-east-1"),
    )


def bucket_name() -> str:
    return os.getenv("MODEL_BUCKET", "fake-news-model-artifacts")


def ensure_bucket() -> None:
    client = _client()
    bucket = bucket_name()
    try:
        client.head_bucket(Bucket=bucket)
    except ClientError:
        client.create_bucket(Bucket=bucket)


def upload_artifacts(paths: list[Path] | None = None) -> list[dict]:
    paths = paths or DEFAULT_ARTIFACTS
    ensure_bucket()
    client = _client()
    uploaded = []

    for path in paths:
        if not path.exists():
            continue
        key = f"artifacts/{path.name}"
        client.upload_file(str(path), bucket_name(), key)
        uploaded.append({"file": str(path.relative_to(ROOT)), "bucket": bucket_name(), "key": key})

    return uploaded


def cloud_status() -> dict:
    try:
        ensure_bucket()
        response = _client().list_objects_v2(Bucket=bucket_name(), MaxKeys=20)
        objects = [
            {"key": item["Key"], "size": item["Size"]}
            for item in response.get("Contents", [])
        ]
        return {
            "enabled": True,
            "provider": "LocalStack S3" if os.getenv("AWS_ENDPOINT_URL") else "AWS S3",
            "bucket": bucket_name(),
            "objects": objects,
        }
    except (BotoCoreError, ClientError, OSError) as exc:
        return {
            "enabled": False,
            "provider": "LocalStack S3 / AWS S3",
            "bucket": bucket_name(),
            "error": str(exc),
        }
