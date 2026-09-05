from src.cloud_storage import upload_artifacts


if __name__ == "__main__":
    uploaded = upload_artifacts()
    if not uploaded:
        print("No artifacts found to upload.")
    for item in uploaded:
        print(f"Uploaded {item['file']} to s3://{item['bucket']}/{item['key']}")
