import os

# Model API
MODEL_API_URL = os.getenv("MODEL_API_URL", "http://model-mock:8003")

# Storage (S3-compatible: RustFS in production, MinIO for local dev)
# Reads RESTFS_* env vars first (shared with rest of the platform),
# falls back to MINIO_* for standalone dev usage.
MINIO_ENDPOINT = os.getenv("RESTFS_ENDPOINT", os.getenv("MINIO_ENDPOINT", "minio:9000"))
MINIO_ACCESS_KEY = os.getenv("RESTFS_ACCESS_KEY", os.getenv("MINIO_ACCESS_KEY", "minioadmin"))
MINIO_SECRET_KEY = os.getenv("RESTFS_SECRET_KEY", os.getenv("MINIO_SECRET_KEY", "minioadmin"))
MINIO_SECURE = os.getenv("RESTFS_SECURE", os.getenv("MINIO_SECURE", "false")).lower() == "true"

# Buckets
UC1_BUCKET = os.getenv("UC1_BUCKET", "uc1-robotics")
UC2_BUCKET = os.getenv("UC2_BUCKET", "uc2-finance")
UC3_BUCKET = os.getenv("UC3_BUCKET", "uc3-telecom")

# Results are stored under this prefix inside each bucket
RESULTS_PREFIX = os.getenv("RESULTS_PREFIX", "results")
