import io
from minio import Minio
import config


class StorageClient:
    def __init__(self):
        self._client = Minio(
            config.MINIO_ENDPOINT,
            access_key=config.MINIO_ACCESS_KEY,
            secret_key=config.MINIO_SECRET_KEY,
            secure=config.MINIO_SECURE,
        )

    def get_object_bytes(self, bucket: str, object_name: str) -> bytes:
        response = self._client.get_object(bucket, object_name)
        return response.read()

    def put_object_bytes(self, bucket: str, object_name: str, data: bytes, content_type: str = "application/octet-stream"):
        self._client.put_object(
            bucket, object_name,
            data=io.BytesIO(data),
            length=len(data),
            content_type=content_type,
        )

    def ref(self, bucket: str, object_name: str) -> str:
        """Return a minio:// reference URL."""
        return f"minio://{bucket}/{object_name}"
