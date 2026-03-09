"""
RustFS client – wraps RustFS (S3-compatible, Apache 2.0 MinIO alternative)
for plot and dataset storage.

RustFS exposes an S3-compatible API, so we use the ``minio`` Python SDK
which works seamlessly against any S3-compatible endpoint.

Buckets
-------
* ``xai-plots``    – PNG images of generated visualizations
* ``xai-datasets`` – uploaded CSV / JSON data files
* ``xai-metadata`` – JSON sidecar files (plot_metadata, dataset info)

Every object is stored with user-scoped keys:
    {user_id}/{object_id}.{ext}

Remote instance: https://rustfs.extra-brain.unparallel.pt
RustFS docs:     https://rustfs.com
"""

import io
import json
import logging
import os
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration – all overridable via env vars
# ---------------------------------------------------------------------------
RESTFS_ENDPOINT = os.environ.get("RESTFS_ENDPOINT", "rustfs.extra-brain.unparallel.pt")
RESTFS_ACCESS_KEY = os.environ.get("RESTFS_ACCESS_KEY", "")
RESTFS_SECRET_KEY = os.environ.get("RESTFS_SECRET_KEY", "")
RESTFS_SECURE = os.environ.get("RESTFS_SECURE", "true").lower() == "true"
RESTFS_REGION = os.environ.get("RESTFS_REGION", "us-east-1")

BUCKET_PLOTS = os.environ.get("RESTFS_BUCKET_PLOTS", "xai-plots")
BUCKET_DATASETS = os.environ.get("RESTFS_BUCKET_DATASETS", "xai-datasets")
BUCKET_METADATA = os.environ.get("RESTFS_BUCKET_METADATA", "xai-metadata")

ALL_BUCKETS = [BUCKET_PLOTS, BUCKET_DATASETS, BUCKET_METADATA]

_client = None  # lazy singleton


def _get_client():
    """Return a cached S3 client (lazy init).
    
    Uses the ``minio`` Python SDK which is fully compatible with RustFS's
    S3 API.  RustFS is a drop-in replacement – no code changes needed.
    """
    global _client
    if _client is not None:
        return _client

    if not RESTFS_ACCESS_KEY or not RESTFS_SECRET_KEY:
        logger.warning("RustFS credentials not set (RESTFS_ACCESS_KEY / RESTFS_SECRET_KEY) – falling back to local FS")
        return None

    try:
        from minio import Minio
        _client = Minio(
            RESTFS_ENDPOINT,
            access_key=RESTFS_ACCESS_KEY,
            secret_key=RESTFS_SECRET_KEY,
            secure=RESTFS_SECURE,
            region=RESTFS_REGION,
        )
        # Ensure buckets exist
        for bucket in ALL_BUCKETS:
            if not _client.bucket_exists(bucket):
                _client.make_bucket(bucket)
                logger.info("Created bucket %s on RustFS", bucket)
        logger.info("RustFS client initialised -> %s (secure=%s)", RESTFS_ENDPOINT, RESTFS_SECURE)
        return _client
    except Exception as exc:
        logger.warning("RustFS unavailable (%s) – falling back to local FS", exc)
        return None


def is_available() -> bool:
    """True when the RustFS endpoint is reachable."""
    return _get_client() is not None


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------

def put_bytes(bucket: str, key: str, data: bytes,
              content_type: str = "application/octet-stream",
              metadata: Optional[Dict[str, str]] = None) -> bool:
    """Upload raw bytes. Returns True on success."""
    client = _get_client()
    if client is None:
        return False
    try:
        client.put_object(
            bucket, key,
            io.BytesIO(data), len(data),
            content_type=content_type,
            metadata=metadata or {},
        )
        return True
    except Exception as exc:
        logger.warning("put_bytes failed bucket=%s key=%s: %s", bucket, key, exc)
        return False


def get_bytes(bucket: str, key: str) -> Optional[bytes]:
    """Download raw bytes. Returns None on failure."""
    client = _get_client()
    if client is None:
        return None
    try:
        resp = client.get_object(bucket, key)
        data = resp.read()
        resp.close()
        resp.release_conn()
        return data
    except Exception as exc:
        logger.warning("get_bytes failed bucket=%s key=%s: %s", bucket, key, exc)
        return None


def delete_object(bucket: str, key: str) -> bool:
    client = _get_client()
    if client is None:
        return False
    try:
        client.remove_object(bucket, key)
        return True
    except Exception as exc:
        logger.warning("delete_object failed: %s", exc)
        return False


def list_objects(bucket: str, prefix: str = "",
                 recursive: bool = True) -> List[Dict[str, Any]]:
    """List objects with a given prefix. Returns list of dicts."""
    client = _get_client()
    if client is None:
        return []
    try:
        results = []
        for obj in client.list_objects(bucket, prefix=prefix, recursive=recursive):
            results.append({
                "key": obj.object_name,
                "size": obj.size,
                "last_modified": obj.last_modified.isoformat() if obj.last_modified else None,
                "etag": obj.etag,
            })
        return results
    except Exception as exc:
        logger.warning("list_objects failed: %s", exc)
        return []


# ---------------------------------------------------------------------------
# Plot-specific helpers
# ---------------------------------------------------------------------------

def save_plot_image(image_bytes: bytes, user_id: str, plot_id: str,
                    metadata: Optional[Dict[str, str]] = None) -> str:
    """
    Save a PNG plot image to RustFS.
    Returns the object key on success, empty string on failure.
    """
    key = f"{user_id}/{plot_id}.png"
    meta = {"x-amz-meta-user-id": user_id, "x-amz-meta-plot-id": plot_id}
    if metadata:
        for k, v in metadata.items():
            meta[f"x-amz-meta-{k}"] = str(v)[:256]
    ok = put_bytes(BUCKET_PLOTS, key, image_bytes,
                   content_type="image/png", metadata=meta)
    return key if ok else ""


def get_plot_image(user_id: str, plot_id: str) -> Optional[bytes]:
    """Retrieve a plot PNG from RustFS."""
    key = f"{user_id}/{plot_id}.png"
    return get_bytes(BUCKET_PLOTS, key)


def list_user_plots(user_id: str) -> List[Dict[str, Any]]:
    """List all plot objects for a user."""
    return list_objects(BUCKET_PLOTS, prefix=f"{user_id}/")


def delete_plot_image(user_id: str, plot_id: str) -> bool:
    key = f"{user_id}/{plot_id}.png"
    return delete_object(BUCKET_PLOTS, key)


# ---------------------------------------------------------------------------
# Metadata JSON helpers
# ---------------------------------------------------------------------------

def save_metadata_json(user_id: str, doc_id: str,
                       data: Dict[str, Any]) -> bool:
    key = f"{user_id}/{doc_id}.json"
    payload = json.dumps(data, default=str).encode("utf-8")
    return put_bytes(BUCKET_METADATA, key, payload,
                     content_type="application/json")


def get_metadata_json(user_id: str, doc_id: str) -> Optional[Dict[str, Any]]:
    key = f"{user_id}/{doc_id}.json"
    raw = get_bytes(BUCKET_METADATA, key)
    if raw is None:
        return None
    try:
        return json.loads(raw.decode("utf-8"))
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def save_dataset(file_bytes: bytes, user_id: str, filename: str,
                 content_type: str = "text/csv") -> str:
    """Save an uploaded dataset file. Returns the object key."""
    key = f"{user_id}/{filename}"
    ok = put_bytes(BUCKET_DATASETS, key, file_bytes,
                   content_type=content_type,
                   metadata={"x-amz-meta-user-id": user_id,
                             "x-amz-meta-filename": filename})
    return key if ok else ""


def get_dataset(user_id: str, filename: str) -> Optional[bytes]:
    key = f"{user_id}/{filename}"
    return get_bytes(BUCKET_DATASETS, key)


def list_user_datasets(user_id: str) -> List[Dict[str, Any]]:
    return list_objects(BUCKET_DATASETS, prefix=f"{user_id}/")


def delete_dataset(user_id: str, filename: str) -> bool:
    key = f"{user_id}/{filename}"
    return delete_object(BUCKET_DATASETS, key)
