"""
Storage abstraction for plot images, HTML, and metadata.
Backend priority: RustFS (S3-compatible) -> local shared_volume fallback.

RustFS is an Apache 2.0 licensed, high-performance object storage system
built in Rust.  It exposes a standard S3 API so the ``minio`` Python SDK
works without changes.  See https://rustfs.com
"""

import base64
import json
import logging
import os
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Local paths (fallback)
SHARED_DATA_DIR = os.environ.get('SHARED_DATA_DIR', '/app/shared_data')
PLOTS_BASE = os.path.join(SHARED_DATA_DIR, 'plots')
IMAGES_BASE = os.path.join(SHARED_DATA_DIR, 'images')
REGISTRY_FILENAME = 'plots_registry.json'

# Try to import RustFS client
try:
    import restfs_client as restfs
    _RESTFS_OK = restfs.is_available()
except Exception:
    _RESTFS_OK = False
    restfs = None  # type: ignore


def _using_restfs() -> bool:
    """Check once per call whether RustFS is available."""
    global _RESTFS_OK
    if restfs is None:
        return False
    if not _RESTFS_OK:
        _RESTFS_OK = restfs.is_available()
    return _RESTFS_OK


# --------------------------------------------------------------------------
# Local helpers
# --------------------------------------------------------------------------

def _user_plots_dir(user_id: str) -> str:
    path = os.path.join(PLOTS_BASE, user_id)
    os.makedirs(path, exist_ok=True)
    return path


def _user_images_dir(user_id: str) -> str:
    path = os.path.join(IMAGES_BASE, user_id)
    os.makedirs(path, exist_ok=True)
    return path


def _user_registry_path(user_id: str) -> str:
    results_dir = os.path.join(SHARED_DATA_DIR, 'results', user_id)
    os.makedirs(results_dir, exist_ok=True)
    return os.path.join(results_dir, REGISTRY_FILENAME)


# --------------------------------------------------------------------------
# Plot HTML (interactive Plotly)
# --------------------------------------------------------------------------

def save_plot_html(plot_html: str, user_id: str, plot_id: str) -> str:
    if _using_restfs():
        key = f"{user_id}/{plot_id}.html"
        ok = restfs.put_bytes(
            restfs.BUCKET_PLOTS, key,
            plot_html.encode('utf-8'),
            content_type='text/html',
        )
        if ok:
            return f"restfs://{restfs.BUCKET_PLOTS}/{key}"

    base = _user_plots_dir(user_id)
    path = os.path.join(base, f'{plot_id}.html')
    try:
        with open(path, 'w', encoding='utf-8') as f:
            f.write(plot_html)
        return path
    except Exception as e:
        logger.warning('Failed to save plot HTML to disk: %s', e)
        return ''


def load_plot_html(storage_location: str, plot_html_path: str) -> Optional[str]:
    ref = plot_html_path or storage_location or ''

    if ref.startswith('restfs://') and _using_restfs():
        parts = ref.replace('restfs://', '').split('/', 1)
        if len(parts) == 2:
            data = restfs.get_bytes(parts[0], parts[1])
            if data:
                return data.decode('utf-8')

    local = plot_html_path if plot_html_path else ''
    if local and os.path.isfile(local):
        try:
            with open(local, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            logger.warning('Failed to load plot HTML from local: %s', e)

    return None


# --------------------------------------------------------------------------
# Plot images (PNG)
# --------------------------------------------------------------------------

def save_plot_image(image_b64_or_bytes, user_id: str, plot_id: str,
                    metadata: Optional[Dict[str, str]] = None) -> str:
    """Save a plot PNG image. Returns a storage reference string."""
    if isinstance(image_b64_or_bytes, str):
        img_bytes = base64.b64decode(image_b64_or_bytes)
    else:
        img_bytes = image_b64_or_bytes

    if _using_restfs():
        key = restfs.save_plot_image(img_bytes, user_id, plot_id, metadata)
        if key:
            return f"restfs://{restfs.BUCKET_PLOTS}/{key}"

    base = _user_images_dir(user_id)
    path = os.path.join(base, f'{plot_id}.png')
    try:
        with open(path, 'wb') as f:
            f.write(img_bytes)
        return path
    except Exception as e:
        logger.warning('Failed to save plot image locally: %s', e)
        return ''


def load_plot_image(ref: str, user_id: str = '',
                    plot_id: str = '') -> Optional[bytes]:
    if ref and ref.startswith('restfs://') and _using_restfs():
        parts = ref.replace('restfs://', '').split('/', 1)
        if len(parts) == 2:
            return restfs.get_bytes(parts[0], parts[1])

    if ref and os.path.isfile(ref):
        try:
            with open(ref, 'rb') as f:
                return f.read()
        except Exception:
            pass

    if user_id and plot_id:
        path = os.path.join(_user_images_dir(user_id), f'{plot_id}.png')
        if os.path.isfile(path):
            try:
                with open(path, 'rb') as f:
                    return f.read()
            except Exception:
                pass

    return None


def load_plot_image_b64(ref: str, user_id: str = '',
                        plot_id: str = '') -> Optional[str]:
    raw = load_plot_image(ref, user_id, plot_id)
    if raw is None:
        return None
    return base64.b64encode(raw).decode('ascii')


# --------------------------------------------------------------------------
# Plot registry (JSON manifest per user)
# --------------------------------------------------------------------------

def load_registry(user_id: str) -> list:
    if _using_restfs():
        data = restfs.get_metadata_json(user_id, 'plots_registry')
        if data is not None:
            return data.get('plots', [])

    path = _user_registry_path(user_id)
    if not os.path.isfile(path):
        return []
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data.get('plots', [])
    except Exception as e:
        logger.warning('Failed to load plots registry: %s', e)
        return []


def save_registry(user_id: str, plots: list) -> bool:
    payload = {'plots': plots}

    if _using_restfs():
        restfs.save_metadata_json(user_id, 'plots_registry', payload)

    path = _user_registry_path(user_id)
    try:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(payload, f, indent=2, default=str)
        return True
    except Exception as e:
        logger.warning('Failed to save plots registry: %s', e)
        return False


# --------------------------------------------------------------------------
# Cleanup
# --------------------------------------------------------------------------

def delete_plot_file(plot_html_path: str) -> bool:
    if plot_html_path and plot_html_path.startswith('restfs://') and _using_restfs():
        parts = plot_html_path.replace('restfs://', '').split('/', 1)
        if len(parts) == 2:
            restfs.delete_object(parts[0], parts[1])
            return True

    if not plot_html_path or not os.path.isfile(plot_html_path):
        return True
    try:
        os.remove(plot_html_path)
        return True
    except Exception as e:
        logger.warning('Failed to delete plot file: %s', e)
        return False


def delete_plot_image_file(ref: str) -> bool:
    if ref and ref.startswith('restfs://') and _using_restfs():
        parts = ref.replace('restfs://', '').split('/', 1)
        if len(parts) == 2:
            restfs.delete_object(parts[0], parts[1])
            return True

    if ref and os.path.isfile(ref):
        try:
            os.remove(ref)
            return True
        except Exception:
            pass
    return True
