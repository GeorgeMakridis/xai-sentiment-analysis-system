"""
Dashboard service — web UI, file management, proxy to XAI and AI Outputs.
Supports Keycloak OIDC (AUTH_MODE=keycloak) or legacy dev auth.
"""
from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from flask_cors import CORS
import os
import sys
import requests
import json
import logging
from datetime import datetime
import io
import uuid
from typing import Optional, Tuple
from werkzeug.utils import secure_filename
from werkzeug.middleware.proxy_fix import ProxyFix

# ── Auth module (shared across services) ─────────────────────────────────────
# Add parent dir to path so `auth` package is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from auth import init_auth, login_required, role_required, get_current_user

# ── App setup ────────────────────────────────────────────────────────────────
logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", os.urandom(32).hex())

# Behind TLS-terminating reverse proxies, trust X-Forwarded-Proto / Host so
# OAuth redirect_uri values use https:// when the browser uses HTTPS.
app.wsgi_app = ProxyFix(
    app.wsgi_app,
    x_for=1,
    x_proto=1,
    x_host=1,
    x_port=1,
    x_prefix=1,
)
app.config["PREFERRED_URL_SCHEME"] = os.environ.get("PREFERRED_URL_SCHEME", "http")

CORS(app)

# Initialise auth (registers /login, /logout, /callback routes)
init_auth(app)

# ── Configuration ────────────────────────────────────────────────────────────
XAI_SERVICE_URL = os.environ.get("XAI_SERVICE_URL", "http://xai_service:8000")
AI_OUTPUTS_SERVICE_URL = os.environ.get("AI_OUTPUTS_SERVICE_URL", "http://ai_outputs:8001")
SHARED_DATA_DIR = os.environ.get("SHARED_DATA_DIR", "/app/shared_data")
UPLOAD_FOLDER = os.path.join(SHARED_DATA_DIR, "uploads")
MODELS_FOLDER = os.path.join(SHARED_DATA_DIR, "models")
RESULTS_FOLDER = os.path.join(SHARED_DATA_DIR, "results")

for d in (UPLOAD_FOLDER, MODELS_FOLDER, RESULTS_FOLDER):
    os.makedirs(d, exist_ok=True)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _user_id() -> str:
    return session.get("user_id", "anonymous")


def _auth_headers() -> dict:
    """Forward the access token (Keycloak) or user-id header (legacy)."""
    token = session.get("access_token")
    if token:
        return {"Authorization": f"Bearer {token}"}
    return {"X-User-Id": _user_id()}


def _proxy_post(url: str, payload: dict, timeout: int = 60):
    """POST JSON to a backend service, forwarding auth."""
    return requests.post(url, json=payload, headers=_auth_headers(), timeout=timeout)


def _proxy_get(url: str, params: dict = None, timeout: int = 15):
    return requests.get(url, params=params, headers=_auth_headers(), timeout=timeout)


# ═════════════════════════════════════════════════════════════════════════════
# Routes
# ═════════════════════════════════════════════════════════════════════════════

@app.route("/")
@login_required
def index():
    user = get_current_user()
    return render_template(
        "index.html",
        username=user["user_id"],
        roles=user["roles"],
        restfs_bucket_datasets=os.environ.get("RESTFS_BUCKET_DATASETS", "xai-datasets"),
    )


# ── Data upload & ingestion ──────────────────────────────────────────────────

@app.route("/api/upload-data", methods=["POST"])
@login_required
@role_required("analyst", "admin")
def upload_data():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    data_type = request.form.get("data_type", "timeseries")
    filename = secure_filename(file.filename)
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(file_path)

    try:
        resp = _proxy_post(f"{XAI_SERVICE_URL}/ingest", {
            "file_path": file_path,
            "user_id": _user_id(),
            "data_type": data_type,
        })
        if resp.status_code == 200:
            result = resp.json()

            # Persist dataset to RustFS so it survives container restarts
            try:
                _proxy_post(f"{AI_OUTPUTS_SERVICE_URL}/api/store-dataset", {
                    "user_id": _user_id(),
                    "filename": filename,
                }, timeout=30)
            except Exception as e:
                logger.warning("Failed to persist dataset to RustFS: %s", e)

            return jsonify({
                "message": "Data uploaded and ingested successfully",
                "file_path": file_path,
                "data_summary": result.get("data_summary", {}),
            })
        return jsonify({"error": "Ingestion failed"}), 500
    except Exception as e:
        logger.exception("upload_data failed")
        return jsonify({"error": f"Ingestion failed: {e}"}), 500


@app.route("/api/chat", methods=["POST"])
@login_required
def chat():
    question = (request.json or {}).get("question")
    if not question:
        return jsonify({"error": "No question provided"}), 400

    try:
        resp = _proxy_post(f"{AI_OUTPUTS_SERVICE_URL}/chat", {
            "question": question,
            "user_id": _user_id(),
        })
        if resp.status_code == 200:
            return jsonify(resp.json())
        if resp.status_code == 404:
            return jsonify(resp.json()), 404
        return jsonify({"error": "Chat failed"}), 500
    except Exception as e:
        return jsonify({"error": f"Chat failed: {e}"}), 500



@app.route("/api/data-statistics", methods=["POST"])
@login_required
def data_statistics():
    user_id = _user_id()
    try:
        resp = _proxy_post(f"{XAI_SERVICE_URL}/data-statistics", {
            "user_id": user_id,
        })
        if resp.status_code != 200:
            return jsonify({"error": "Failed to generate data statistics"}), 500

        result = resp.json()

        # Store each image in RustFS via ai_outputs and collect plot_ids
        images_with_refs = []
        for i, img_item in enumerate(result.get("images", [])):
            entry = img_item if isinstance(img_item, dict) else {"image": img_item}
            image_b64 = entry.get("image", "")
            plot_type = entry.get("type", f"plot_{i}")
            title = entry.get("title", plot_type)
            summary = entry.get("summary", "")
            plot_data = entry.get("data", {})

            if image_b64:
                try:
                    store_resp = _proxy_post(
                        f"{AI_OUTPUTS_SERVICE_URL}/api/store-plot-image",
                        {
                            "user_id": user_id,
                            "image": image_b64,
                            "plot_type": plot_type,
                            "title": title,
                            "description": summary,
                            "summary_text": summary,
                            "data": plot_data,
                        },
                        timeout=30,
                    )
                    if store_resp.status_code == 200:
                        resp_data = store_resp.json()
                        images_with_refs.append({
                            "type": plot_type,
                            "plot_id": resp_data.get("plot_id", ""),
                            "title": title,
                            "summary": summary,
                            "data": plot_data,
                        })
                        continue
                except Exception as e:
                    logger.warning("Failed to store plot in RustFS: %s", e)

            # Fallback: keep original entry
            images_with_refs.append(entry)

        # Store full results for RAG context
        try:
            _proxy_post(f"{AI_OUTPUTS_SERVICE_URL}/store-results", {
                "user_id": user_id,
                "results": {
                    "type": "data_statistics",
                    "images": result.get("images", []),
                    "data_type": result.get("data_type", "unknown"),
                    "insights": result.get("insights", {}),
                    "data_statistics": result.get("data_statistics", {}),
                    "plot_summaries": result.get("plot_summaries", []),
                    "timestamp": datetime.now().isoformat(),
                },
            }, timeout=15)
        except Exception as e:
            logger.warning("Failed to store data statistics for RAG: %s", e)

        return jsonify({
            "message": "Data statistics generated successfully",
            "images": images_with_refs,
            "data_type": result.get("data_type", "unknown"),
            "data_shape": result.get("data_shape"),
            "columns_count": result.get("columns_count"),
            "overview_stats": result.get("overview_stats", {}),
            "user_id": user_id,
        })

    except Exception as e:
        logger.exception("data_statistics failed")
        return jsonify({"error": f"Failed to generate data statistics: {e}"}), 500


# ── Plot image proxy ─────────────────────────────────────────────────────────

@app.route("/api/plot-image/<plot_id>")
@login_required
def proxy_plot_image(plot_id):
    try:
        resp = _proxy_get(
            f"{AI_OUTPUTS_SERVICE_URL}/api/plots/{plot_id}/image",
            params={"user_id": _user_id()},
        )
        if resp.status_code == 200:
            return resp.content, 200, {
                "Content-Type": "image/png",
                "Cache-Control": "public, max-age=3600",
            }
        return "", 404
    except Exception:
        return "", 502


# ── Dataset browser ──────────────────────────────────────────────────────────

@app.route("/api/datasets")
@login_required
def list_datasets():
    try:
        resp = _proxy_get(
            f"{AI_OUTPUTS_SERVICE_URL}/api/user-datasets/{_user_id()}"
        )
        if resp.status_code == 200:
            return jsonify(resp.json())
    except Exception:
        pass

    # Fallback: local scan
    datasets = []
    if os.path.isdir(UPLOAD_FOLDER):
        for fname in os.listdir(UPLOAD_FOLDER):
            fpath = os.path.join(UPLOAD_FOLDER, fname)
            if os.path.isfile(fpath) and fname.endswith((".csv", ".json", ".txt")):
                datasets.append({
                    "filename": fname,
                    "key": fpath,
                    "size": os.path.getsize(fpath),
                    "source": "local",
                })
    return jsonify({"datasets": datasets})


@app.route("/api/select-dataset", methods=["POST"])
@login_required
@role_required("analyst", "admin")
def select_dataset():
    data = request.json or {}
    filename = data.get("filename")
    source = data.get("source", "local")
    data_type = data.get("data_type", "text")
    if not filename:
        return jsonify({"error": "No filename provided"}), 400

    file_path = os.path.join(UPLOAD_FOLDER, filename)

    # If dataset lives in RustFS but not locally, fetch it first
    if source == "restfs" and not os.path.exists(file_path):
        try:
            fetch_resp = _proxy_post(
                f"{AI_OUTPUTS_SERVICE_URL}/api/fetch-dataset",
                {"user_id": _user_id(), "filename": filename},
                timeout=30,
            )
            if fetch_resp.status_code != 200:
                return jsonify({"error": f"Failed to fetch dataset from RustFS: {fetch_resp.text}"}), 500
            file_path = fetch_resp.json().get("file_path", file_path)
        except Exception as e:
            return jsonify({"error": f"Failed to fetch dataset from RustFS: {e}"}), 500

    if not os.path.exists(file_path):
        return jsonify({"error": f"Dataset not found: {filename}"}), 404

    try:
        resp = _proxy_post(f"{XAI_SERVICE_URL}/ingest", {
            "file_path": file_path,
            "user_id": _user_id(),
            "data_type": data_type,
        })
        if resp.status_code == 200:
            result = resp.json()
            return jsonify({
                "message": f'Dataset "{filename}" loaded successfully',
                "file_path": file_path,
                "data_summary": result.get("data_summary", {}),
            })
        return jsonify({"error": "Ingestion failed"}), 500
    except Exception as e:
        return jsonify({"error": f"Ingestion failed: {e}"}), 500


@app.route("/health")
def health():
    return jsonify({"status": "healthy", "service": "dashboard"})


# ═════════════════════════════════════════════════════════════════════════════
# Per-Sample XAI Routes (xai-api + model-mock integration)
# ═════════════════════════════════════════════════════════════════════════════

XAI_API_URL = os.environ.get("XAI_API_URL", "http://xai_api:8000")
MODEL_API_URL = os.environ.get("MODEL_API_URL", "http://model_mock:8003")


def _index_xai_for_chatbot(plot_type, title, summary_text, word_scores=None, extra_data=None):
    """Index a per-sample XAI result in ai_outputs Vector DB so the chatbot can reference it."""
    try:
        payload = {
            "user_id": _user_id(),
            "type": "per_sample_xai",
            "plot_summaries": [
                {
                    "plot_type": plot_type,
                    "title": title,
                    "description": summary_text,
                    "summary_text": summary_text,
                    "data": extra_data or {},
                }
            ],
        }
        if word_scores:
            payload["plot_summaries"][0]["data"]["word_scores"] = word_scores[:10]

        _proxy_post(
            f"{AI_OUTPUTS_SERVICE_URL}/store-results",
            payload,
            timeout=10,
        )
    except Exception as e:
        logger.warning("Failed to index XAI result for chatbot: %s", e)


def _psx_load_dataset_file(filename: str) -> Tuple[Optional[bytes], str]:
    """Load dataset bytes from shared uploads or RustFS via ai_outputs fetch."""
    if not filename:
        return None, "filename required"
    safe = secure_filename(filename)
    if not safe:
        return None, "invalid filename"
    path = os.path.join(UPLOAD_FOLDER, safe)
    if os.path.isfile(path):
        with open(path, "rb") as f:
            return f.read(), ""
    try:
        resp = _proxy_post(
            f"{AI_OUTPUTS_SERVICE_URL}/api/fetch-dataset",
            {"user_id": _user_id(), "filename": safe},
            timeout=90,
        )
        if resp.status_code != 200:
            err = (resp.json() or {}).get("error", resp.text)
            return None, str(err)
        fp = (resp.json() or {}).get("file_path")
        if fp and os.path.isfile(fp):
            with open(fp, "rb") as f:
                return f.read(), ""
    except Exception as e:
        logger.exception("fetch dataset for per-sample XAI")
        return None, str(e)
    return None, f"dataset not found: {safe}"


# ── Model listing ─────────────────────────────────────────────────────────

@app.route("/api/xai/models")
@login_required
def xai_list_models():
    """List available models from model-mock."""
    try:
        resp = requests.get(f"{MODEL_API_URL}/models", timeout=5)
        if resp.status_code == 200:
            return jsonify(resp.json())
    except Exception as e:
        logger.warning("model-mock /models unavailable: %s", e)
    return jsonify({"models": []})


# ── UC1: Vision (image listing, prediction, saliency) ────────────────────

@app.route("/api/xai/uc1/images")
@login_required
def xai_uc1_list_images():
    """List images: default UC1 robotics bucket, or current user's uploads in datasets bucket."""
    try:
        from minio import Minio
        endpoint = os.environ.get("RESTFS_ENDPOINT", "rustfs.extra-brain.unparallel.pt")
        access_key = os.environ.get("RESTFS_ACCESS_KEY", "")
        secret_key = os.environ.get("RESTFS_SECRET_KEY", "")
        secure = os.environ.get("RESTFS_SECURE", "true").lower() == "true"
        source = (request.args.get("source") or "default").strip()
        if source == "current":
            bucket = os.environ.get("RESTFS_BUCKET_DATASETS", "xai-datasets")
            prefix = f"{_user_id()}/"
        else:
            bucket = os.environ.get("UC1_BUCKET", "uc1-robotics")
            prefix = "train/images/"

        client = Minio(endpoint, access_key=access_key, secret_key=secret_key, secure=secure)
        images = []
        for obj in client.list_objects(bucket, prefix=prefix, recursive=True):
            if obj.object_name and not getattr(obj, "is_dir", False) and obj.object_name.lower().endswith(
                (".jpg", ".jpeg", ".png", ".gif", ".webp")
            ):
                images.append(obj.object_name)
        return jsonify({"images": images, "bucket": bucket, "prefix": prefix})
    except Exception as e:
        logger.warning("UC1 image listing failed: %s", e)
        return jsonify({"images": [], "error": str(e)})


@app.route("/api/xai/uc1/image/<path:object_name>")
@login_required
def xai_uc1_get_image(object_name):
    """Serve a UC1 image from RustFS. Query `bucket=` overrides default uc1-robotics."""
    try:
        from minio import Minio
        endpoint = os.environ.get("RESTFS_ENDPOINT", "rustfs.extra-brain.unparallel.pt")
        access_key = os.environ.get("RESTFS_ACCESS_KEY", "")
        secret_key = os.environ.get("RESTFS_SECRET_KEY", "")
        secure = os.environ.get("RESTFS_SECURE", "true").lower() == "true"
        bucket = request.args.get("bucket") or os.environ.get("UC1_BUCKET", "uc1-robotics")

        client = Minio(endpoint, access_key=access_key, secret_key=secret_key, secure=secure)
        resp = client.get_object(bucket, object_name)
        data = resp.read()
        resp.close()
        resp.release_conn()
        ct = "image/jpeg" if object_name.lower().endswith((".jpg", ".jpeg")) else "image/png"
        return data, 200, {"Content-Type": ct}
    except Exception as e:
        return jsonify({"error": str(e)}), 404


def _psx_collect_image_paths_from_dataset(raw: bytes, filename: str, n: int) -> list:
    """Extract image path strings from CSV/JSON dataset bytes."""
    import csv as csv_mod

    images = []
    name = (filename or "").lower()
    if name.endswith(".json"):
        payload = json.loads(raw.decode("utf-8"))
        if isinstance(payload, list):
            for i, item in enumerate(payload):
                if i >= n:
                    break
                if isinstance(item, str) and item.strip():
                    p = item.strip()
                    if p.startswith("data:image") or len(p) > 512:
                        continue
                    images.append({"index": i, "path": p, "label": os.path.basename(p)})
                elif isinstance(item, dict):
                    for key in ("image", "img", "file_path", "path", "filename", "image_path"):
                        if key in item and item[key]:
                            p = str(item[key]).strip()
                            if p.startswith("data:image") or len(p) > 512:
                                continue
                            images.append({"index": i, "path": p, "label": os.path.basename(p)})
                            break
        return images
    reader = csv_mod.DictReader(io.StringIO(raw.decode("utf-8")))
    if not reader.fieldnames:
        return images
    image_col = None
    for col in reader.fieldnames:
        cl = (col or "").lower()
        if any(t in cl for t in ("image", "img", "file_path", "path", "photo", "picture", "filename")):
            image_col = col
            break
    if image_col:
        for i, row in enumerate(reader):
            if i >= n:
                break
            img_path = (row.get(image_col) or "").strip()
            if not img_path or img_path.startswith("data:image") or len(img_path) > 512:
                continue
            images.append({"index": i, "path": img_path, "label": os.path.basename(img_path)})
    return images


@app.route("/api/xai/uc1/images-from-current")
@login_required
def xai_uc1_images_from_current():
    """List image paths from the user's current dataset (local uploads or RustFS fetch)."""
    try:
        filename = (request.args.get("filename") or "").strip()
        n = int(request.args.get("n", 50))
        if not filename:
            return jsonify({"images": [], "error": "No dataset filename provided"}), 400
        safe = secure_filename(filename)
        if not safe:
            return jsonify({"images": [], "error": "invalid filename"}), 400
        path = os.path.join(UPLOAD_FOLDER, safe)
        raw = None
        if os.path.isfile(path):
            with open(path, "rb") as f:
                raw = f.read()
        else:
            raw, err = _psx_load_dataset_file(filename)
            if raw is None:
                return jsonify({"images": [], "error": err or f"File not found: {safe}"}), 404
        images = _psx_collect_image_paths_from_dataset(raw, filename, n)
        if not images:
            uploads_dir = UPLOAD_FOLDER
            for fname in sorted(os.listdir(uploads_dir)):
                if fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp")):
                    images.append({"index": len(images), "path": fname, "label": fname})
                    if len(images) >= n:
                        break
        return jsonify({"images": images, "total": len(images), "filename": safe})
    except Exception as e:
        logger.warning("UC1 images-from-current failed: %s", e)
        return jsonify({"images": [], "error": str(e)}), 500


@app.route("/api/xai/uc1/uploaded-image/<path:filename>")
@login_required
def xai_uc1_serve_uploaded_image(filename):
    """Serve an image from the shared uploads directory (no path traversal)."""
    try:
        rel = (filename or "").replace("\\", "/").lstrip("/")
        parts = [secure_filename(p) for p in rel.split("/") if p and p not in ("..", ".")]
        if not parts:
            return jsonify({"error": "Invalid path"}), 400
        safe = parts[-1]
        if not safe.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp")):
            return jsonify({"error": "Not an image file"}), 400
        file_path = os.path.abspath(os.path.join(UPLOAD_FOLDER, *parts))
        base_abs = os.path.abspath(UPLOAD_FOLDER)
        if not file_path.startswith(base_abs + os.sep) and file_path != base_abs:
            return jsonify({"error": "Invalid path"}), 400
        if not os.path.isfile(file_path):
            for root, _dirs, files in os.walk(UPLOAD_FOLDER):
                if safe in files:
                    cand = os.path.join(root, safe)
                    if os.path.abspath(cand).startswith(base_abs + os.sep) or os.path.abspath(cand) == base_abs:
                        file_path = cand
                        break
        if not os.path.isfile(file_path):
            return jsonify({"error": f"Image not found: {safe}"}), 404
        with open(file_path, "rb") as f:
            data = f.read()
        ext = safe.rsplit(".", 1)[-1].lower()
        ct = {
            "jpg": "image/jpeg",
            "jpeg": "image/jpeg",
            "png": "image/png",
            "gif": "image/gif",
            "bmp": "image/bmp",
            "webp": "image/webp",
        }.get(ext, "application/octet-stream")
        return data, 200, {"Content-Type": ct}
    except Exception as e:
        logger.exception("serve uploaded UC1 image")
        return jsonify({"error": str(e)}), 500


def _read_uc1_image_bytes_from_ref(image_ref: str) -> Optional[bytes]:
    """Load UC1 image bytes from RustFS, else from shared uploads (same basename or relative path)."""
    from minio import Minio

    endpoint = os.environ.get("RESTFS_ENDPOINT", "rustfs.extra-brain.unparallel.pt")
    access_key = os.environ.get("RESTFS_ACCESS_KEY", "")
    secret_key = os.environ.get("RESTFS_SECRET_KEY", "")
    secure = os.environ.get("RESTFS_SECURE", "true").lower() == "true"
    bucket, obj_name = _parse_minio_ref(image_ref)
    try:
        client = Minio(endpoint, access_key=access_key, secret_key=secret_key, secure=secure)
        resp = client.get_object(bucket, obj_name)
        data = resp.read()
        resp.close()
        resp.release_conn()
        return data
    except Exception:
        pass
    uid = _user_id()
    base = secure_filename(os.path.basename(obj_name))
    parts = [secure_filename(p) for p in obj_name.replace("\\", "/").split("/") if p]
    rel_join = os.path.join(*parts) if parts else ""
    candidates = [
        os.path.join(UPLOAD_FOLDER, base) if base else None,
        os.path.join(UPLOAD_FOLDER, rel_join) if rel_join else None,
        os.path.join(UPLOAD_FOLDER, uid, base) if base else None,
    ]
    base_abs = os.path.abspath(UPLOAD_FOLDER)
    for p in candidates:
        if not p:
            continue
        ap = os.path.abspath(p)
        if not (ap.startswith(base_abs + os.sep) or ap == base_abs):
            continue
        if os.path.isfile(ap):
            with open(ap, "rb") as f:
                return f.read()
    return None


def _ensure_minio_ref_for_uc1_saliency(image_ref: str) -> str:
    """If the object exists in RustFS, return ref; else upload local bytes to a temp key for xai-api."""
    from minio import Minio

    endpoint = os.environ.get("RESTFS_ENDPOINT", "rustfs.extra-brain.unparallel.pt")
    access_key = os.environ.get("RESTFS_ACCESS_KEY", "")
    secret_key = os.environ.get("RESTFS_SECRET_KEY", "")
    secure = os.environ.get("RESTFS_SECURE", "true").lower() == "true"
    bucket, obj_name = _parse_minio_ref(image_ref)
    client = Minio(endpoint, access_key=access_key, secret_key=secret_key, secure=secure)
    try:
        client.stat_object(bucket, obj_name)
        return image_ref
    except Exception:
        pass
    b = _read_uc1_image_bytes_from_ref(image_ref)
    if b is None:
        raise ValueError("Could not resolve image bytes for saliency")
    uc1_bucket = os.environ.get("UC1_BUCKET", "uc1-robotics")
    lower = obj_name.lower()
    if lower.endswith(".png"):
        ext, ct = ".png", "image/png"
    elif lower.endswith(".webp"):
        ext, ct = ".webp", "image/webp"
    elif lower.endswith(".jpeg") or lower.endswith(".jpg"):
        ext, ct = ".jpg", "image/jpeg"
    else:
        ext, ct = ".jpg", "image/jpeg"
    key = f"{_user_id()}/xai_temp/{uuid.uuid4().hex}{ext}"
    client.put_object(
        uc1_bucket,
        key,
        io.BytesIO(b),
        length=len(b),
        content_type=ct,
    )
    return f"minio://{uc1_bucket}/{key}"


@app.route("/api/xai/uc1/predict", methods=["POST"])
@login_required
def xai_uc1_predict():
    """Run YOLO prediction on an image via model-mock."""
    try:
        import base64 as b64

        data = request.json or {}
        image_ref = data.get("image_ref", "")
        img_bytes = _read_uc1_image_bytes_from_ref(image_ref)
        if img_bytes is None:
            return jsonify({"error": "Could not load image from storage or uploads"}), 404

        b64_img = b64.b64encode(img_bytes).decode()
        model_resp = requests.post(
            f"{MODEL_API_URL}/predict/uc1",
            json={"images": [b64_img]},
            timeout=30,
        )
        model_resp.raise_for_status()
        return jsonify(model_resp.json())
    except Exception as e:
        logger.exception("UC1 predict failed")
        return jsonify({"error": str(e)}), 500


@app.route("/api/xai/uc1/saliency", methods=["POST"])
@login_required
def xai_uc1_saliency():
    """Run saliency XAI via xai-api and index result for RAG."""
    try:
        data = dict(request.json or {})
        data["user_id"] = _user_id()
        data["image_ref"] = _ensure_minio_ref_for_uc1_saliency(data.get("image_ref", ""))
        resp = requests.post(
            f"{XAI_API_URL}/uc1/saliency",
            json=data,
            timeout=120,  # saliency can be slow
        )
        resp.raise_for_status()
        result = resp.json()

        storage_ref = result.get("storage_ref", "")
        _index_xai_for_chatbot(
            plot_type="uc1_saliency",
            title=f"Saliency Heatmap — detection #{data.get('detection_index', 0)}",
            summary_text=(
                f"Per-sample occlusion-based saliency map for object detection. "
                f"Image: {data.get('image_ref', 'unknown')}. "
                f"Detection index: {data.get('detection_index', 0)}. "
                f"Patch size: {data.get('patch_size', 64)}, stride: {data.get('stride', 32)}, "
                f"IoU threshold: {data.get('iou_thres', 0.4)}. "
                f"The heatmap shows which image regions most affect the detection confidence. "
                f"Bright regions are critical for the detection. "
                f"Result stored at: {storage_ref}."
            ),
            extra_data={"method": "saliency", "image_ref": data.get("image_ref", ""),
                        "patch_size": data.get("patch_size", 64),
                        "storage_ref": storage_ref},
        )

        return jsonify(result)
    except Exception as e:
        logger.exception("UC1 saliency failed")
        return jsonify({"error": str(e)}), 500


# ── UC2: NLP (per-sample samples, prediction, occlusion, LIME) ───────────

@app.route("/api/xai/persample/samples")
@login_required
def xai_persample_samples():
    """Get samples from xai_service's in-memory data for per-sample XAI display."""
    try:
        n = request.args.get("n", 50)
        user_id = _user_id()

        resp = requests.get(
            f"{XAI_SERVICE_URL}/samples/{user_id}",
            params={"n": n},
            timeout=10,
        )
        if resp.status_code == 200:
            return jsonify(resp.json())
        elif resp.status_code == 404:
            return jsonify({"samples": [], "error": "No dataset ingested. Upload or select data first."})
        else:
            return jsonify({"samples": [], "error": f"xai_service error: {resp.text}"}), resp.status_code
    except Exception as e:
        logger.warning("persample samples failed: %s", e)
        return jsonify({"samples": [], "error": str(e)})


@app.route("/api/xai/persample/export-csv", methods=["POST"])
@login_required
def xai_persample_export_csv():
    """Export the user's ingested data as a flat CSV to RustFS so xai-api can read it."""
    try:
        user_id = _user_id()

        resp = requests.get(
            f"{XAI_SERVICE_URL}/samples/{user_id}",
            params={"n": 500},
            timeout=10,
        )
        if resp.status_code != 200:
            return jsonify({"error": "No data to export"}), 404

        data = resp.json()
        samples = data.get("samples", [])
        text_col = data.get("columns", {}).get("text")
        sent_col = data.get("columns", {}).get("sentiment")

        if not samples or not text_col:
            return jsonify({"error": "No text samples found in dataset"}), 400

        import csv as csv_mod

        csv_filename = f"_xai_export_{user_id}.csv"
        csv_path = os.path.join(UPLOAD_FOLDER, csv_filename)
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv_mod.writer(f)
            writer.writerow(["text", "finbert_sentiment"])
            for s in samples:
                writer.writerow([s.get("text", ""), s.get("sentiment", "unknown")])

        try:
            _proxy_post(
                f"{AI_OUTPUTS_SERVICE_URL}/api/store-dataset",
                {"user_id": user_id, "filename": csv_filename},
                timeout=15,
            )
        except Exception as e:
            logger.warning("Failed to upload export CSV to RustFS: %s", e)

        ds_bucket = os.environ.get("RESTFS_BUCKET_DATASETS", "xai-datasets")
        text_ref = f"minio://{ds_bucket}/{user_id}/{csv_filename}"

        return jsonify({
            "text_ref": text_ref,
            "total_samples": len(samples),
            "csv_filename": csv_filename,
        })
    except Exception as e:
        logger.exception("export-csv failed")
        return jsonify({"error": str(e)}), 500


@app.route("/api/xai/uc2/predict", methods=["POST"])
@login_required
def xai_uc2_predict():
    """Run FinBERT prediction on a text via model-mock."""
    try:
        data = request.json or {}
        text = data.get("text", "")
        resp = requests.post(
            f"{MODEL_API_URL}/predict/uc2",
            json={"texts": [text]},
            timeout=15,
        )
        resp.raise_for_status()
        return jsonify(resp.json())
    except Exception as e:
        logger.exception("UC2 predict failed")
        return jsonify({"error": str(e)}), 500


@app.route("/api/xai/uc2/occlusion", methods=["POST"])
@login_required
def xai_uc2_occlusion():
    """Run occlusion XAI via xai-api using the exported CSV in RustFS."""
    try:
        data = request.json or {}
        text_ref = data.get("text_ref", "")
        sample_index = data.get("sample_index", 0)

        if not text_ref:
            return jsonify({"error": "No text_ref provided. Export dataset first."}), 400

        resp = requests.post(
            f"{XAI_API_URL}/uc2/occlusion",
            json={"text_ref": text_ref, "sample_index": sample_index, "user_id": _user_id()},
            timeout=60,
        )
        resp.raise_for_status()
        result = resp.json()

        res = result.get("result", {}) if isinstance(result, dict) else {}
        ws = res.get("word_scores", [])
        top_words = ", ".join(f"{w['word']} ({w['importance']:+.3f})" for w in ws[:5])
        target = res.get("target_class", "unknown")
        baseline = res.get("baseline_score", "")
        _index_xai_for_chatbot(
            plot_type="uc2_occlusion",
            title=f"Occlusion XAI — sample #{data.get('sample_index', '?')}",
            summary_text=(
                f"Per-sample occlusion word importance analysis. "
                f"Target class: {target}. Baseline score: {baseline}. "
                f"Most important words: {top_words}. "
                f"Method: each word was masked one at a time, measuring the drop in predicted class score. "
                f"Positive importance means the word supports the prediction, negative means it opposes it."
            ),
            word_scores=ws,
            extra_data={"target_class": target, "baseline_score": baseline, "method": "occlusion"},
        )
        return jsonify(result)
    except Exception as e:
        logger.exception("UC2 occlusion failed")
        return jsonify({"error": str(e)}), 500


@app.route("/api/xai/uc2/lime", methods=["POST"])
@login_required
def xai_uc2_lime():
    """Run LIME XAI via xai-api using the exported CSV in RustFS."""
    try:
        data = request.json or {}
        text_ref = data.get("text_ref", "")
        sample_index = data.get("sample_index", 0)
        num_features = data.get("num_features", 10)
        num_samples = data.get("num_samples", 100)

        if not text_ref:
            return jsonify({"error": "No text_ref provided. Export dataset first."}), 400

        resp = requests.post(
            f"{XAI_API_URL}/uc2/lime",
            json={
                "text_ref": text_ref,
                "sample_index": sample_index,
                "num_features": num_features,
                "num_samples": num_samples,
                "user_id": _user_id(),
            },
            timeout=60,
        )
        resp.raise_for_status()
        result = resp.json()

        res = result.get("result", {}) if isinstance(result, dict) else {}
        ws = res.get("word_scores", [])
        top_words = ", ".join(f"{w['word']} ({w['importance']:+.3f})" for w in ws[:5])
        target = res.get("target_class", "unknown")
        _index_xai_for_chatbot(
            plot_type="uc2_lime",
            title=f"LIME XAI — sample #{data.get('sample_index', '?')}",
            summary_text=(
                f"Per-sample LIME word importance analysis. "
                f"Target class: {target}. "
                f"Most important words by LIME: {top_words}. "
                f"Method: LIME generated perturbed text versions, fit a local linear model, "
                f"and extracted the top contributing words. "
                f"Positive importance supports the predicted class, negative opposes it."
            ),
            word_scores=ws,
            extra_data={"target_class": target, "method": "lime",
                        "num_features": data.get("num_features", 10),
                        "num_samples": data.get("num_samples", 100)},
        )
        return jsonify(result)
    except Exception as e:
        logger.exception("UC2 LIME failed")
        return jsonify({"error": str(e)}), 500


# ── Helpers ───────────────────────────────────────────────────────────────

def _parse_minio_ref(ref: str):
    """Parse 'minio://bucket/path/to/object' -> (bucket, object_name)."""
    without_scheme = ref.removeprefix("minio://")
    bucket, _, object_name = without_scheme.partition("/")
    return bucket, object_name


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=3000, debug=os.environ.get("FLASK_DEBUG", "0") == "1")
