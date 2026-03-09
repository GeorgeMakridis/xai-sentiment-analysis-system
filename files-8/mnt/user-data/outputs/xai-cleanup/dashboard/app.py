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
from werkzeug.utils import secure_filename

# ── Auth module (shared across services) ─────────────────────────────────────
# Add parent dir to path so `auth` package is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from auth import init_auth, login_required, role_required, get_current_user

# ── App setup ────────────────────────────────────────────────────────────────
logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", os.urandom(32).hex())
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
    return render_template("index.html", username=user["user_id"],
                           roles=user["roles"])


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


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=3000, debug=os.environ.get("FLASK_DEBUG", "0") == "1")
