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
            return jsonify({
                "message": "Data uploaded and ingested successfully",
                "file_path": file_path,
                "data_summary": result.get("data_summary", {}),
            })
        return jsonify({"error": "Ingestion failed"}), 500
    except Exception as e:
        logger.exception("upload_data failed")
        return jsonify({"error": f"Ingestion failed: {e}"}), 500


@app.route("/api/upload-model", methods=["POST"])
@login_required
@role_required("analyst", "admin")
def upload_model():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    filename = secure_filename(file.filename)
    file_path = os.path.join(MODELS_FOLDER, filename)
    file.save(file_path)

    try:
        resp = _proxy_post(f"{XAI_SERVICE_URL}/analyze", {
            "model_path": file_path,
            "user_id": _user_id(),
        })
        if resp.status_code != 200:
            return jsonify({"error": "Analysis failed"}), 500

        result_data = resp.json()

        # Store in AI outputs for RAG
        try:
            _proxy_post(f"{AI_OUTPUTS_SERVICE_URL}/store-results", {
                "user_id": _user_id(),
                "results": result_data,
            }, timeout=15)
        except Exception as e:
            logger.warning("Failed to store results in AI outputs: %s", e)

        return jsonify({"message": "Model uploaded and analysis completed",
                        "results": result_data})
    except Exception as e:
        logger.exception("upload_model failed")
        return jsonify({"error": f"Analysis failed: {e}"}), 500


# ── Results retrieval ────────────────────────────────────────────────────────

@app.route("/api/get-results")
@login_required
def get_results():
    try:
        resp = _proxy_get(f"{AI_OUTPUTS_SERVICE_URL}/results/{_user_id()}")
        if resp.status_code == 200:
            data = resp.json()
            if data.get("images"):
                return jsonify(data)
    except Exception as e:
        logger.warning("AI outputs results unavailable: %s", e)

    # Fallback: local result files
    try:
        results_dir = os.path.join(SHARED_DATA_DIR, "results")
        if os.path.isdir(results_dir):
            user_files = sorted(
                [f for f in os.listdir(results_dir)
                 if f.endswith(".json") and _user_id() in f]
            )
            if user_files:
                with open(os.path.join(results_dir, user_files[-1])) as fh:
                    return jsonify(json.load(fh))
    except Exception as e:
        logger.warning("Local results fallback failed: %s", e)

    return jsonify({"message": "No results found. Upload and analyse data first."})


# ── Chat ─────────────────────────────────────────────────────────────────────

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


# ── Model training ───────────────────────────────────────────────────────────

@app.route("/api/create-sp100-model", methods=["POST"])
@login_required
@role_required("analyst", "admin")
def create_sp100_model():
    data = request.json or {}
    model_type = data.get("model_type")
    if not model_type:
        return jsonify({"error": "No model type provided"}), 400

    sp100_path = os.path.join(SHARED_DATA_DIR, "uploads", "sp100_daily_prices.csv")
    if not os.path.exists(sp100_path):
        return jsonify({"error": "SP100 data not found"}), 404

    try:
        resp = _proxy_post(f"{XAI_SERVICE_URL}/create-model", {
            "data_path": sp100_path,
            "model_type": model_type,
            "user_id": _user_id(),
        })
        if resp.status_code == 200:
            return jsonify({
                "message": f"SP100 {model_type.replace('_', ' ').title()} model created",
                "results": resp.json(),
            })
        return jsonify({"error": "Model creation failed"}), 500
    except Exception as e:
        return jsonify({"error": f"Model creation failed: {e}"}), 500


@app.route("/api/train-model", methods=["POST"])
@login_required
@role_required("analyst", "admin")
def train_model():
    data = request.json or {}
    model_type = data.get("model_type")
    data_type = data.get("data_type")
    if not model_type or not data_type:
        return jsonify({"error": "Missing model_type or data_type"}), 400

    try:
        resp = _proxy_post(f"{XAI_SERVICE_URL}/train-model", {
            "model_type": model_type,
            "data_type": data_type,
            "user_id": _user_id(),
        })
        if resp.status_code == 200:
            return jsonify({
                "message": f"{model_type.replace('_', ' ').title()} model trained",
                "results": resp.json(),
            })
        return jsonify({"error": "Model training failed"}), 500
    except Exception as e:
        return jsonify({"error": f"Model training failed: {e}"}), 500


# ── Data statistics (plot generation) ────────────────────────────────────────

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
    data_type = data.get("data_type", "text")
    if not filename:
        return jsonify({"error": "No filename provided"}), 400

    file_path = os.path.join(UPLOAD_FOLDER, filename)
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


# ── Preprocessing ────────────────────────────────────────────────────────────

@app.route("/api/preprocess-data", methods=["POST"])
@login_required
@role_required("analyst", "admin")
def preprocess_data():
    data = request.json or {}
    data["user_id"] = _user_id()
    try:
        resp = _proxy_post(f"{XAI_SERVICE_URL}/preprocess-data", data)
        if resp.status_code == 200:
            return jsonify({"message": "Data preprocessed successfully",
                            "preprocessed_data": resp.json()})
        return jsonify({"error": "Data preprocessing failed"}), 500
    except Exception as e:
        return jsonify({"error": f"Data preprocessing failed: {e}"}), 500


# ── Enhanced XAI ─────────────────────────────────────────────────────────────

@app.route("/api/enhanced-xai", methods=["POST"])
@login_required
@role_required("analyst", "admin")
def enhanced_xai():
    data = request.json or {}
    data.setdefault("user_id", _user_id())
    if not data.get("model_path") or not data.get("data_path"):
        return jsonify({"error": "Missing model_path or data_path"}), 400
    try:
        resp = _proxy_post(f"{XAI_SERVICE_URL}/enhanced-xai", data)
        if resp.status_code == 200:
            return jsonify(resp.json())
        return jsonify({"error": "Enhanced XAI analysis failed"}), 500
    except Exception as e:
        return jsonify({"error": f"Enhanced XAI analysis failed: {e}"}), 500


# ── FinBERT download ─────────────────────────────────────────────────────────

@app.route("/api/download-finbert", methods=["POST"])
@login_required
@role_required("analyst", "admin")
def download_finbert():
    try:
        resp = _proxy_post(f"{XAI_SERVICE_URL}/download-finbert", {
            "user_id": _user_id(),
        })
        if resp.status_code == 200:
            return jsonify({"message": "FinBERT model downloaded successfully",
                            "model_info": resp.json().get("model_info", {})})
        return jsonify({"error": "FinBERT download failed"}), 500
    except Exception as e:
        return jsonify({"error": f"FinBERT download failed: {e}"}), 500


# ── MNIST download ───────────────────────────────────────────────────────────

@app.route("/api/download-mnist", methods=["POST"])
@login_required
@role_required("analyst", "admin")
def download_mnist():
    data = request.json or {}
    try:
        resp = _proxy_post(f"{XAI_SERVICE_URL}/download-mnist", {
            "user_id": _user_id(),
            "sample_size": data.get("sample_size", 1000),
        }, timeout=300)
        if resp.status_code == 200:
            return jsonify(resp.json())
        error_data = resp.json() if resp.content else {}
        return jsonify({"error": error_data.get("error", "MNIST download failed")}), resp.status_code
    except requests.exceptions.Timeout:
        return jsonify({"error": "MNIST download timed out. Try a smaller sample size."}), 504
    except Exception as e:
        return jsonify({"error": f"MNIST download failed: {e}"}), 500


# ── Health ───────────────────────────────────────────────────────────────────

@app.route("/health")
def health():
    return jsonify({"status": "healthy", "service": "dashboard"})


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=3000, debug=os.environ.get("FLASK_DEBUG", "0") == "1")
