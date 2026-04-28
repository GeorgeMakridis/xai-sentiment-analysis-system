import json
import uuid
from datetime import datetime
from fastapi import FastAPI, HTTPException
from clients.storage import StorageClient
from clients.model import ModelClient
from schemas.common import (
    XAIResponse,
    UC1SaliencyRequest, UC1PrototypeRequest,
    UC2OcclusionRequest, UC2LIMERequest,
)
from xai.uc1_vision.saliency import run_saliency
from xai.uc2_nlp.occlusion import load_text_from_csv, run_occlusion
from xai.uc2_nlp.lime import run_lime
import config

app = FastAPI(title="xai-api")

storage = StorageClient()
model = ModelClient()

def make_result_key(prefix: str, method: str, user_id: str = "default",
                    sample_index: int = None, detection_index: int = None,
                    target_class: str = None, ext: str = "json") -> str:
    """Build a human-readable result filename."""
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    short_id = uuid.uuid4().hex[:8]

    parts = [user_id, method]
    if detection_index is not None:
        parts.append(f"det{detection_index}")
    if sample_index is not None:
        parts.append(f"sample{sample_index}")
    if target_class:
        parts.append(target_class.lower().replace(" ", "_"))
    parts.append(ts)
    parts.append(short_id)

    filename = "_".join(parts) + f".{ext}"
    return f"{prefix}/{filename}"


@app.on_event("startup")
def ensure_buckets():
    """Create buckets if they don't exist (needed for fresh RustFS)."""
    for bucket in [config.UC1_BUCKET, config.UC2_BUCKET, config.UC3_BUCKET]:
        try:
            if not storage._client.bucket_exists(bucket):
                storage._client.make_bucket(bucket)
                print(f"Created bucket: {bucket}")
        except Exception as e:
            print(f"Warning: could not ensure bucket {bucket}: {e}")


def parse_minio_ref(ref: str) -> tuple[str, str]:
    """Parse 'minio://bucket/path/to/object' -> (bucket, object_name)."""
    without_scheme = ref.removeprefix("minio://")
    bucket, _, object_name = without_scheme.partition("/")
    return bucket, object_name


# --- Health ---

@app.get("/health")
def health():
    return {"status": "ok"}


# --- UC1: Vision ---

@app.post("/uc1/saliency", response_model=XAIResponse)
def uc1_saliency(req: UC1SaliencyRequest):
    import base64

    # 1. Load image from MinIO
    bucket, object_name = parse_minio_ref(req.image_ref)
    image_bytes = storage.get_object_bytes(bucket, object_name)

    # 2. Run saliency
    heatmap_png, detection_info = run_saliency(
        image_bytes,
        detection_index=req.detection_index,
        model_predict=model.predict_uc1,
        patch_size=req.patch_size,
        stride=req.stride,
        iou_thres=req.iou_thres,
    )

    if heatmap_png is None:
        raise HTTPException(status_code=422, detail="No detections found in image")

    # 3. Store full result to MinIO
    result_key = make_result_key(
        prefix=f"{config.RESULTS_PREFIX}/uc1/saliency",
        method="saliency",
        user_id=req.user_id,
        detection_index=req.detection_index,
        ext="png",
    )
    storage.put_object_bytes(config.UC1_BUCKET, result_key, heatmap_png, content_type="image/png")

    # 4. Return heatmap as base64 + storage ref
    return XAIResponse(
        result=base64.b64encode(heatmap_png).decode(),
        storage_ref=storage.ref(config.UC1_BUCKET, result_key),
    )


@app.post("/uc1/prototypes", response_model=XAIResponse)
def uc1_prototypes(req: UC1PrototypeRequest):
    raise HTTPException(status_code=501, detail="Not implemented yet")


# --- UC2: NLP ---

@app.post("/uc2/occlusion", response_model=XAIResponse)
def uc2_occlusion(req: UC2OcclusionRequest):
    if req.text is not None:
        text = req.text
        true_sentiment = req.true_sentiment or ""
    else:
        if not req.text_ref:
            raise HTTPException(status_code=400, detail="Provide `text` or `text_ref` + sample_index")
        bucket, object_name = parse_minio_ref(req.text_ref)
        csv_bytes = storage.get_object_bytes(bucket, object_name)
        text, true_sentiment = load_text_from_csv(csv_bytes, req.sample_index)

    # 2. Run occlusion (model_predict is the ModelClient call)
    result = run_occlusion(text, model_predict=model.predict_uc2)

    # 3. Store full result to MinIO
    result_key = make_result_key(
        prefix=f"{config.RESULTS_PREFIX}/uc2/occlusion",
        method="occlusion",
        user_id=req.user_id,
        sample_index=req.sample_index,
        target_class=result.get("target_class", ""),
        ext="json",
    )
    full_result = {"text": text, "true_sentiment": true_sentiment, **result}
    storage.put_object_bytes(
        config.UC2_BUCKET, result_key,
        json.dumps(full_result).encode(),
        content_type="application/json",
    )

    # 4. Return all word scores + storage ref
    return XAIResponse(
        result={
            "target_class": result["target_class"],
            "baseline_score": result["baseline_score"],
            "word_scores": result["word_scores"],  # all words, sorted by |importance|
        },
        storage_ref=storage.ref(config.UC2_BUCKET, result_key),
    )


@app.post("/uc2/lime", response_model=XAIResponse)
def uc2_lime(req: UC2LIMERequest):
    if req.text is not None:
        text = req.text
        true_sentiment = req.true_sentiment or ""
    else:
        if not req.text_ref:
            raise HTTPException(status_code=400, detail="Provide `text` or `text_ref` + sample_index")
        bucket, object_name = parse_minio_ref(req.text_ref)
        csv_bytes = storage.get_object_bytes(bucket, object_name)
        text, true_sentiment = load_text_from_csv(csv_bytes, req.sample_index)

    # 2. Run LIME
    result = run_lime(
        text,
        model_predict=model.predict_uc2,
        num_features=req.num_features,
        num_samples=req.num_samples,
    )

    # 3. Store full result to MinIO
    result_key = make_result_key(
        prefix=f"{config.RESULTS_PREFIX}/uc2/lime",
        method="lime",
        user_id=req.user_id,
        sample_index=req.sample_index,
        target_class=result.get("target_class", ""),
        ext="json",
    )
    full_result = {"text": text, "true_sentiment": true_sentiment, **result}
    storage.put_object_bytes(
        config.UC2_BUCKET, result_key,
        json.dumps(full_result).encode(),
        content_type="application/json",
    )

    # 4. Return all word scores + storage ref
    return XAIResponse(
        result={
            "target_class": result["target_class"],
            "word_scores": result["word_scores"],  # top num_features words from LIME
        },
        storage_ref=storage.ref(config.UC2_BUCKET, result_key),
    )


# --- UC3: Telecom (placeholder) ---

@app.post("/uc3/sensitivity")
def uc3_sensitivity():
    raise HTTPException(status_code=501, detail="UC3 not implemented yet")
