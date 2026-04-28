import io
import base64
import numpy as np
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from PIL import Image
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from ultralytics import YOLO

FINBERT_PATH = "/models/finbert"
YOLO_PATH = "/models/yolo/best.pt"
FINBERT_LABELS = ["positive", "negative", "neutral"]

# --- Model loading ---

finbert_model = None
finbert_tokenizer = None
yolo_model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global finbert_model, finbert_tokenizer, yolo_model

    print("Loading FinBERT...")
    finbert_tokenizer = AutoTokenizer.from_pretrained(FINBERT_PATH)
    finbert_model = AutoModelForSequenceClassification.from_pretrained(FINBERT_PATH)
    finbert_model.eval()
    print("FinBERT ready.")

    print("Loading YOLO...")
    yolo_model = YOLO(YOLO_PATH)
    print("YOLO ready.")

    yield
    print("Shutting down.")


app = FastAPI(title="model-mock", lifespan=lifespan)


# --- UC1: YOLO (real model) ---

class UC1Request(BaseModel):
    images: list[str]  # base64-encoded images


class Detection(BaseModel):
    class_id: int
    class_name: str
    confidence: float
    bbox: list[float]  # [x1, y1, x2, y2] normalized [0-1]


class UC1Response(BaseModel):
    detections: list[list[Detection]]


@app.post("/predict/uc1", response_model=UC1Response)
def predict_uc1(req: UC1Request):
    detections = []
    for b64 in req.images:
        try:
            img_bytes = base64.b64decode(b64)
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            img_np = np.array(img)
            H, W = img_np.shape[:2]
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid base64 image")

        results = yolo_model.predict(img_np, verbose=False)
        res = results[0]

        image_detections = []
        if hasattr(res, "boxes") and len(res.boxes) > 0:
            boxes = res.boxes.xyxy.cpu().numpy()   # pixel coords [x1,y1,x2,y2]
            confs = res.boxes.conf.cpu().numpy()
            clss = res.boxes.cls.cpu().numpy().astype(int)

            for box, conf, cls_id in zip(boxes, confs, clss):
                x1, y1, x2, y2 = box
                image_detections.append(Detection(
                    class_id=int(cls_id),
                    class_name=yolo_model.names[int(cls_id)],
                    confidence=round(float(conf), 4),
                    bbox=[
                        round(x1 / W, 4), round(y1 / H, 4),
                        round(x2 / W, 4), round(y2 / H, 4),
                    ],
                ))
        detections.append(image_detections)
    return UC1Response(detections=detections)


# --- UC2: FinBERT (real model) ---

class UC2Request(BaseModel):
    texts: list[str]


class UC2Prediction(BaseModel):
    positive: float
    negative: float
    neutral: float


class UC2Response(BaseModel):
    predictions: list[UC2Prediction]


@app.post("/predict/uc2", response_model=UC2Response)
def predict_uc2(req: UC2Request):
    inputs = finbert_tokenizer(
        req.texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    )
    with torch.no_grad():
        logits = finbert_model(**inputs).logits
        probs = torch.softmax(logits, dim=-1).tolist()

    predictions = []
    for p in probs:
        # FinBERT label order: positive=0, negative=1, neutral=2
        predictions.append(UC2Prediction(positive=p[0], negative=p[1], neutral=p[2]))
    return UC2Response(predictions=predictions)


# --- UC3: Placeholder ---

@app.post("/predict/uc3")
def predict_uc3():
    raise HTTPException(status_code=501, detail="UC3 not implemented yet")


# --- Health ---

@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/models")
def list_models():
    """Return available models for the dashboard model selector."""
    models = []
    if yolo_model is not None:
        models.append({
            "id": "yolo-uc1",
            "name": "YOLO Object Detection",
            "use_case": "uc1",
            "type": "vision",
            "status": "loaded",
        })
    if finbert_model is not None:
        models.append({
            "id": "finbert-uc2",
            "name": "FinBERT Sentiment",
            "use_case": "uc2",
            "type": "nlp",
            "status": "loaded",
        })
    return {"models": models}
