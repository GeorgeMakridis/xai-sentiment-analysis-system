"""
AI Service Adapter — standardised contract for plugging in external AI/ML
services behind the XAI dashboard.

Any model service that implements the three endpoints described in this module
can be used as a prediction backend.  The XAI service will call the adapter to
get predictions + optional explanations, then generate its own visualisations
and feed everything into the RAG pipeline.

Usage
-----
1. Set the env var  AI_ADAPTER_URL  to the base URL of your model service
   (default: internal sklearn/TF-IDF mock).
2. The model service must expose:
       POST /predict
       POST /explain   (optional)
       GET  /model-info
3. If  AI_ADAPTER_URL  is not set, the built-in mock adapter is used.

See ``ADAPTER_CONTRACT`` below for the full request/response schemas.
"""

import logging
import os
from typing import Any, Dict, List, Optional

import requests

logger = logging.getLogger(__name__)

AI_ADAPTER_URL = os.environ.get("AI_ADAPTER_URL", "")
AI_ADAPTER_TIMEOUT = int(os.environ.get("AI_ADAPTER_TIMEOUT", "60"))

# ─── Contract documentation ─────────────────────────────────────────────────
ADAPTER_CONTRACT = {
    "predict": {
        "method": "POST",
        "path": "/predict",
        "request": {
            "data": "list[str] | list[list[float]]",
            "data_type": "text | tabular | image | timeseries",
            "options": {
                "return_probabilities": "bool (default true)",
            },
        },
        "response": {
            "predictions": "list[str | int | float]",
            "confidence": "list[float]   (0-1 per sample)",
            "labels": "list[str]        (class label names)",
            "model_id": "str            (identifier for provenance)",
        },
    },
    "explain": {
        "method": "POST",
        "path": "/explain",
        "note": "Optional. If not implemented, XAI service runs its own LIME/SHAP.",
        "request": {
            "data": "list[str] | list[list[float]]",
            "instance_index": "int",
            "method": "lime | shap | gradcam | attention | auto",
        },
        "response": {
            "feature_importances": "list[tuple[str, float]]",
            "attention_weights": "list[tuple[str, float]]  (transformers only)",
            "method_used": "str",
            "confidence_score": "float",
        },
    },
    "model_info": {
        "method": "GET",
        "path": "/model-info",
        "response": {
            "model_type": "str           (e.g. FinBERT, ResNet, XGBoost)",
            "model_id": "str",
            "version": "str",
            "supported_data_types": "list[str]",
            "xai_methods": "list[str]    (methods the service can run itself)",
            "description": "str",
        },
    },
}


# ─── Helper: call remote adapter ────────────────────────────────────────────

def _call(path: str, method: str = "POST",
          json_body: Optional[Dict] = None) -> Optional[Dict]:
    """Call the external AI adapter.  Returns parsed JSON or None on error."""
    if not AI_ADAPTER_URL:
        return None
    url = f"{AI_ADAPTER_URL.rstrip('/')}{path}"
    try:
        if method == "GET":
            resp = requests.get(url, timeout=AI_ADAPTER_TIMEOUT)
        else:
            resp = requests.post(url, json=json_body or {},
                                 timeout=AI_ADAPTER_TIMEOUT)
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        logger.warning("AI adapter call %s %s failed: %s", method, path, exc)
        return None


# ─── Public API ──────────────────────────────────────────────────────────────

def is_external_adapter_configured() -> bool:
    """True when an external AI_ADAPTER_URL is set."""
    return bool(AI_ADAPTER_URL)


def get_model_info() -> Dict[str, Any]:
    """Return model metadata from the external adapter or the built-in mock."""
    result = _call("/model-info", method="GET")
    if result:
        return result
    return {
        "model_type": "sklearn-TF-IDF-RF (built-in mock)",
        "model_id": "builtin-mock-v1",
        "version": "1.0.0",
        "supported_data_types": ["text", "tabular", "timeseries"],
        "xai_methods": ["lime", "shap"],
        "description": (
            "Built-in mock model using TF-IDF + RandomForest.  "
            "Set AI_ADAPTER_URL to connect a real model service."
        ),
    }


def predict(data: List[Any], data_type: str = "text",
            return_probabilities: bool = True) -> Optional[Dict[str, Any]]:
    """
    Get predictions from the external adapter.

    Returns
    -------
    dict with keys: predictions, confidence, labels, model_id
    None if the external adapter is not configured or fails
         (caller should fall back to the built-in model).
    """
    result = _call("/predict", json_body={
        "data": data,
        "data_type": data_type,
        "options": {"return_probabilities": return_probabilities},
    })
    return result


def explain(data: List[Any], instance_index: int = 0,
            method: str = "auto") -> Optional[Dict[str, Any]]:
    """
    Get explanations from the external adapter.

    Returns
    -------
    dict with keys: feature_importances, attention_weights,
                    method_used, confidence_score
    None if the adapter doesn't implement /explain or is not configured
         (caller should run its own LIME/SHAP).
    """
    result = _call("/explain", json_body={
        "data": data,
        "instance_index": instance_index,
        "method": method,
    })
    return result


# ─── Built-in mock adapter (used when AI_ADAPTER_URL is not set) ────────────

def mock_predict_text(texts: List[str]) -> Dict[str, Any]:
    """
    Quick sentiment prediction using the built-in TF-IDF + RandomForest
    pipeline.  This is the legacy behaviour from create_finbert_sentiment_model.
    """
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.ensemble import RandomForestClassifier
        import numpy as np

        vectoriser = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
        X = vectoriser.fit_transform(texts)

        # Without training data we can only return dummy predictions
        # The real path is to call train-model first, then run-xai
        labels = ["positive", "neutral", "negative"]
        n = len(texts)
        predictions = [labels[i % 3] for i in range(n)]
        confidence = [0.5 + 0.1 * (i % 5) for i in range(n)]

        return {
            "predictions": predictions,
            "confidence": confidence,
            "labels": labels,
            "model_id": "builtin-mock-v1",
        }
    except Exception as exc:
        logger.warning("mock_predict_text failed: %s", exc)
        return {
            "predictions": [],
            "confidence": [],
            "labels": [],
            "model_id": "builtin-mock-v1",
        }


# ─── Unified entry point used by xai_service routes ─────────────────────────

def get_predictions(data: List[Any], data_type: str = "text") -> Dict[str, Any]:
    """
    Try external adapter first, fall back to built-in mock.
    Always returns a dict with predictions/confidence/labels/model_id.
    """
    if is_external_adapter_configured():
        result = predict(data, data_type)
        if result and result.get("predictions"):
            return result

    # Fallback
    if data_type == "text":
        return mock_predict_text(data)

    return {
        "predictions": [],
        "confidence": [],
        "labels": [],
        "model_id": "builtin-mock-v1",
    }


def get_explanations(data: List[Any], instance_index: int = 0,
                     method: str = "auto") -> Optional[Dict[str, Any]]:
    """
    Try external adapter first, return None if unavailable
    (caller should run its own LIME/SHAP).
    """
    if is_external_adapter_configured():
        return explain(data, instance_index, method)
    return None
