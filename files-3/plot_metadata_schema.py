"""
Standardised plot metadata schema for the XAI platform.

Every plot persisted to RustFS gets a companion JSON sidecar built by
``build_plot_metadata()``.  The ``summary_for_rag`` section is the SINGLE
source of text that gets embedded into the vector DB — this guarantees
consistency and makes rehydration on restart trivial.

Changes from v1
----------------
- Added ``summary_for_rag``   (text, keywords, numeric_facts)
- Added ``provenance``         (model, xai_method, dataset_shape, duration)
- Added ``storage``            (image_ref, html_ref, metadata_ref)
- Added ``dataset_id``         linking to the source dataset
- Removed duplicate fields (storage_location, plot_html_path merged into storage)
"""

import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional


# ─── Schema field reference ──────────────────────────────────────────────────

PLOT_METADATA_FIELDS = {
    # Identity
    "plot_id":      str,
    "user_id":      str,
    "dataset_id":   str,
    "created_at":   str,   # ISO-8601
    "updated_at":   str,

    # Classification
    "plot_type":    str,   # e.g. word_sentiment_association, feature_importance
    "data_mode":    str,   # text | image | tabular | timeseries
    "title":        str,
    "description":  str,

    # Storage references  (restfs:// URIs or local paths)
    "storage": {
        "image_ref":    str,
        "html_ref":     str,
        "metadata_ref": str,
    },

    # RAG-optimised content  — this is what gets embedded
    "summary_for_rag": {
        "text":          str,           # pre-written natural-language summary
        "keywords":      List[str],     # for keyword fallback search
        "numeric_facts": Dict[str, Any],  # exact numbers the chatbot can cite
    },

    # Provenance — how this plot was created
    "provenance": {
        "model":           str,         # e.g. FinBERT, RandomForest
        "xai_method":      str,         # lime, shap, attention, gradcam
        "dataset_shape":   List[int],   # [rows, cols]
        "analysis_duration_seconds": float,
    },

    # Legacy compat
    "plot_spec":     dict,
    "plot_summary":  dict,
    "tags":          list,
    "data_metrics":  dict,
}


# ─── Builder ─────────────────────────────────────────────────────────────────

def build_plot_metadata(
    user_id: str,
    plot_type: str,
    title: str = "",
    description: str = "",
    *,
    # RAG summary
    summary_text: str = "",
    keywords: Optional[List[str]] = None,
    numeric_facts: Optional[Dict[str, Any]] = None,
    # Provenance
    model: str = "",
    xai_method: str = "",
    dataset_shape: Optional[List[int]] = None,
    analysis_duration: float = 0.0,
    # Storage
    image_ref: str = "",
    html_ref: str = "",
    metadata_ref: str = "",
    # Identity
    plot_id: Optional[str] = None,
    dataset_id: str = "",
    data_mode: str = "text",
    # Legacy / extra
    query: str = "",
    plot_spec: Optional[Dict] = None,
    plot_summary: Optional[Dict] = None,
    tags: Optional[List[str]] = None,
    file_path: str = "",
    file_name: str = "",
) -> Dict[str, Any]:
    """
    Build a complete metadata dict for one plot.

    This is the SINGLE constructor — all code paths must use this function
    instead of assembling metadata dicts ad-hoc.
    """
    now = datetime.utcnow().isoformat() + "Z"
    pid = plot_id or uuid.uuid4().hex[:12]

    # Auto-generate summary_text from plot_summary if not provided
    if not summary_text and plot_summary:
        summary_text = _auto_summary(plot_type, title, plot_summary)

    # Auto-extract keywords
    if not keywords:
        keywords = _auto_keywords(plot_type, title, description, plot_summary)

    # Auto-extract numeric facts
    if not numeric_facts and plot_summary:
        numeric_facts = _extract_numeric_facts(plot_summary)

    return {
        "plot_id": pid,
        "user_id": user_id,
        "dataset_id": dataset_id,
        "created_at": now,
        "updated_at": now,

        "plot_type": plot_type,
        "data_mode": data_mode,
        "title": title or f"Plot: {plot_type}",
        "description": description,

        "storage": {
            "image_ref": image_ref,
            "html_ref": html_ref,
            "metadata_ref": metadata_ref,
        },

        "summary_for_rag": {
            "text": summary_text,
            "keywords": keywords or [],
            "numeric_facts": numeric_facts or {},
        },

        "provenance": {
            "model": model,
            "xai_method": xai_method,
            "dataset_shape": dataset_shape or [],
            "analysis_duration_seconds": analysis_duration,
        },

        # Legacy compat
        "plot_spec": plot_spec or {},
        "plot_summary": plot_summary or {},
        "tags": tags or [],
        "data_metrics": _extract_data_metrics(plot_summary),

        # File info (for local fallback)
        "file_path": file_path,
        "file_name": file_name,
        "query": query,
    }


# ─── Vector DB text builder ──────────────────────────────────────────────────

def metadata_to_vector_text(meta: Dict[str, Any]) -> str:
    """
    Convert a metadata dict to the single text string that gets embedded
    in the vector DB.  This is the ONLY function that should produce
    embeddings input — no more ad-hoc doc strings scattered in app.py.
    """
    parts = []

    title = meta.get("title", "")
    if title:
        parts.append(f"{title}.")

    plot_type = meta.get("plot_type", "")
    if plot_type:
        parts.append(f"Plot type: {plot_type}.")

    rag = meta.get("summary_for_rag", {})
    rag_text = rag.get("text", "")
    if rag_text:
        parts.append(rag_text)

    # Include numeric facts as natural sentences
    facts = rag.get("numeric_facts", {})
    if facts:
        fact_strs = [f"{k}: {v}" for k, v in facts.items()
                     if v is not None and v != ""]
        if fact_strs:
            parts.append("Key metrics: " + ", ".join(fact_strs) + ".")

    desc = meta.get("description", "")
    if desc and desc != rag_text:
        parts.append(f"Description: {desc}")

    prov = meta.get("provenance", {})
    model = prov.get("model", "")
    xai = prov.get("xai_method", "")
    if model:
        parts.append(f"Model: {model}.")
    if xai:
        parts.append(f"XAI method: {xai}.")

    return " ".join(parts)


def metadata_to_vector_meta(meta: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build the metadata dict stored alongside the embedding in the vector DB.
    """
    rag = meta.get("summary_for_rag", {})
    prov = meta.get("provenance", {})
    return {
        "doc_type": "plot_summary",
        "plot_id": meta.get("plot_id", ""),
        "plot_type": meta.get("plot_type", ""),
        "title": meta.get("title", ""),
        "description": meta.get("description", ""),
        "summary_text": rag.get("text", ""),
        "keywords": rag.get("keywords", []),
        "numeric_facts": rag.get("numeric_facts", {}),
        "model": prov.get("model", ""),
        "xai_method": prov.get("xai_method", ""),
        "data_mode": meta.get("data_mode", ""),
        "dataset_id": meta.get("dataset_id", ""),
        "source": "plot_metadata",
        "timestamp": meta.get("created_at", ""),
    }


# ─── Auto-generation helpers ─────────────────────────────────────────────────

def _auto_summary(plot_type: str, title: str,
                  plot_summary: Dict[str, Any]) -> str:
    """Generate a natural-language summary from structured plot_summary data."""
    parts = []
    data = plot_summary.get("data", {})
    desc = plot_summary.get("description", "")

    if title:
        parts.append(f"{title} analysis")
    if desc:
        parts.append(desc)

    # Type-specific summaries
    if plot_type in ("word_sentiment_association", "sentiment_distribution"):
        if isinstance(data, dict):
            pos = data.get("top_positive_words", [])
            neg = data.get("top_negative_words", [])
            if pos:
                top = pos[:3] if isinstance(pos[0], str) else [p[0] for p in pos[:3]]
                parts.append(f"Top positive drivers: {', '.join(str(w) for w in top)}")
            if neg:
                top = neg[:3] if isinstance(neg[0], str) else [p[0] for p in neg[:3]]
                parts.append(f"Top negative drivers: {', '.join(str(w) for w in top)}")

    elif plot_type in ("feature_importance", "shap_summary"):
        if isinstance(data, dict):
            features = data.get("top_features", data.get("features", []))
            if features:
                top = features[:5] if isinstance(features[0], str) \
                    else [f[0] for f in features[:5]]
                parts.append(f"Most important features: {', '.join(str(f) for f in top)}")

    elif plot_type in ("class_distribution", "data_overview"):
        if isinstance(data, dict):
            n_classes = data.get("num_classes", data.get("n_classes"))
            total = data.get("total_samples", data.get("total_images"))
            if n_classes:
                parts.append(f"Number of classes: {n_classes}")
            if total:
                parts.append(f"Total samples: {total}")

    if not parts:
        parts.append(f"{plot_type} visualization generated")

    return ". ".join(parts) + "."


def _auto_keywords(plot_type: str, title: str, description: str,
                   plot_summary: Optional[Dict] = None) -> List[str]:
    """Extract search keywords from available metadata."""
    kw = set()

    # From plot_type
    for word in plot_type.replace("_", " ").split():
        if len(word) > 2:
            kw.add(word.lower())

    # From title
    for word in title.lower().split():
        if len(word) > 3:
            kw.add(word)

    # Standard domain keywords
    type_keywords = {
        "sentiment": ["sentiment", "positive", "negative", "neutral"],
        "feature_importance": ["feature", "importance", "ranking"],
        "shap": ["shap", "shapley", "explanation"],
        "lime": ["lime", "explanation", "local"],
        "attention": ["attention", "tokens", "weights"],
        "distribution": ["distribution", "histogram", "frequency"],
        "correlation": ["correlation", "heatmap", "relationship"],
        "confusion": ["confusion", "matrix", "accuracy"],
    }
    for key, words in type_keywords.items():
        if key in plot_type.lower():
            kw.update(words)

    return sorted(kw)


def _extract_numeric_facts(plot_summary: Optional[Dict]) -> Dict[str, Any]:
    """Pull numeric facts from plot_summary.data for citation by the chatbot."""
    if not plot_summary or not isinstance(plot_summary, dict):
        return {}
    data = plot_summary.get("data", {})
    if not isinstance(data, dict):
        return {}

    # Collect any numeric or small-string values
    facts = {}
    useful_keys = {
        "accuracy", "precision", "recall", "f1_score", "auc",
        "total_samples", "total_images", "num_classes", "n_classes",
        "imbalance_ratio", "row_count", "column_count",
        "top_positive_word", "top_positive_score",
        "top_negative_word", "top_negative_score",
        "mean_confidence", "median_confidence",
        "class_counts", "split_counts",
    }
    for k, v in data.items():
        if k in useful_keys or isinstance(v, (int, float)):
            facts[k] = v
    return facts


def _extract_data_metrics(plot_summary: Optional[Dict]) -> Dict[str, Any]:
    """Legacy helper — kept for backward compat."""
    if not plot_summary or not isinstance(plot_summary, dict):
        return {}
    data = plot_summary.get("data", {})
    if not isinstance(data, dict):
        return {}
    return {
        k: v for k, v in data.items()
        if k in (
            "class_counts", "class_percentages", "split_counts",
            "split_percentages", "total_images", "num_classes",
            "imbalance_ratio", "row_count", "plot_type",
        )
    }
