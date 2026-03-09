"""
Consolidated vector store operations for the AI Outputs service.

This module replaces the 13+ scattered ``store_plot_summary()`` calls
in ``app.py`` with a single, consistent pipeline:

    1. Build metadata via ``plot_metadata_schema.build_plot_metadata()``
    2. Generate embedding text via ``metadata_to_vector_text()``
    3. Store in vector DB via ``index_plot()``
    4. Optionally persist metadata JSON to RustFS

Usage in app.py
---------------
Replace every ``store_plot_summary(...)`` call with::

    from vector_store import index_plot
    index_plot(vector_db, user_id, {
        "title": ...,
        "plot_type": ...,
        "description": ...,
        "summary_text": ...,
        "data": ...,
    })

Or for full metadata control::

    from vector_store import index_plot_metadata
    meta = build_plot_metadata(user_id, plot_type, ...)
    index_plot_metadata(vector_db, user_id, meta)
"""

import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from plot_metadata_schema import (
    build_plot_metadata,
    metadata_to_vector_meta,
    metadata_to_vector_text,
)

logger = logging.getLogger(__name__)

# Try importing RustFS for metadata persistence
try:
    import restfs_client as restfs
    _has_restfs = True
except ImportError:
    _has_restfs = False


# ─── Primary API ─────────────────────────────────────────────────────────────

def index_plot(
    vector_db,
    user_id: str,
    summary: Dict[str, Any],
    *,
    source: str = "auto",
    plot_id: str = "",
    image_ref: str = "",
    model: str = "",
    xai_method: str = "",
    dataset_id: str = "",
    data_mode: str = "text",
    persist_metadata: bool = True,
) -> Dict[str, Any]:
    """
    Index a single plot/result in the vector DB.

    This is the ONE function to call from all code paths — data_statistics,
    store-results, store-plot-image, run-xai, etc.

    Parameters
    ----------
    vector_db : SimpleVectorDB
        The vector DB instance.
    user_id : str
    summary : dict
        Must contain at least ``title`` and ``plot_type``.
        Optional: ``description``, ``summary_text``, ``data``, ``metadata``.
    source : str
        Tag for debugging (e.g. 'data_statistics', 'xai_results').
    plot_id : str
        If provided, reuse this ID; otherwise auto-generated.
    image_ref : str
        RustFS URI if the plot image was already stored.
    persist_metadata : bool
        If True and RustFS is available, save the metadata JSON sidecar.

    Returns
    -------
    dict : The full metadata dict that was indexed.
    """
    title = summary.get("title", "Plot Summary")
    plot_type = summary.get("plot_type", summary.get("type", "unknown"))
    description = summary.get("description", "")
    summary_text = summary.get("summary_text", "")
    data = summary.get("data", {})
    extra_meta = summary.get("metadata", {})

    # Build standardised metadata
    meta = build_plot_metadata(
        user_id=user_id,
        plot_type=plot_type,
        title=title,
        description=description,
        summary_text=summary_text,
        plot_id=plot_id or None,
        image_ref=image_ref,
        model=model or extra_meta.get("model", ""),
        xai_method=xai_method or extra_meta.get("xai_method", ""),
        dataset_id=dataset_id,
        data_mode=data_mode,
        plot_summary={"data": data, "description": description,
                      "title": title, "metadata": extra_meta},
        keywords=summary.get("keywords"),
        numeric_facts=summary.get("numeric_facts"),
    )

    # Generate embedding text and metadata
    doc_text = metadata_to_vector_text(meta)
    doc_meta = metadata_to_vector_meta(meta)
    doc_meta["source"] = source

    # Store in vector DB
    try:
        vector_db.add_document(user_id, doc_text, doc_meta)
        logger.debug("Indexed plot %s for user %s (source=%s)",
                     meta["plot_id"], user_id, source)
    except Exception as exc:
        logger.warning("Failed to index plot in vector DB: %s", exc)

    # Persist metadata JSON to RustFS
    if persist_metadata:
        _persist_metadata(user_id, meta)

    return meta


def index_plot_metadata(
    vector_db,
    user_id: str,
    meta: Dict[str, Any],
    source: str = "auto",
) -> None:
    """
    Index a pre-built metadata dict.  Use when you've already called
    ``build_plot_metadata()`` yourself.
    """
    doc_text = metadata_to_vector_text(meta)
    doc_meta = metadata_to_vector_meta(meta)
    doc_meta["source"] = source

    try:
        vector_db.add_document(user_id, doc_text, doc_meta)
    except Exception as exc:
        logger.warning("Failed to index pre-built metadata: %s", exc)


# ─── Batch indexing ──────────────────────────────────────────────────────────

def index_plots_batch(
    vector_db,
    user_id: str,
    summaries: List[Dict[str, Any]],
    source: str = "batch",
    **kwargs,
) -> List[Dict[str, Any]]:
    """Index multiple plots at once.  Returns list of metadata dicts."""
    results = []
    for s in summaries:
        meta = index_plot(vector_db, user_id, s, source=source, **kwargs)
        results.append(meta)
    return results


# ─── Rehydration (startup) ──────────────────────────────────────────────────

def rehydrate_from_restfs(vector_db, user_id: str) -> int:
    """
    On startup (or after container restart), scan RustFS for persisted
    metadata JSONs and re-index them in the vector DB.

    Returns the number of plots rehydrated.
    """
    if not _has_restfs or not restfs.is_available():
        return 0

    count = 0
    try:
        objects = restfs.list_objects(restfs.BUCKET_METADATA,
                                     prefix=f"{user_id}/", recursive=True)
        for obj in objects:
            key = obj.get("key", "")
            if not key.endswith(".json") or key.endswith("plots_registry.json"):
                continue
            raw = restfs.get_bytes(restfs.BUCKET_METADATA, key)
            if raw is None:
                continue
            try:
                meta = json.loads(raw.decode("utf-8"))
                index_plot_metadata(vector_db, user_id, meta,
                                    source="rehydrated")
                count += 1
            except Exception:
                continue
    except Exception as exc:
        logger.warning("Rehydration failed for user %s: %s", user_id, exc)

    if count:
        logger.info("Rehydrated %d plots for user %s from RustFS", count, user_id)
    return count


# ─── Internal helpers ────────────────────────────────────────────────────────

def _persist_metadata(user_id: str, meta: Dict[str, Any]) -> None:
    """Save metadata JSON sidecar to RustFS (best-effort)."""
    if not _has_restfs or not restfs.is_available():
        return
    try:
        plot_id = meta.get("plot_id", "unknown")
        restfs.save_metadata_json(user_id, plot_id, meta)
    except Exception as exc:
        logger.debug("Could not persist metadata to RustFS: %s", exc)
