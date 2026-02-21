"""
Standardized plot metadata schema for saved plots.
Used for plot registry, listing, retrieval, and future restFES/Keycloak integration.
"""

import uuid
from typing import Dict, Any, List, Optional
from datetime import datetime


# Field names and types for validation/documentation
PLOT_METADATA_FIELDS = {
    'plot_id': str,
    'user_id': str,
    'dataset_id': str,
    'file_path': str,
    'file_name': str,
    'plot_type': str,
    'data_mode': str,
    'query': str,
    'title': str,
    'description': str,
    'plot_spec': dict,
    'plot_summary': dict,
    'storage_location': str,
    'plot_html_path': str,
    'tags': list,
    'created_at': str,
    'updated_at': str,
    'data_metrics': dict,
}


def build_plot_metadata(
    user_id: str,
    plot_type: str,
    query: str,
    plot_spec: Dict[str, Any],
    plot_summary: Dict[str, Any],
    plot_html: str = '',
    dataset_id: str = '',
    file_path: str = '',
    file_name: str = '',
    data_mode: str = 'image',
    tags: Optional[List[str]] = None,
    plot_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Build a full plot metadata dict conforming to the schema.
    """
    now = datetime.utcnow().isoformat() + 'Z'
    return {
        'plot_id': plot_id or uuid.uuid4().hex,
        'user_id': user_id,
        'dataset_id': dataset_id or '',
        'file_path': file_path or '',
        'file_name': file_name or '',
        'plot_type': plot_type,
        'data_mode': data_mode,
        'query': query,
        'title': (plot_summary or {}).get('title', f'Plot: {query[:50]}'),
        'description': (plot_summary or {}).get('description', ''),
        'plot_spec': plot_spec or {},
        'plot_summary': plot_summary or {},
        'storage_location': 'local/shared_volume',
        'plot_html_path': '',  # Set by storage layer when HTML is saved
        'tags': list(tags) if tags else [],
        'created_at': now,
        'updated_at': now,
        'data_metrics': _extract_data_metrics(plot_summary),
    }


def _extract_data_metrics(plot_summary: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Extract searchable/filterable metrics from plot_summary.data."""
    if not plot_summary or not isinstance(plot_summary, dict):
        return {}
    data = plot_summary.get('data') or {}
    if not isinstance(data, dict):
        return {}
    # Include keys useful for filtering (class counts, split info, etc.)
    return {
        k: v for k, v in data.items()
        if k in (
            'class_counts', 'class_percentages', 'split_counts', 'split_percentages',
            'total_images', 'num_classes', 'imbalance_ratio', 'row_count', 'plot_type'
        )
    }
