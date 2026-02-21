"""
Storage abstraction for plot HTML and metadata.
Current: local shared_volume. Future: restFES or other backends.
"""

import os
import json
import logging
from typing import Optional

logger = logging.getLogger(__name__)

SHARED_DATA_DIR = os.environ.get('SHARED_DATA_DIR', '/app/shared_data')
PLOTS_BASE = os.path.join(SHARED_DATA_DIR, 'plots')
REGISTRY_FILENAME = 'plots_registry.json'


def _user_plots_dir(user_id: str) -> str:
    path = os.path.join(PLOTS_BASE, user_id)
    os.makedirs(path, exist_ok=True)
    return path


def _user_registry_path(user_id: str) -> str:
    results_dir = os.path.join(SHARED_DATA_DIR, 'results', user_id)
    os.makedirs(results_dir, exist_ok=True)
    return os.path.join(results_dir, REGISTRY_FILENAME)


def save_plot_html(plot_html: str, user_id: str, plot_id: str) -> str:
    """
    Save plot HTML to storage. Returns storage path or future restFES URL.
    Current: shared_volume/plots/{user_id}/{plot_id}.html
    """
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
    """
    Load plot HTML from storage.
    Current: read from local path. Future: support restFES URL.
    """
    if not plot_html_path:
        return None
    if storage_location and not storage_location.startswith('local'):
        # Future: fetch from restFES
        logger.warning('Non-local storage not implemented: %s', storage_location)
        return None
    try:
        if os.path.isfile(plot_html_path):
            with open(plot_html_path, 'r', encoding='utf-8') as f:
                return f.read()
    except Exception as e:
        logger.warning('Failed to load plot HTML: %s', e)
    return None


def load_registry(user_id: str) -> list:
    """Load plots registry for user. Returns list of plot metadata dicts."""
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
    """Save plots registry for user."""
    path = _user_registry_path(user_id)
    try:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({'plots': plots}, f, indent=2, default=str)
        return True
    except Exception as e:
        logger.warning('Failed to save plots registry: %s', e)
        return False


def delete_plot_file(plot_html_path: str) -> bool:
    """Delete plot HTML file if it exists."""
    if not plot_html_path or not os.path.isfile(plot_html_path):
        return True
    try:
        os.remove(plot_html_path)
        return True
    except Exception as e:
        logger.warning('Failed to delete plot file: %s', e)
        return False
