"""
Load YOLO/Roboflow or plain image-folder datasets into a manifest dataframe.
"""
from __future__ import annotations

import ast
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp", ".tif", ".tiff"}
YOLO_SPLITS = ("train", "valid", "val", "test")


def is_yolo_dataset(path: str) -> bool:
    return os.path.isdir(path) and os.path.isfile(os.path.join(path, "data.yaml"))


def is_image_folder(path: str) -> bool:
    """True when path is a directory containing images (YOLO layout or plain folder)."""
    if not os.path.isdir(path):
        return False
    if is_yolo_dataset(path):
        return True
    for _root, _dirs, files in os.walk(path):
        for fname in files:
            if os.path.splitext(fname)[1].lower() in IMAGE_EXTENSIONS:
                return True
    return False


def _parse_yolo_class_names(data_yaml_path: str) -> List[str]:
    with open(data_yaml_path, encoding="utf-8") as f:
        content = f.read()
    match = re.search(r"^names:\s*(.+)$", content, re.MULTILINE)
    if not match:
        return []
    raw = match.group(1).strip()
    try:
        names = ast.literal_eval(raw)
        if isinstance(names, list):
            return [str(n) for n in names]
    except (SyntaxError, ValueError):
        pass
    return []


def _parse_label_file(label_path: str, class_names: List[str]) -> List[str]:
    labels: List[str] = []
    if not os.path.isfile(label_path):
        return labels
    with open(label_path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            try:
                cls_id = int(parts[0])
            except ValueError:
                continue
            if class_names and 0 <= cls_id < len(class_names):
                name = class_names[cls_id]
            else:
                name = str(cls_id)
            if name not in labels:
                labels.append(name)
    return labels


def _rel_upload_path(full_path: str, uploads_root: str) -> str:
    uploads_root = os.path.abspath(uploads_root)
    full_path = os.path.abspath(full_path)
    if full_path.startswith(uploads_root + os.sep):
        return full_path[len(uploads_root) + 1 :].replace("\\", "/")
    return os.path.basename(full_path)


def _iter_image_records(
    folder_path: str,
    uploads_root: str,
    class_names: List[str],
) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    split_dirs: List[Tuple[str, str]] = []

    for split in YOLO_SPLITS:
        images_dir = os.path.join(folder_path, split, "images")
        if os.path.isdir(images_dir):
            normalized = "valid" if split == "val" else split
            split_dirs.append((normalized, images_dir))

    if not split_dirs:
        split_dirs.append(("all", folder_path))

    for split, images_dir in split_dirs:
        labels_dir = os.path.join(os.path.dirname(images_dir), "labels")
        for root, _dirs, files in os.walk(images_dir):
            for fname in sorted(files):
                ext = os.path.splitext(fname)[1].lower()
                if ext not in IMAGE_EXTENSIONS:
                    continue
                full_path = os.path.join(root, fname)
                rel_path = _rel_upload_path(full_path, uploads_root)
                label_names: List[str] = []
                if os.path.isdir(labels_dir):
                    stem = os.path.splitext(fname)[0]
                    label_path = os.path.join(labels_dir, f"{stem}.txt")
                    label_names = _parse_label_file(label_path, class_names)
                primary = label_names[0] if label_names else "unknown"
                records.append(
                    {
                        "image_path": rel_path,
                        "split": split,
                        "label": primary,
                        "label_names": ",".join(label_names),
                        "num_objects": len(label_names),
                    }
                )
    return records


def load_image_folder_dataset(
    folder_path: str,
    uploads_root: Optional[str] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Build a manifest dataframe for a YOLO export or plain image folder.

    Returns (dataframe, preprocessing_info).
    """
    folder_path = os.path.abspath(folder_path)
    if uploads_root is None:
        uploads_root = os.path.dirname(folder_path)
    uploads_root = os.path.abspath(uploads_root)

    dataset_name = os.path.basename(folder_path.rstrip(os.sep))
    class_names: List[str] = []
    dataset_format = "image_folder"
    if is_yolo_dataset(folder_path):
        dataset_format = "yolo"
        class_names = _parse_yolo_class_names(os.path.join(folder_path, "data.yaml"))

    records = _iter_image_records(folder_path, uploads_root, class_names)
    if not records:
        raise ValueError(f"No images found in folder dataset: {folder_path}")

    df = pd.DataFrame(records)
    preprocessing_info = {
        "data_type": "image",
        "dataset_format": dataset_format,
        "dataset_name": dataset_name,
        "image_columns": ["image_path"],
        "class_names": class_names,
        "preprocessing_steps": ["folder_scan", "manifest_build"],
    }
    return df, preprocessing_info


def collect_images_from_folder(
    folder_path: str,
    uploads_root: str,
    limit: int = 50,
) -> List[Dict[str, Any]]:
    """List images under a folder dataset for UC1 UI (path relative to uploads root)."""
    folder_path = os.path.abspath(folder_path)
    uploads_root = os.path.abspath(uploads_root)
    class_names: List[str] = []
    if is_yolo_dataset(folder_path):
        class_names = _parse_yolo_class_names(os.path.join(folder_path, "data.yaml"))

    images: List[Dict[str, Any]] = []
    for i, rec in enumerate(_iter_image_records(folder_path, uploads_root, class_names)):
        if i >= limit:
            break
        images.append(
            {
                "index": i,
                "path": rec["image_path"],
                "label": rec.get("label") or os.path.basename(rec["image_path"]),
            }
        )
    return images
