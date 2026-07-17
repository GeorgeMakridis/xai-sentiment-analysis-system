"""
Image-folder dataset helpers for the dashboard (mirrors xai_service/image_folder_loader).
"""
from __future__ import annotations

import ast
import io
import os
import re
import zipfile
from typing import Any, Dict, List, Optional, Tuple

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp", ".tif", ".tiff"}
YOLO_SPLITS = ("train", "valid", "val", "test")


def is_yolo_dataset(path: str) -> bool:
    return os.path.isdir(path) and os.path.isfile(os.path.join(path, "data.yaml"))


def is_image_folder(path: str) -> bool:
    if not os.path.isdir(path):
        return False
    if is_yolo_dataset(path):
        return True
    for _root, _dirs, files in os.walk(path):
        for fname in files:
            if os.path.splitext(fname)[1].lower() in IMAGE_EXTENSIONS:
                return True
    return False


def folder_size(path: str) -> int:
    total = 0
    for root, _dirs, files in os.walk(path):
        for fname in files:
            try:
                total += os.path.getsize(os.path.join(root, fname))
            except OSError:
                pass
    return total


def _parse_yolo_class_names(data_yaml_path: str) -> List[str]:
    with open(data_yaml_path, encoding="utf-8") as f:
        content = f.read()
    match = re.search(r"^names:\s*(.+)$", content, re.MULTILINE)
    if not match:
        return []
    try:
        names = ast.literal_eval(match.group(1).strip())
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


def collect_images_from_folder(
    folder_path: str,
    uploads_root: str,
    limit: int = 50,
) -> List[Dict[str, Any]]:
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


def zip_folder_to_bytes(folder_path: str) -> bytes:
    """Zip a dataset folder for RustFS persistence."""
    folder_path = os.path.abspath(folder_path)
    base_name = os.path.basename(folder_path.rstrip(os.sep))
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, _dirs, files in os.walk(folder_path):
            for fname in files:
                full = os.path.join(root, fname)
                arcname = os.path.join(base_name, os.path.relpath(full, folder_path))
                zf.write(full, arcname)
    return buf.getvalue()


def unzip_bytes_to_folder(zip_bytes: bytes, dest_root: str, folder_name: str) -> str:
    """Extract a dataset zip under dest_root. Returns absolute folder path."""
    dest_root = os.path.abspath(dest_root)
    os.makedirs(dest_root, exist_ok=True)
    folder_path = os.path.join(dest_root, folder_name)
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        zf.extractall(dest_root)
    if not os.path.isdir(folder_path):
        # zip may use a single top-level directory with a different name
        top_dirs = [
            os.path.join(dest_root, d)
            for d in os.listdir(dest_root)
            if os.path.isdir(os.path.join(dest_root, d))
        ]
        for candidate in top_dirs:
            if is_image_folder(candidate):
                return candidate
    return folder_path


def save_manifest_csv(folder_path: str, uploads_root: str) -> Optional[str]:
    """Write a manifest CSV next to the folder for lightweight RustFS backup."""
    import csv

    folder_path = os.path.abspath(folder_path)
    dataset_name = os.path.basename(folder_path.rstrip(os.sep))
    class_names: List[str] = []
    if is_yolo_dataset(folder_path):
        class_names = _parse_yolo_class_names(os.path.join(folder_path, "data.yaml"))
    records = _iter_image_records(folder_path, uploads_root, class_names)
    if not records:
        return None
    manifest_name = f"{dataset_name}_manifest.csv"
    manifest_path = os.path.join(uploads_root, manifest_name)
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["image_path", "split", "label", "label_names", "num_objects"],
        )
        writer.writeheader()
        writer.writerows(records)
    return manifest_path
