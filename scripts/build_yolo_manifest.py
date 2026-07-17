"""
Build a manifest CSV from a YOLO/Roboflow image-folder dataset.

Usage (from repo root):
    python scripts/build_yolo_manifest.py shared_volume/uploads/5_navetext
"""
import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "xai_service"))

from image_folder_loader import _iter_image_records, is_image_folder, is_yolo_dataset, _parse_yolo_class_names  # noqa: E402


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/build_yolo_manifest.py <dataset_folder>")
        sys.exit(1)

    folder = Path(sys.argv[1]).resolve()
    if not is_image_folder(str(folder)):
        print(f"Not a supported image folder: {folder}")
        sys.exit(1)

    uploads_root = folder.parent
    class_names = []
    if is_yolo_dataset(str(folder)):
        class_names = _parse_yolo_class_names(str(folder / "data.yaml"))

    records = _iter_image_records(str(folder), str(uploads_root), class_names)
    out = uploads_root / f"{folder.name}_manifest.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["image_path", "split", "label", "label_names", "num_objects"],
        )
        writer.writeheader()
        writer.writerows(records)
    print(f"Wrote {len(records)} rows to {out}")


if __name__ == "__main__":
    main()
