"""
Occlusion-based saliency for UC1 vision (YOLO via REST API).

Ported from uc1_robotics/saliency_utils.py.
Key adaptations:
  - Model is called via REST API (model_predict callable) instead of local Ultralytics object.
  - Bboxes are normalized [0-1] throughout (API contract), not pixel coords.
  - iou() is coordinate-agnostic and unchanged.
  - get_best_conf_for_target() rewritten to accept detection dicts instead of Ultralytics result.
  - Occluded image is re-encoded to JPEG bytes for each model call.
  - Output includes a rendered heatmap overlay (jet colormap, alpha=0.45) as PNG bytes.
"""
import io
import numpy as np
from scipy.ndimage import zoom
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm


# --- Default params (from uc1_robotics/config.py) ---
DEFAULT_PATCH_SIZE = 32
DEFAULT_STRIDE = 16
DEFAULT_IOU_THRES = 0.4
SALIENCY_COLORMAP = "jet"
SALIENCY_ALPHA = 0.45


def iou(boxA, boxB):
    """
    Intersection over Union between two boxes.
    Works with any consistent coordinate system (pixel or normalized).
    Unchanged from uc1_robotics/saliency_utils.py.
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interW = max(0.0, xB - xA)
    interH = max(0.0, yB - yA)
    inter = interW * interH
    areaA = max(0.0, (boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    areaB = max(0.0, (boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))
    union = areaA + areaB - inter
    return inter / union if union > 0 else 0.0


def get_best_conf_for_target(detections, target_cls_id, target_box_norm, iou_thres=0.4):
    """
    Get best confidence for a target class+box from API detection dicts.

    Adapted from uc1_robotics/saliency_utils.py::get_best_conf_for_target().
    Original read from Ultralytics result object with pixel bboxes.
    New reads from list of dicts with normalized [0-1] bboxes.

    Args:
        detections: list of {class_id, class_name, confidence, bbox: [x1,y1,x2,y2] normalized}
        target_cls_id: int
        target_box_norm: [x1, y1, x2, y2] normalized — from baseline detection
        iou_thres: float

    Returns:
        float: best matching confidence, 0.0 if no match
    """
    if not detections:
        return 0.0
    best_conf = 0.0
    for det in detections:
        if det["class_id"] == target_cls_id:
            if iou(det["bbox"], target_box_norm) >= iou_thres:
                best_conf = max(best_conf, det["confidence"])
    return best_conf


def _numpy_to_jpeg_bytes(img_np):
    """Encode HxWx3 uint8 numpy array to JPEG bytes."""
    buf = io.BytesIO()
    Image.fromarray(img_np).save(buf, format="JPEG", quality=95)
    return buf.getvalue()


def _render_heatmap_overlay(img_np, heatmap, target_box_norm, target_cls_name, orig_conf):
    """
    Render heatmap overlay on original image.
    Replicates uc1_robotics/saliency_visualization.py style:
      - jet colormap heatmap overlaid with alpha=0.45
      - detection bbox drawn on top

    Args:
        img_np: HxWx3 uint8 numpy array
        heatmap: HxW float [0,1] numpy array
        target_box_norm: [x1, y1, x2, y2] normalized
        target_cls_name: str
        orig_conf: float

    Returns:
        PNG bytes
    """
    H, W, _ = img_np.shape
    x1, y1, x2, y2 = [int(v * d) for v, d in zip(target_box_norm, [W, H, W, H])]

    colormap = cm.get_cmap(SALIENCY_COLORMAP)
    heatmap_rgba = colormap(heatmap)  # HxWx4 float
    heatmap_rgb = (heatmap_rgba[:, :, :3] * 255).astype(np.uint8)

    # Blend original image with heatmap
    overlay = (img_np * (1 - SALIENCY_ALPHA) + heatmap_rgb * SALIENCY_ALPHA).astype(np.uint8)

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.imshow(overlay)
    rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                          linewidth=2, edgecolor="white", facecolor="none")
    ax.add_patch(rect)
    ax.set_title(f"{target_cls_name} (conf={orig_conf:.2f})", fontsize=10)
    ax.axis("off")
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def run_saliency(
    image_bytes,
    detection_index,
    model_predict,
    patch_size=DEFAULT_PATCH_SIZE,
    stride=DEFAULT_STRIDE,
    iou_thres=DEFAULT_IOU_THRES,
):
    """
    Occlusion-based saliency for a single detection via REST model API.

    Ported from uc1_robotics/saliency_utils.py::saliency() + saliency_single().
    The two original functions are merged since we no longer need the
    Ultralytics-specific separation.

    Args:
        image_bytes: raw image bytes (JPEG/PNG) from MinIO
        detection_index: int — which detection to explain (0 = highest confidence)
        model_predict: callable(list[bytes]) -> list[list[dict]]
                       each dict: {class_id, class_name, confidence, bbox: [x1,y1,x2,y2] normalized}
        patch_size: occlusion patch size in pixels
        stride: stride between patches in pixels
        iou_thres: IoU threshold to match detections across occluded images

    Returns:
        heatmap_png_bytes: PNG bytes of heatmap overlay, or None if no detections
        detection_info: dict with target detection metadata, or None if no detections
    """
    # Decode image bytes to numpy
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img_np = np.array(img)
    H, W, _ = img_np.shape

    # Baseline: run model once on original image
    baseline_dets = model_predict([image_bytes])[0]

    # Handle no detections
    if not baseline_dets:
        return None, None

    # Sort by confidence descending — same as original idx_sorted = np.argsort(-confs_np)
    sorted_dets = sorted(baseline_dets, key=lambda d: d["confidence"], reverse=True)

    # Clamp detection_index to valid range
    detection_index = min(detection_index, len(sorted_dets) - 1)
    target_det = sorted_dets[detection_index]

    target_cls_id = target_det["class_id"]
    target_cls_name = target_det["class_name"]
    target_box_norm = target_det["bbox"]   # normalized [0-1] — used for IoU matching
    orig_conf = target_det["confidence"]

    # Occlusion grid (same logic as original)
    ys = list(range(0, H - patch_size + 1, stride))
    xs = list(range(0, W - patch_size + 1, stride))
    if not ys:
        ys = [0]
    if not xs:
        xs = [0]

    heatmap = np.zeros((len(ys), len(xs)), dtype=np.float32)

    # Baseline fill color = mean image color (same as original)
    baseline_color = tuple(int(img_np[..., c].mean()) for c in range(3))

    for i, y in enumerate(ys):
        for j, x in enumerate(xs):
            occluded = img_np.copy()
            occluded[y:y + patch_size, x:x + patch_size, :] = baseline_color

            # Encode occluded numpy to bytes for REST API call
            occluded_bytes = _numpy_to_jpeg_bytes(occluded)

            dets = model_predict([occluded_bytes])[0]
            new_conf = get_best_conf_for_target(dets, target_cls_id, target_box_norm, iou_thres)

            drop = orig_conf - new_conf
            heatmap[i, j] = max(drop, 0.0)  # clamp negative drops to 0

    # Upsample heatmap to image size (same as original)
    zoom_y = float(H) / float(heatmap.shape[0])
    zoom_x = float(W) / float(heatmap.shape[1])
    heatmap_up = zoom(heatmap, (zoom_y, zoom_x), order=1)

    # Normalize to [0,1] (same as original)
    heatmap_up = heatmap_up.astype(np.float32)
    heatmap_up -= heatmap_up.min()
    if heatmap_up.max() > 0:
        heatmap_up /= heatmap_up.max()

    # Render colored overlay
    heatmap_png_bytes = _render_heatmap_overlay(
        img_np, heatmap_up, target_box_norm, target_cls_name, orig_conf
    )

    detection_info = {
        "class_id": target_cls_id,
        "class_name": target_cls_name,
        "confidence": orig_conf,
        "bbox": target_box_norm,
    }

    return heatmap_png_bytes, detection_info
