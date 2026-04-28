from typing import Optional

from pydantic import BaseModel


class XAIResponse(BaseModel):
    result: dict | str        # main XAI output: scores dict or base64 image
    storage_ref: str          # minio://bucket/path/to/full_result


# --- UC1 ---

class UC1SaliencyRequest(BaseModel):
    image_ref: str            # minio://uc1-robotics/train/images/filename.jpg
    detection_index: int = 0  # which detection to explain (0 = highest confidence)
    patch_size: int = 32      # occlusion patch size in pixels
    stride: int = 16          # stride between patches in pixels
    iou_thres: float = 0.4    # IoU threshold to match detections across occluded images
    user_id: str = "default"


class UC1PrototypeRequest(BaseModel):
    n_prototypes: int = 5


# --- UC2 ---

class UC2OcclusionRequest(BaseModel):
    text_ref: Optional[str] = None
    sample_index: int = 0
    text: Optional[str] = None  # if set, skip CSV load
    true_sentiment: Optional[str] = None
    user_id: str = "default"


class UC2LIMERequest(BaseModel):
    text_ref: Optional[str] = None
    sample_index: int = 0
    text: Optional[str] = None
    true_sentiment: Optional[str] = None
    num_features: int = 10
    num_samples: int = 300
    user_id: str = "default"
