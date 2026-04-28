import base64
import requests
import config


class ModelClient:
    def __init__(self):
        self._base_url = config.MODEL_API_URL

    def predict_uc1(self, images: list[bytes]) -> list[list[dict]]:
        """
        Send a batch of raw image bytes to the model API.
        Returns a list of detections per image.
        Each detection: {class_id, class_name, confidence, bbox}
        """
        payload = {"images": [base64.b64encode(img).decode() for img in images]}
        response = requests.post(f"{self._base_url}/predict/uc1", json=payload)
        response.raise_for_status()
        return response.json()["detections"]

    def predict_uc2(self, texts: list[str]) -> list[dict]:
        """
        Send a batch of texts to the model API.
        Returns a list of {positive, negative, neutral} dicts.
        """
        payload = {"texts": texts}
        response = requests.post(f"{self._base_url}/predict/uc2", json=payload)
        response.raise_for_status()
        return response.json()["predictions"]
