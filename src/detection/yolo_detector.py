"""YOLO-based person detector with crop extraction."""

from typing import List, Dict, Any, Optional
import logging

import cv2
import numpy as np
import torch
from ultralytics import YOLO


logger = logging.getLogger(__name__)


class YOLODetector:
    """Reusable YOLO person detector.

    Detects only the COCO 'person' class and returns structured detection
    dictionaries compatible with ByteTrack (Member 2 integration).
    """

    def __init__(
        self,
        model_path: str = "yolo11n.pt",
        confidence: float = 0.5,
        person_class_id: int = 0,
    ) -> None:
        """Initialize the YOLO detector.

        Args:
            model_path: Path to the YOLO model weights.
            confidence: Minimum confidence threshold for detections.
            person_class_id: COCO class ID for person (default 0).
        """
        self.confidence = confidence
        self.person_class_id = person_class_id
        self.device = self._select_device()
        self.model = self._load_model(model_path)

    def _select_device(self) -> str:
        """Automatically select CUDA if available, otherwise CPU."""
        if torch.cuda.is_available():
            device = "cuda"
            logger.info("Device: CUDA (%s)", torch.cuda.get_device_name(0))
        else:
            device = "cpu"
            logger.info("Device: CPU")
        return device

    def _load_model(self, model_path: str) -> YOLO:
        """Load the YOLO model and warm it up."""
        try:
            model = YOLO(model_path)
            model.to(self.device)
            # Warm-up inference to initialize model internals
            dummy = np.zeros((640, 640, 3), dtype=np.uint8)
            model.predict(dummy, verbose=False)
            logger.info("Model loaded: %s", model_path)
            return model
        except Exception as exc:
            raise RuntimeError(f"Failed to load YOLO model '{model_path}': {exc}") from exc

    def detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Run person detection on a single frame.

        Args:
            frame: BGR image as a NumPy array.

        Returns:
            List of detection dictionaries with keys:
                bbox, confidence, class_id, class_name, crop
        """
        if frame is None or frame.size == 0:
            return []

        h, w = frame.shape[:2]

        results = self.model.predict(
            frame,
            classes=[self.person_class_id],
            conf=self.confidence,
            verbose=False,
            device=self.device,
        )

        detections: List[Dict[str, Any]] = []

        if not results or len(results) == 0:
            return detections

        result = results[0]
        if result.boxes is None:
            return detections

        boxes = result.boxes.xyxy.cpu().numpy()  # type: ignore
        confs = result.boxes.conf.cpu().numpy()   # type: ignore
        cls_ids = result.boxes.cls.cpu().numpy()  # type: ignore

        for box, conf, cls_id in zip(boxes, confs, cls_ids):
            x1, y1, x2, y2 = map(float, box)

            # Clip to valid frame dimensions
            x1 = max(0.0, min(x1, w - 1))
            y1 = max(0.0, min(y1, h - 1))
            x2 = max(0.0, min(x2, w))
            y2 = max(0.0, min(y2, h))

            # Ensure valid crop
            x1_int, y1_int, x2_int, y2_int = int(x1), int(y1), int(x2), int(y2)
            if x2_int <= x1_int or y2_int <= y1_int:
                continue

            crop = frame[y1_int:y2_int, x1_int:x2_int].copy()

            detection = {
                "bbox": [x1, y1, x2, y2],
                "confidence": float(conf),
                "class_id": int(cls_id),
                "class_name": "person",
                "crop": crop,
            }
            detections.append(detection)

        return detections