"""Detection module for YOLO-based person detection."""

from .yolo_detector import YOLODetector
from .config import Config

__all__ = ["YOLODetector", "Config"]