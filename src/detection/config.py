"""Centralized configuration for the detection pipeline."""

from typing import Union


class Config:
    """Application configuration."""

    # Input source: 0 for webcam, or path to MP4/video file
    SOURCE: Union[int, str] = 0

    # YOLO model path — lightweight student prototype model
    MODEL_PATH: str = "yolo11n.pt"

    # Detection confidence threshold
    CONFIDENCE_THRESHOLD: float = 0.5

    # COCO class ID for "person"
    PERSON_CLASS_ID: int = 0

    # Display window name
    WINDOW_NAME: str = "Border Surveillance — Person Detection"

    # Frame save key
    SAVE_KEY: str = "s"

    # Quit key
    QUIT_KEY: str = "q"