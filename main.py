"""Main entry point for real-time person detection."""

import os
import sys
import time
import logging

import cv2
import numpy as np

from src.detection.yolo_detector import YOLODetector
from src.detection.config import Config


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def draw_detections(frame: np.ndarray, detections: list) -> np.ndarray:
    """Draw bounding boxes, labels, and confidence scores."""
    for det in detections:
        x1, y1, x2, y2 = map(int, det["bbox"])
        conf = det["confidence"]

        # Bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Label with confidence
        label = f"Person {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw, y1), (0, 255, 0), -1)
        cv2.putText(
            frame,
            label,
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            2,
        )

    return frame


def open_source(source):
    """Open video capture from webcam or file."""
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        if isinstance(source, int):
            raise RuntimeError("ERROR: Unable to open webcam.")
        raise RuntimeError(f"ERROR: Video file not found or cannot be opened: {source}")
    return cap


def main() -> int:
    """Run the person detection pipeline."""
    logger.info("=" * 50)
    logger.info("Border Surveillance — Phase 1: Person Detection")
    logger.info("=" * 50)

    # Resolve source
    source = Config.SOURCE
    if isinstance(source, str) and not os.path.isabs(source):
        source = os.path.abspath(source)

    logger.info("Input source: %s", source)

    # Open video source
    try:
        cap = open_source(source)
    except RuntimeError as exc:
        logger.error(str(exc))
        return 1

    # Get source properties
    fps_source = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    logger.info("Source resolution: %dx%d @ %.1f FPS", width, height, fps_source)

    # Initialize detector
    try:
        detector = YOLODetector(
            model_path=Config.MODEL_PATH,
            confidence=Config.CONFIDENCE_THRESHOLD,
            person_class_id=Config.PERSON_CLASS_ID,
        )
    except RuntimeError as exc:
        logger.error(str(exc))
        cap.release()
        return 1

    frame_count = 0
    start_time = time.time()

    logger.info("Press 'Q' to quit, 'S' to save current frame.")

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            logger.info("End of stream or empty frame received.")
            break

        frame_count += 1

        # Run detection
        detections = detector.detect(frame)

        # Draw results
        vis_frame = draw_detections(frame.copy(), detections)

        # Compute FPS
        elapsed = time.time() - start_time
        fps = frame_count / elapsed if elapsed > 0 else 0.0
        fps_text = f"FPS: {fps:.1f}"
        cv2.putText(
            vis_frame,
            fps_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
        )

        # Detection count
        count_text = f"Detections: {len(detections)}"
        cv2.putText(
            vis_frame,
            count_text,
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 0, 0),
            2,
        )

        # Show frame
        cv2.imshow(Config.WINDOW_NAME, vis_frame)

        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord(Config.QUIT_KEY.lower()) or key == ord(Config.QUIT_KEY.upper()):
            logger.info("Quit requested.")
            break
        elif key == ord(Config.SAVE_KEY.lower()) or key == ord(Config.SAVE_KEY.upper()):
            save_path = f"saved_frame_{frame_count:06d}.png"
            cv2.imwrite(save_path, frame)
            logger.info("Frame saved to: %s", save_path)

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

    logger.info("Total frames processed: %d", frame_count)
    logger.info("Average FPS: %.2f", frame_count / (time.time() - start_time))
    logger.info("Shutdown complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())