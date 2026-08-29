# Border Surveillance — Phase 1: YOLO Person Detection

**Member 1 Responsibility:** Real-time person detection using YOLO.

This module consumes a video stream (webcam or MP4), detects only the **person** class using a lightweight YOLO model, and outputs structured detection dictionaries ready for Member 2's ByteTrack integration.

---

## 1. Project Setup (UV)

All environment and dependency management is handled by **UV**.

### Prerequisites
- UV installed: &lt;https://docs.astral.sh/uv/getting-started/installation/&gt;
- Python 3.11+ (managed automatically by UV)

### Initialize & Install

```bash
uv init --name border-surveillance-detection
uv python pin 3.11
uv add ultralytics opencv-python
uv sync