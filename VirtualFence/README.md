# IBVAP - IP Webcam Ready Prototype

This is a clean, single-folder project. It supports:
- Laptop webcam
- USB webcam
- Phone/IP webcam over HTTP/MJPEG
- RTSP CCTV/IP camera
- YOLO person detection
- ByteTrack tracking IDs
- Interactive 4-point virtual fence
- Real-time intrusion alerts
- CSV event logging
- Low-latency latest-frame capture
- Network camera reconnect

## IMPORTANT: Put this project in a short Windows path

Recommended:
C:\IBVAP

Do NOT nest the project repeatedly inside itself. A short path also avoids Windows
path-length problems during PyTorch installation.

## INSTALL

Open PowerShell inside the project folder:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## LAPTOP WEBCAM

```powershell
.\.venv\Scripts\python.exe main.py --source 0 --camera-id CAM-001
```

## IP WEBCAM / PHONE

1. Install an IP webcam application on the phone.
2. Start its video server.
3. Connect the phone and laptop to the same Wi-Fi/network.
4. Copy the VIDEO STREAM URL shown by the app.
5. Run:

```powershell
.\.venv\Scripts\python.exe main.py --source "http://PHONE_IP:PORT/video" --camera-id CAM-002
```

Example only:

```powershell
.\.venv\Scripts\python.exe main.py --source "http://192.168.1.10:8080/video" --camera-id CAM-002
```

Use the exact URL provided by your IP webcam application. Different apps expose
different paths such as /video, /mjpeg, or an RTSP URL.

You can also edit `ip_camera.txt` and use:

```powershell
.\.venv\Scripts\python.exe main.py --ip-file ip_camera.txt --camera-id CAM-002
```

## RTSP CCTV

```powershell
.\.venv\Scripts\python.exe main.py --source "rtsp://USER:PASSWORD@CAMERA_IP:554/STREAM" --camera-id CAM-003
```

## INTERACTIVE VIRTUAL FENCE

When the camera window opens:

1. Click four points around the real boundary in the camera image.
2. Click clockwise or counter-clockwise around the boundary.
3. Press ENTER.
4. The four-point polygon becomes the restricted zone.
5. The system checks the bottom-center of each person's bounding box.
6. Entering the zone generates an intrusion event.

Controls:
- Mouse left click = add fence point
- ENTER = confirm 4 points
- C = clear and redraw
- R = reset tracking state
- S = save screenshot
- Q = quit

## LOW LATENCY

The capture thread keeps only the newest frame so inference does not build up
a long queue of stale frames. Network-camera latency can still depend on the
phone app, camera encoder, Wi-Fi and stream settings.

For more speed:
- use yolo11n.pt
- lower IMAGE_SIZE in src/config.py
- use an NVIDIA GPU if supported

## DEMO ARCHITECTURE

Phone/IP Webcam
       |
       | HTTP/MJPEG or RTSP
       v
OpenCV Latest-Frame Capture
       |
       v
YOLO Person Detection
       |
       v
ByteTrack
       |
       v
Person ID + Foot Point
       |
       v
4-Point Virtual Fence
       |
       +---- SAFE
       |
       +---- INTRUSION -> Alert + events/events.csv

Cross-camera re-identification is a separate module for the next milestone.
