import argparse
import os
import time
import cv2

from src.capture import LatestFrameCapture
from src.config import Config
from src.events import EventLogger
from src.fence import LockedVirtualFence
from src.tracker import PersonTracker
from src.alert import AlertSystem
from src.config_alert import ALERT_COOLDOWN_SECONDS, ALERT_DISPLAY_SECONDS


def parse_source(value):
    try:
        return int(value)
    except ValueError:
        if value.lower().startswith(("http://", "https://", "rtsp://")):
            return value
        return os.path.abspath(value)


def read_ip_file(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                return line
    raise ValueError("No IP camera URL found.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", help="Webcam index, HTTP/MJPEG URL, RTSP URL, or video file")
    parser.add_argument("--ip-file", help="Read IP webcam URL from a text file")
    parser.add_argument("--camera-id", default="CAM-001")
    args = parser.parse_args()

    if args.ip_file:
        source = read_ip_file(args.ip_file)
    elif args.source is not None:
        source = parse_source(args.source)
    else:
        source = 0

    print(f"Camera: {args.camera_id}")
    print(f"Source: {source}")

    capture = LatestFrameCapture(source, Config.RECONNECT_SECONDS)
    if not capture.open():
        raise RuntimeError(
            f"Could not open camera source: {source}. "
            "Check camera/IP URL, network and firewall."
        )

    tracker = PersonTracker(
        Config.MODEL_PATH,
        Config.CONFIDENCE_THRESHOLD,
        Config.IMAGE_SIZE,
        Config.DEVICE,
    )

    fence = LockedVirtualFence(
        Config.ENTER_DWELL_SECONDS,
        Config.EXIT_DWELL_SECONDS,
        Config.BOUNDARY_MARGIN_PX,
    )

    logger = EventLogger(Config.EVENT_LOG)

    alert_system = AlertSystem(
        cooldown_seconds=ALERT_COOLDOWN_SECONDS,
        display_seconds=ALERT_DISPLAY_SECONDS,
    )

    cv2.namedWindow(Config.WINDOW_NAME, cv2.WINDOW_NORMAL)

    def mouse_callback(event, x, y, flags, param):
        # Once ENTER locks the fence, clicks are ignored.
        if event == cv2.EVENT_LBUTTONDOWN:
            fence.add_point(x, y)

    cv2.setMouseCallback(Config.WINDOW_NAME, mouse_callback)

    last_time = time.perf_counter()
    fps = 0.0

    print()
    print("GOD'S EYE ALERT SYSTEM READY")
    print("Click 4 points -> ENTER = LOCK")
    print("C = clear/unlock | R = reset tracking | S = screenshot | Q = quit")
    print()

    try:
        while True:
            ok, frame = capture.read()

            if not ok:
                time.sleep(0.01)
                continue

            tracks = tracker.track(frame)
            now = time.time()

            for item in tracks:
                track_id = item["track_id"]
                x1, y1, x2, y2 = map(int, item["bbox"])
                confidence = item["confidence"]

                # Ground-contact point: more appropriate for a ground virtual fence.
                foot = ((x1 + x2) // 2, y2)

                result = fence.update(track_id, foot, now)
                state = result["state"]
                event = result["event"]

                if state == "INSIDE":
                    box_color = (0, 0, 255)
                else:
                    box_color = (0, 220, 0)

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    box_color,
                    2,
                )

                cv2.circle(
                    frame,
                    foot,
                    5,
                    (255, 255, 0),
                    -1,
                )

                cv2.putText(
                    frame,
                    f"Person ID {track_id}  {confidence:.2f}  {state}",
                    (x1, max(125, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.50,
                    box_color,
                    2,
                )

                # EVENT -> ALERT
                if event:
                    if alert_system.trigger(
                        track_id,
                        event,
                        args.camera_id,
                    ):
                        if event == "VIRTUAL_FENCE_INTRUSION":
                            direction = "ENTERING"
                        else:
                            direction = "EXITING"

                        logger.log(
                            args.camera_id,
                            track_id,
                            event,
                            direction,
                            confidence,
                        )

                        print(
                            f"[ALERT] {args.camera_id} | "
                            f"Target {track_id} | {direction}"
                        )

            # Draw locked fence after detections.
            fence.draw(frame)

            # Draw active alerts directly on video.
            active_alerts = alert_system.get_active()

            for track_id, alert in active_alerts.items():
                # Find the matching current track so the text appears near it.
                for item in tracks:
                    if item["track_id"] == track_id:
                        x1, y1, x2, y2 = map(int, item["bbox"])

                        if alert["event"] == "VIRTUAL_FENCE_INTRUSION":
                            alert_color = (0, 0, 255)
                        else:
                            alert_color = (0, 215, 255)

                        cv2.putText(
                            frame,
                            alert["text"],
                            (x1, min(frame.shape[0] - 20, y2 + 25)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.55,
                            alert_color,
                            2,
                        )
                        break

            # FPS and camera status.
            current = time.perf_counter()
            instant = 1.0 / max(current - last_time, 1e-6)
            last_time = current
            fps = instant if fps == 0 else 0.85 * fps + 0.15 * instant

            cv2.rectangle(
                frame,
                (0, 0),
                (frame.shape[1], 72),
                (25, 25, 25),
                -1,
            )

            source_type = "WEBCAM" if isinstance(source, int) else "IP/RTSP"
            status = "CONNECTED" if capture.connected else "RECONNECTING"

            cv2.putText(
                frame,
                "GOD'S EYE - REAL-TIME BORDER SURVEILLANCE",
                (15, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.68,
                (255, 255, 255),
                2,
            )

            cv2.putText(
                frame,
                f"{args.camera_id} | {source_type} | {status} | FPS: {fps:.1f}",
                (15, 56),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (220, 220, 220),
                1,
            )

            # Large global alert banner.
            if active_alerts:
                newest = list(active_alerts.values())[-1]

                if newest["event"] == "VIRTUAL_FENCE_INTRUSION":
                    banner_color = (0, 0, 180)
                else:
                    banner_color = (0, 160, 200)

                y = frame.shape[0] - 70

                cv2.rectangle(
                    frame,
                    (10, y),
                    (min(frame.shape[1] - 10, 800), frame.shape[0] - 10),
                    banner_color,
                    -1,
                )

                cv2.putText(
                    frame,
                    newest["text"],
                    (22, frame.shape[0] - 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.60,
                    (255, 255, 255),
                    2,
                )

            cv2.imshow(Config.WINDOW_NAME, frame)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), ord("Q")):
                break

            elif key in (ord("c"), ord("C")):
                fence.clear()
                alert_system.clear()
                print("Fence unlocked and cleared.")

            elif key == 13:
                if fence.confirm():
                    print("Fence LOCKED. Alerts are active.")
                else:
                    print(
                        f"Need exactly 4 points; currently "
                        f"{len(fence.points)}."
                    )

            elif key in (ord("r"), ord("R")):
                fence.reset_tracking()
                tracker.reset()
                alert_system.clear()
                print("Tracking reset. Locked fence unchanged.")

            elif key in (ord("s"), ord("S")):
                filename = f"ibvap_alert_{int(time.time())}.jpg"
                cv2.imwrite(filename, frame)
                print(f"Saved {filename}")

    finally:
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
