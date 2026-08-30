import argparse
import os
import time
import cv2

from src.capture import LatestFrameCapture
from src.config import Config
from src.events import EventLogger
from src.fence import LockedVirtualFence
from src.tracker import PersonTracker

from src.snapshot import save_snapshot
from src.database import initialize_database, create_event

from src.alert import AlertSystem
from src.config_alert import (
    ALERT_COOLDOWN_SECONDS,
    ALERT_DISPLAY_SECONDS,
)


def parse_source(value):
    value = value.strip()

    if value.lower().startswith(
        ("http://", "https://", "rtsp://")
    ):
        return value

    raise ValueError(
        "Invalid camera source. "
        "Use an IP camera HTTP/MJPEG URL or RTSP URL."
    )


def read_ip_file(path):
    with open(path, "r", encoding="utf-8") as f:

        for line in f:
            line = line.strip()

            if line and not line.startswith("#"):
                return line

    raise ValueError("No IP camera URL found.")


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--source",
        required=True,
        help="IP camera HTTP/MJPEG URL or RTSP URL"
    )

    parser.add_argument(
        "--ip-file",
        help="Read IP webcam URL from a text file"
    )

    parser.add_argument(
        "--camera-id",
        default="CAM-001"
    )

    args = parser.parse_args()


    # --------------------------------------------------
    # CAMERA SOURCE
    # --------------------------------------------------

    if args.ip_file:

        source = read_ip_file(args.ip_file)

    elif args.source:

        source = parse_source(args.source)

    else:

        raise ValueError(
            "Camera source is required. "
            "Provide an IP camera HTTP/MJPEG URL or RTSP URL."
        )


    print()
    print("========================================")
    print("        GOD'S EYE SURVEILLANCE")
    print("========================================")
    print(f"Camera ID : {args.camera_id}")
    print(f"Source    : {source}")
    print()


    # --------------------------------------------------
    # CAMERA
    # --------------------------------------------------

    capture = LatestFrameCapture(
        source,
        Config.RECONNECT_SECONDS
    )

    if not capture.open():

        raise RuntimeError(
            f"Could not open camera source: {source}\n"
            "Check camera/IP URL, network and firewall."
        )


    # --------------------------------------------------
    # YOLO + BYTE TRACK
    # --------------------------------------------------

    tracker = PersonTracker(
        Config.MODEL_PATH,
        Config.CONFIDENCE_THRESHOLD,
        Config.IMAGE_SIZE,
        Config.DEVICE,
    )


    # --------------------------------------------------
    # VIRTUAL FENCE
    # --------------------------------------------------

    fence = LockedVirtualFence(
        Config.ENTER_DWELL_SECONDS,
        Config.EXIT_DWELL_SECONDS,
        Config.BOUNDARY_MARGIN_PX,
    )


    # --------------------------------------------------
    # OLD CSV LOGGER
    # --------------------------------------------------

    logger = EventLogger(
        Config.EVENT_LOG
    )


    # --------------------------------------------------
    # SQLITE DATABASE
    # --------------------------------------------------

    initialize_database()

    print("Database: READY")


    # --------------------------------------------------
    # ALERT SYSTEM
    # --------------------------------------------------

    alert_system = AlertSystem(
        cooldown_seconds=ALERT_COOLDOWN_SECONDS,
        display_seconds=ALERT_DISPLAY_SECONDS,
    )


    # --------------------------------------------------
    # OPEN CAMERA WINDOW
    # --------------------------------------------------

    cv2.namedWindow(
        Config.WINDOW_NAME,
        cv2.WINDOW_NORMAL
    )


    # --------------------------------------------------
    # MOUSE CONTROL
    # --------------------------------------------------

    def mouse_callback(event, x, y, flags, param):

        # IMPORTANT:
        # Once the fence is locked, clicks are ignored.

        if event == cv2.EVENT_LBUTTONDOWN:

            fence.add_point(x, y)


    cv2.setMouseCallback(
        Config.WINDOW_NAME,
        mouse_callback
    )


    # --------------------------------------------------
    # FPS
    # --------------------------------------------------

    last_time = time.perf_counter()

    fps = 0.0


    print()
    print("GOD'S EYE READY")
    print("----------------------------------------")
    print("1. Click four fence points")
    print("2. Press ENTER to LOCK the fence")
    print("3. Walk into the restricted zone")
    print()
    print("C = Clear / redraw fence")
    print("R = Reset tracking")
    print("S = Save screenshot")
    print("Q = Quit")
    print("----------------------------------------")
    print()


    # --------------------------------------------------
    # MAIN LOOP
    # --------------------------------------------------

    try:

        while True:

            # ------------------------------------------
            # GET LATEST CAMERA FRAME
            # ------------------------------------------

            ok, frame = capture.read()

            if not ok:

                time.sleep(0.01)

                continue


            # ------------------------------------------
            # YOLO + BYTE TRACK
            # ------------------------------------------

            tracks = tracker.track(frame)

            now = time.time()


            # ------------------------------------------
            # PROCESS EACH PERSON
            # ------------------------------------------

            for item in tracks:

                track_id = item["track_id"]

                x1, y1, x2, y2 = map(
                    int,
                    item["bbox"]
                )

                confidence = item["confidence"]


                # --------------------------------------
                # FOOT POINT
                # --------------------------------------

                foot = (
                    (x1 + x2) // 2,
                    y2
                )


                # --------------------------------------
                # CHECK VIRTUAL FENCE
                # --------------------------------------

                result = fence.update(
                    track_id,
                    foot,
                    now
                )

                state = result["state"]

                event = result["event"]


                # --------------------------------------
                # PERSON COLOR
                # --------------------------------------

                if state == "INSIDE":

                    box_color = (
                        0,
                        0,
                        255
                    )

                else:

                    box_color = (
                        0,
                        220,
                        0
                    )


                # --------------------------------------
                # DRAW PERSON
                # --------------------------------------

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    box_color,
                    2
                )


                # Foot point

                cv2.circle(
                    frame,
                    foot,
                    5,
                    (255, 255, 0),
                    -1
                )


                # Person information

                cv2.putText(
                    frame,
                    (
                        f"Person ID {track_id} "
                        f"{confidence:.2f} "
                        f"{state}"
                    ),
                    (
                        x1,
                        max(
                            125,
                            y1 - 8
                        )
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.50,
                    box_color,
                    2
                )


                # ==================================================
                # SECURITY EVENT
                # ==================================================

                if event:

                    # ----------------------------------------------
                    # EVENT TYPE
                    # ----------------------------------------------

                    if event == "VIRTUAL_FENCE_INTRUSION":

                        event_type = "ENTER"

                        direction = "ENTERING"

                        event_message = (
                            f"INTRUSION - "
                            f"TARGET {track_id} ENTERING"
                        )

                    else:

                        event_type = "EXIT"

                        direction = "EXITING"

                        event_message = (
                            f"TARGET {track_id} "
                            f"EXITING RESTRICTED ZONE"
                        )


                    # ----------------------------------------------
                    # CREATE EVENT SNAPSHOT
                    # ----------------------------------------------
                    #
                    # We create a COPY so the evidence contains
                    # the detected person's bounding box and event.
                    #

                    event_frame = frame.copy()


                    # Draw event bounding box

                    cv2.rectangle(
                        event_frame,
                        (x1, y1),
                        (x2, y2),
                        (0, 0, 255),
                        3
                    )


                    # Draw foot point

                    cv2.circle(
                        event_frame,
                        foot,
                        7,
                        (255, 255, 0),
                        -1
                    )


                    # Draw event text

                    cv2.putText(
                        event_frame,
                        event_message,
                        (
                            x1,
                            min(
                                frame.shape[0] - 20,
                                y2 + 30
                            )
                        ),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        (0, 0, 255),
                        2
                    )


                    # ----------------------------------------------
                    # SAVE SNAPSHOT
                    # ----------------------------------------------

                    snapshot_path, snapshot_hash = save_snapshot(
                        event_frame,
                        event,
                        args.camera_id,
                        track_id
                    )


                    # ----------------------------------------------
                    # SAVE SQLITE DATABASE RECORD
                    # ----------------------------------------------

                    event_id = create_event(

                        camera_id=args.camera_id,

                        track_id=track_id,

                        event_type=event_type,

                        confidence=confidence,

                        snapshot_path=snapshot_path,

                        snapshot_hash=snapshot_hash
                    )


                    # ----------------------------------------------
                    # SAVE OLD CSV RECORD
                    # ----------------------------------------------

                    logger.log(

                        args.camera_id,

                        track_id,

                        event,

                        direction,

                        confidence
                    )


                    # ----------------------------------------------
                    # TRIGGER SOUND + VISUAL ALERT
                    # ----------------------------------------------
                    #
                    # IMPORTANT:
                    # Database and snapshot are already saved.
                    #
                    # The alert cooldown only controls the alert,
                    # NOT the evidence.
                    #

                    alert_system.trigger(

                        track_id,

                        event,

                        args.camera_id
                    )


                    # ----------------------------------------------
                    # TERMINAL SECURITY MESSAGE
                    # ----------------------------------------------

                    print()
                    print("========================================")

                    if event_type == "ENTER":

                        print("🚨 INTRUSION DETECTED")

                    else:

                        print("⚠️ ZONE EXIT DETECTED")

                    print("----------------------------------------")

                    print(
                        f"Event ID   : {event_id}"
                    )

                    print(
                        f"Camera     : {args.camera_id}"
                    )

                    print(
                        f"Target     : {track_id}"
                    )

                    print(
                        f"Event      : {event_type}"
                    )

                    print(
                        f"Confidence : {confidence:.2f}"
                    )

                    print(
                        f"Snapshot   : {snapshot_path}"
                    )

                    print(
                        f"SHA-256    : {snapshot_hash}"
                    )

                    print("========================================")
                    print()


            # ======================================================
            # DRAW LOCKED FENCE
            # ======================================================

            fence.draw(frame)


            # ======================================================
            # ACTIVE ALERTS
            # ======================================================

            active_alerts = (
                alert_system.get_active()
            )


            # ------------------------------------------------------
            # DRAW ALERT NEXT TO PERSON
            # ------------------------------------------------------

            for track_id, alert in active_alerts.items():

                for item in tracks:

                    if item["track_id"] == track_id:

                        x1, y1, x2, y2 = map(
                            int,
                            item["bbox"]
                        )


                        if (
                            alert["event"]
                            == "VIRTUAL_FENCE_INTRUSION"
                        ):

                            alert_color = (
                                0,
                                0,
                                255
                            )

                        else:

                            alert_color = (
                                0,
                                215,
                                255
                            )


                        cv2.putText(
                            frame,
                            alert["text"],
                            (
                                x1,
                                min(
                                    frame.shape[0] - 20,
                                    y2 + 25
                                )
                            ),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.55,
                            alert_color,
                            2
                        )

                        break


            # ======================================================
            # FPS
            # ======================================================

            current = time.perf_counter()

            instant = (
                1.0
                /
                max(
                    current - last_time,
                    1e-6
                )
            )

            last_time = current


            if fps == 0:

                fps = instant

            else:

                fps = (
                    0.85 * fps
                    +
                    0.15 * instant
                )


            # ======================================================
            # HEADER
            # ======================================================

            cv2.rectangle(
                frame,
                (0, 0),
                (
                    frame.shape[1],
                    72
                ),
                (25, 25, 25),
                -1
            )


            if isinstance(
                source,
                int
            ):

                source_type = "WEBCAM"

            else:

                source_type = "IP/RTSP"


            if capture.connected:

                status = "CONNECTED"

            else:

                status = "RECONNECTING"


            cv2.putText(
                frame,
                "GOD'S EYE - REAL-TIME BORDER SURVEILLANCE",
                (15, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.68,
                (255, 255, 255),
                2
            )


            cv2.putText(
                frame,
                (
                    f"{args.camera_id} | "
                    f"{source_type} | "
                    f"{status} | "
                    f"FPS: {fps:.1f}"
                ),
                (15, 56),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (220, 220, 220),
                1
            )


            # ======================================================
            # GLOBAL ALERT BANNER
            # ======================================================

            if active_alerts:

                newest = list(
                    active_alerts.values()
                )[-1]


                if (
                    newest["event"]
                    == "VIRTUAL_FENCE_INTRUSION"
                ):

                    banner_color = (
                        0,
                        0,
                        180
                    )

                else:

                    banner_color = (
                        0,
                        160,
                        200
                    )


                y = (
                    frame.shape[0]
                    -
                    70
                )


                cv2.rectangle(
                    frame,
                    (10, y),
                    (
                        min(
                            frame.shape[1] - 10,
                            800
                        ),
                        frame.shape[0] - 10
                    ),
                    banner_color,
                    -1
                )


                cv2.putText(
                    frame,
                    newest["text"],
                    (
                        22,
                        frame.shape[0] - 35
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.60,
                    (255, 255, 255),
                    2
                )


            # ======================================================
            # SHOW
            # ======================================================

            cv2.imshow(
                Config.WINDOW_NAME,
                frame
            )


            key = (
                cv2.waitKey(1)
                &
                0xFF
            )


            # ======================================================
            # KEYBOARD CONTROLS
            # ======================================================

            if key in (
                ord("q"),
                ord("Q")
            ):

                break


            elif key in (
                ord("c"),
                ord("C")
            ):

                fence.clear()

                alert_system.clear()

                print(
                    "Fence unlocked and cleared."
                )


            elif key == 13:

                if fence.confirm():

                    print(
                        "🔒 Fence LOCKED."
                    )

                    print(
                        "Security alerts are ACTIVE."
                    )

                else:

                    print(
                        f"Need exactly 4 points. "
                        f"Current: {len(fence.points)}"
                    )


            elif key in (
                ord("r"),
                ord("R")
            ):

                fence.reset_tracking()

                tracker.reset()

                alert_system.clear()

                print(
                    "Tracking reset."
                )

                print(
                    "Locked fence remains unchanged."
                )


            elif key in (
                ord("s"),
                ord("S")
            ):

                filename = (
                    f"ibvap_alert_"
                    f"{int(time.time())}.jpg"
                )

                cv2.imwrite(
                    filename,
                    frame
                )

                print(
                    f"Saved screenshot: {filename}"
                )


    finally:

        capture.release()

        cv2.destroyAllWindows()


if __name__ == "__main__":

    main()