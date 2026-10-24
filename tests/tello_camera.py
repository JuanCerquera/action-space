# Copyright (c) 2026, Action Space team.
# SPDX-License-Identifier: Apache-2.0
"""Real-hardware check: show the Tello's camera feed. Does not fly.

Connect this machine to the Tello's Wi-Fi first (inbound UDP 8890 and
11111 must be allowed through the firewall), then:

    python tests/tello_camera.py

Press q in the video window to quit.
"""

import sys

import cv2
from djitellopy import Tello


def main() -> int:
    tello = Tello()
    tello.connect()
    print(f"[INFO] Connected. Battery {tello.get_battery()}%")
    tello.streamon()
    try:
        reader = tello.get_frame_read()
        while True:
            frame = reader.frame
            if frame is not None:
                # djitellopy delivers RGB; OpenCV displays BGR
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                status = f"bat {tello.get_battery()}%  temp {tello.get_temperature():.0f}"
                cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.imshow("Tello camera (q to quit)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    except KeyboardInterrupt:
        pass
    finally:
        tello.streamoff()
        cv2.destroyAllWindows()
        tello.end()
    return 0


if __name__ == "__main__":
    sys.exit(main())
