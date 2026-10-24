# Copyright (c) 2026, Action Space team.
# SPDX-License-Identifier: Apache-2.0
"""Real-hardware check: fly a Tello from the keyboard with a live camera feed.

Connect this machine to the Tello's Wi-Fi first (inbound UDP 8890 and
11111 must be allowed through the firewall), then:

    python tests/tello_control.py --speed 30

Click the video window so it has keyboard focus. Hold a key to move;
the drone stops when keys are released.

    t takeoff        l land          Esc  land and quit
    w/s forward/back a/d left/right  r/f  up/down   q/e  turn left/right
    x EMERGENCY: cuts all motors, the drone drops
"""

import argparse
import sys
import threading
import time

import cv2
from djitellopy import Tello


MIN_BATTERY = 20  # percent; refuse takeoff below this
HOLD_S = 0.15  # stop moving if no key repeat arrives within this window

# key -> (left/right, forward/back, up/down, yaw) direction
MOVES = {
    ord("w"): (0, 1, 0, 0),
    ord("s"): (0, -1, 0, 0),
    ord("a"): (-1, 0, 0, 0),
    ord("d"): (1, 0, 0, 0),
    ord("r"): (0, 0, 1, 0),
    ord("f"): (0, 0, -1, 0),
    ord("q"): (0, 0, 0, -1),
    ord("e"): (0, 0, 0, 1),
}
HELP = "t takeoff  l land  wasd move  r/f up/down  q/e turn  Esc quit  x EMERGENCY"


def run_blocking(busy: threading.Event, command, results: list):
    """Run a command that waits for the drone's reply off the UI thread, so video keeps updating."""

    def worker():
        try:
            command()
            results.append("")
        except Exception as exc:
            results.append(f"{command.__name__} failed: {exc}")
            print(f"[ERROR] {command.__name__} failed: {exc}")
        finally:
            busy.clear()

    busy.set()
    threading.Thread(target=worker, daemon=True).start()


def draw(frame, tello: Tello, flying: bool, message: str):
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # djitellopy delivers RGB
    state = "FLYING" if flying else "landed"
    status = f"{state}  bat {tello.get_battery()}%  h {tello.get_height()}cm  temp {tello.get_temperature():.0f}"
    cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(frame, HELP, (10, frame.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    if message:
        cv2.putText(frame, message, (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)
    cv2.imshow("Tello control", frame)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--speed", type=int, default=30, help="movement speed, 10-100 (default 30)")
    args = parser.parse_args()
    speed = max(10, min(100, args.speed))

    tello = Tello()
    tello.connect()
    print(f"[INFO] Connected. Battery {tello.get_battery()}%")
    tello.streamon()
    reader = tello.get_frame_read()

    flying = False
    busy = threading.Event()  # set while takeoff/land waits for the drone's reply
    results: list[str] = []
    moving = False
    last_key_time = 0.0
    message = "click this window, then press t to take off"
    try:
        while True:
            if reader.frame is not None:
                draw(reader.frame, tello, flying, message)
            key = cv2.waitKey(20) & 0xFF

            if key == ord("x"):
                tello.emergency()
                flying = False
                message = "EMERGENCY STOP"
                print("[WARN] Emergency stop.")
            elif key == 27:  # Esc
                break
            elif key == ord("t") and not flying and not busy.is_set():
                if tello.get_battery() < MIN_BATTERY:
                    message = f"battery below {MIN_BATTERY}%, not taking off"
                else:
                    flying = True  # set first: takeoff can time out after the drone is already up
                    message = "taking off..."
                    run_blocking(busy, tello.takeoff, results)
            elif key == ord("l") and flying and not busy.is_set():
                tello.send_rc_control(0, 0, 0, 0)
                flying = False
                message = "landing..."
                run_blocking(busy, tello.land, results)
            elif key in MOVES and flying and not busy.is_set():
                lr, fb, ud, yaw = (speed * v for v in MOVES[key])
                tello.send_rc_control(lr, fb, ud, yaw)
                moving = True
                last_key_time = time.monotonic()

            if results:
                message = results.pop()

            if moving and time.monotonic() - last_key_time > HOLD_S:
                tello.send_rc_control(0, 0, 0, 0)
                moving = False
    except KeyboardInterrupt:
        print("[WARN] Interrupted; landing.")
    except Exception as exc:
        print(f"[ERROR] {exc}; landing.")
    finally:
        if flying:
            try:
                tello.send_rc_control(0, 0, 0, 0)
                tello.land()
            except Exception:
                tello.emergency()  # cuts motors: last resort only
        tello.streamoff()
        cv2.destroyAllWindows()
        tello.end()
    return 0


if __name__ == "__main__":
    sys.exit(main())
