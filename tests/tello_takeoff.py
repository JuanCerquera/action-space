# Copyright (c) 2026, Action Space team.
# SPDX-License-Identifier: Apache-2.0
"""Real-hardware check: lift a DJI Tello, hover, and land.

Connect this machine to the Tello's Wi-Fi first, then:

    pip install djitellopy
    python tests/tello_takeoff.py --hover 5 --up 0

Clear at least 2 m around and above the drone. Press Ctrl+C at any
time to land immediately.
"""

import argparse
import sys
import time

from djitellopy import Tello


MIN_BATTERY = 20  # percent; Tello refuses or behaves poorly below ~10%


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hover", type=float, default=5.0, help="seconds to hover after takeoff")
    parser.add_argument("--up", type=int, default=0, help="extra climb after takeoff, cm (0 or 20-100)")
    args = parser.parse_args()
    if args.up and not 20 <= args.up <= 100:
        parser.error("--up must be 0 or between 20 and 100 cm")

    tello = Tello()
    tello.connect()
    battery = tello.get_battery()
    print(f"[INFO] Connected. Battery {battery}%, temp {tello.get_temperature()} C")
    if battery < MIN_BATTERY:
        print(f"[ERROR] Battery below {MIN_BATTERY}%; charge before flying.")
        tello.end()
        return 1

    flying = False
    try:
        flying = True  # set first: takeoff can time out after the drone is already up
        tello.takeoff()  # Tello climbs to ~0.8-1 m and holds
        print(f"[INFO] Airborne. Height {tello.get_height()} cm")
        if args.up:
            tello.move_up(args.up)
            print(f"[INFO] Climbed {args.up} cm. Height {tello.get_height()} cm")
        time.sleep(args.hover)
        tello.land()
        flying = False
        print("[INFO] Landed.")
        return 0
    except KeyboardInterrupt:
        print("[WARN] Interrupted; landing.")
        return 130
    except Exception as exc:
        print(f"[ERROR] {exc}; landing.")
        return 1
    finally:
        if flying:
            try:
                tello.land()
            except Exception:
                tello.emergency()  # cuts motors: last resort only
        tello.end()


if __name__ == "__main__":
    sys.exit(main())
