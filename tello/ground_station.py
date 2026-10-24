# Copyright (c) 2025, Action Space hackathon team.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Ground station: fly the defender Tellos with the block controller in the real arena.

Loop (20-30 Hz, everything in the arena frame of ``config/arena.json``)::

    RealSense depth --> blobs --> near a defender? --yes--> correct that defender's position
                                                 --no---> attacker tracker (Kalman + Hungarian)
    Tello camera (optional) --> AprilTags --> defender position + heading fix
    Tello height sensor (tof) --> defender height
    defender positions + attacker tracks --> BlockController --> world velocity
        --> body-frame rc (left/right, forward/back, up/down, yaw) --> djitellopy send_rc_control

Between fixes each defender's position is predicted from the commands it was sent (same lag model as
the sim). A defender with no fix for ``--max_no_fix`` seconds hovers in place; after
``--lost_land`` seconds it lands. A defender that drops below 20 cm or stops sending state is
counted as down (the controller re-plans with the rest, as in the sim).

Drones are listed in ``config/tellos.json`` (written with an example on first run). Several Tellos at
once need Tello EDUs in station mode (all on one router, each with its own IP); a standard Tello only
works as its own hotspot, so the laptop can fly one per Wi-Fi adapter (djitellopy supports one).

Keys (in this terminal):  g = start defending   h = hover in place   l or SPACE = land all and quit
x twice = emergency motor stop (drones drop!)   1/2/3 = toggle the mesh of slot 0/1/2   Ctrl+C = land all

Mesh carriers (``"net": true``): the RealSense sees the strip as part of the carrier's blob, so blobs along
the strip are given to the carrier (never mistaken for an attacker) and only correct its x/y. A taped
mesh is single use: when an attacker track ends right at a carrier's strip, that mesh is marked used
and the carrier goes back to normal blocking (or toggle it by hand with 1/2/3).

.. code-block:: bash

    python tello/ground_station.py --sim                   # whole loop with simulated drones, no hardware
    python tello/ground_station.py --speed_test            # once per drone type: measure rc units per m/s
    python tello/ground_station.py                         # real flight (RealSense + Tellos)
    python tello/ground_station.py --no_realsense          # defenders only, localized by their cameras
    python tello/ground_station.py --relay                 # relay blocking strategy
    python tello/ground_station.py --zone 0.6 3.0 -2 2 0.6 2.4   # zone 1v1 strategy (defense box)

Settings from ``scripts/tune_block.py`` (``config/block_tuned.json``) are used when present.
Every run is recorded to ``media/replay/flight_<time>.json``; ``python scripts/build_replay.py``
adds it to the 2D/3D browser replays.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import threading
import time

import numpy as np
from scipy.optimize import linear_sum_assignment

try:  # works both as ``tello.ground_station`` and as a script in ``tello/``
    from .controllers import BlockController, tuned_settings_path
    from .localizer import AprilTagLocalizer, load_arena
    from .realsense_tracker import MultiTargetTracker, RealSenseAttackerTracker
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from controllers import BlockController, tuned_settings_path  # noqa: E402
    from localizer import AprilTagLocalizer, load_arena  # noqa: E402
    from realsense_tracker import MultiTargetTracker, RealSenseAttackerTracker  # noqa: E402

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
DRONES_CFG = os.path.join(REPO, "config", "tellos.json")
TUNED_CFG = tuned_settings_path(REPO)  # tuned at real Tello speed for this arena map
NUM_SLOTS, NUM_ATTACKERS = 3, 2  # the controller always plans for 3 defenders; unused slots count as down

EXAMPLE_DRONES = {
    "_notes": [
        "One entry per defender Tello. slot = which arena.json defender_starts spot it takes off from (0-2).",
        "host: Tello EDU in station mode = its IP on the router; standard Tello = 192.168.10.1 (one drone only).",
        "yaw_deg: where the nose (camera) points at takeoff, in the arena frame: 180 = facing the goal wall",
        "(the camera then sees tags 10 and 5 for localization), 0 = facing into the room.",
        "camera: false | \"forward\" | \"down\": stream video and localize on AprilTags (each EDU needs its own video_port).",
        "  forward = nose camera (sees wall/panel tags); down = Tello EDU downward camera (sees the floor tags below it).",
        "camera_yaml: per-drone forward-camera calibration (config/camera/tello_N.yaml, from the team's calibrate tool).",
        "down_camera_yaml: downward-camera calibration; without one a rough 320x240 guess is used (calibrate it!).",
        "down_yaw_offset_deg: heading of the drone nose relative to the top of the downward image (0 if the top is the nose).",
        "rc_per_mps: rc units (-100..100) per 1 m/s as [right, forward, up]: from the team's motion fit",
        "  (drone_deployment_infra/control/tello_motion_fit.yaml: 0.008, 0.00731, 0.006 m/s per rc) -> full stick ~0.73 m/s.",
        "  Re-check with --speed_test.",
        "net: true = this drone carries a hanging mesh strip; it then flies above the attackers so they meet the mesh.",
    ],
    "rc_per_mps": [125.0, 137.0, 167.0],
    "drones": [
        {"name": "D1", "host": "192.168.10.1", "slot": 1, "yaw_deg": 180, "camera": "down", "video_port": 11111, "net": False,
         "camera_yaml": "config/camera/tello_3.yaml"},
    ],
}


NET_LENGTH = 0.8  # hanging mesh strip length [m], as in the sim
# Tello EDU downward camera: 320x240 grey. ROUGH GUESS (~70 deg field of view) until it is calibrated with the
# team's ChArUco tool (drone_deployment_infra/cal_and_map/scripts/03_calibrate.py) with downvision on.
DOWN_K_GUESS = np.array([[228.0, 0.0, 160.0], [0.0, 228.0, 120.0], [0.0, 0.0, 1.0]])
DOWN_SIZE_GUESS = (320, 240)


def load_camera_yaml(path: str):
    """(K, dist, (width, height)) from a team calibration file (config/camera/tello_N.yaml)."""
    import yaml

    full = path if os.path.isabs(path) else os.path.join(REPO, path)
    with open(full) as f:
        c = yaml.safe_load(f)
    return np.array(c["camera_matrix"], float), np.array(c["dist_coeffs"], float), (int(c["image_width"]), int(c["image_height"]))


def scaled_K(K: np.ndarray, calib_size, frame_size) -> np.ndarray:
    """Intrinsics for a frame of another resolution than the calibration (same field of view)."""
    sx, sy = frame_size[0] / calib_size[0], frame_size[1] / calib_size[1]
    return np.diag([sx, sy, 1.0]) @ K


def mesh_dist(p: np.ndarray, carrier_pos: np.ndarray) -> np.ndarray:
    """Distance from points p (M,3) to the mesh strip hanging below a carrier."""
    a = carrier_pos - np.array([0.0, 0.0, 0.08])
    ab = np.array([0.0, 0.0, -(NET_LENGTH - 0.08)])
    t = np.clip(((p - a) @ ab) / (ab @ ab), 0.0, 1.0)
    return np.linalg.norm(p - (a + t[:, None] * ab), axis=-1)


def wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def world_to_rc(v: np.ndarray, yaw: float, rc_per_mps, rc_limit: int) -> tuple[int, int, int]:
    """World velocity [m/s] -> Tello rc (left_right, forward_backward, up_down) for a nose heading ``yaw``.

    ``rc_per_mps`` is one number or [right, forward, up] (the team's motion fit: ~125, 137, 167).
    """
    fwd = np.array([math.cos(yaw), math.sin(yaw)])
    right = np.array([math.sin(yaw), -math.cos(yaw)])
    rc = np.array([v[:2] @ right, v[:2] @ fwd, v[2]]) * np.broadcast_to(np.asarray(rc_per_mps, float), (3,))
    return tuple(int(x) for x in np.clip(np.round(rc), -rc_limit, rc_limit))


# ----------------------------------------------------------------------------- simulated hardware
class FakeWorld:
    """Stand-in for the real arena (``--sim``): defender Tellos, two attackers and a RealSense."""

    def __init__(self, arena: dict, rc_per_mps: float, seed: int = 0):
        self.rng = np.random.default_rng(seed)
        self.rc_per_mps = rc_per_mps
        zone = arena["goal"]["zone"]
        self.zone_lo = np.array([zone[a][0] for a in "xyz"])
        self.zone_hi = np.array([zone[a][1] for a in "xyz"])
        self.goal = (self.zone_lo + self.zone_hi) / 2
        self.spawn = arena["attacker_spawn"]
        self.drones: dict[str, dict] = {}
        self.att_pos = np.zeros((NUM_ATTACKERS, 3))
        self.att_vel = np.zeros((NUM_ATTACKERS, 3))
        self.att_stun = np.zeros(NUM_ATTACKERS)
        self.att_offset = np.zeros(NUM_ATTACKERS)
        self.breaches = self.bumps = 0
        for j in range(NUM_ATTACKERS):
            self._respawn(j)

    def _respawn(self, j: int) -> None:
        s = self.spawn
        self.att_pos[j] = [self.rng.uniform(*s["x"]), self.rng.uniform(*s["y"]), self.rng.uniform(*s["z"])]
        self.att_vel[j] = 0.0
        self.att_offset[j] = self.rng.uniform(-1.5, 1.5)  # comes in on a curve from one side

    def add(self, host: str, start, yaw: float) -> None:
        self.drones[host] = {"p": np.array(start, float), "v": np.zeros(3), "cmd": np.zeros(3), "yaw": yaw,
                             "flying": False, "drift": self.rng.normal(0, 0.03, 3) * [1, 1, 0]}

    def step(self, dt: float) -> None:
        for d in self.drones.values():
            if d["flying"]:
                d["v"] += (d["cmd"] - d["v"]) * min(dt / 0.3, 1.0)
                d["p"] += (d["v"] + d["drift"]) * dt
                d["yaw"] += self.rng.normal(0, 0.002)
        flying = [d["p"] for d in self.drones.values() if d["flying"]]
        for j in range(NUM_ATTACKERS):
            a = self.att_pos[j]
            if self.att_stun[j] > 0:
                self.att_stun[j] -= dt
            else:
                to_goal = self.goal - a
                dist = np.linalg.norm(to_goal)
                aim = self.goal + np.array([0.0, self.att_offset[j] * min(dist / 4, 1.0), 0.0])
                want = (aim - a) / max(np.linalg.norm(aim - a), 1e-6) * 0.8
                self.att_vel[j] += (want - self.att_vel[j]) * min(dt / 0.3, 1.0)
            for p in flying:  # a touch bumps the attacker away, it recovers and comes back
                d = a - p
                if np.linalg.norm(d) < 0.22:
                    self.att_vel[j] = d / max(np.linalg.norm(d), 1e-6) * 1.0
                    self.att_stun[j] = 0.7
                    self.bumps += 1
            self.att_pos[j] = a + self.att_vel[j] * dt
            if np.all((self.att_pos[j] >= self.zone_lo) & (self.att_pos[j] <= self.zone_hi)):
                self.breaches += 1
                print(f"[sim] BREACH by attacker {j} (total {self.breaches})")
                self._respawn(j)

    def detections(self) -> list[tuple[np.ndarray, int]]:
        pts = [d["p"] for d in self.drones.values() if d["flying"]] + list(self.att_pos)
        return [(p + self.rng.normal(0, 0.03, 3), 100) for p in pts if self.rng.random() > 0.05]


class FakeTello:
    """The subset of djitellopy.Tello the ground station uses, flying in a :class:`FakeWorld`."""

    def __init__(self, world: FakeWorld, host: str):
        self.world, self.host = world, host

    @property
    def d(self) -> dict:
        return self.world.drones[self.host]

    def connect(self):
        pass

    def get_battery(self) -> int:
        return 87

    def get_current_state(self) -> dict:
        d = self.d
        return {"tof": int(d["p"][2] * 100) + 10, "vgx": int(d["v"][0] * 10), "vgy": int(d["v"][1] * 10), "bat": 87}

    def takeoff(self):
        self.d["flying"] = True
        self.d["p"][2] = 1.0

    def land(self):
        self.d["flying"] = False
        self.d["p"][2] = 0.0
        self.d["v"][:] = 0.0

    emergency = land

    def send_rc_control(self, lr: int, fb: int, ud: int, yaw: int):
        d = self.d
        fwd = np.array([math.cos(d["yaw"]), math.sin(d["yaw"])])
        right = np.array([math.sin(d["yaw"]), -math.cos(d["yaw"])])
        d["cmd"][:2] = (lr * right + fb * fwd) / self.world.rc_per_mps
        d["cmd"][2] = ud / self.world.rc_per_mps
        d["yaw"] += math.radians(-yaw) * 0.02  # rc yaw > 0 turns clockwise

    def end(self):
        pass


# ----------------------------------------------------------------------------- defenders
class Defender:
    """One Tello: link, position estimate (prediction + fixes) and optional AprilTag localization thread."""

    def __init__(self, cfg: dict, start, tello, arena: dict, tau: float = 0.3):
        self.name, self.slot, self.tello = cfg["name"], int(cfg["slot"]), tello
        self.p = np.array(start, float)
        self.v = np.zeros(3)
        self.cmd = np.zeros(3)
        self.yaw = math.radians(cfg.get("yaw_deg", 180))
        self.yaw_hold = self.yaw
        self.tau = tau
        self.alive = True
        self.flying = False
        self.takeoff_time = 0.0
        self.last_fix = time.time()
        self.last_yaw_fix = -1e9
        self.low_since = None
        self._state_obj, self._state_time = None, time.time()
        self.battery = None
        mode = cfg.get("camera", False)
        self.camera = {True: "forward", False: None}.get(mode, mode)  # None | "forward" | "down"
        self._fix = None  # (t, est) from the camera thread
        self._fix_used = None
        self.arena = arena
        if self.camera == "down":
            if cfg.get("down_camera_yaml"):
                self.cam_K, self.cam_dist, self.cam_size = load_camera_yaml(cfg["down_camera_yaml"])
            else:
                self.cam_K, self.cam_dist, self.cam_size = DOWN_K_GUESS, np.zeros(5), DOWN_SIZE_GUESS
                print(f"[{self.name}] downward camera NOT calibrated: using a rough guess (heights/positions may be off)")
            self.down_yaw_offset = math.radians(cfg.get("down_yaw_offset_deg", 0.0))
        elif self.camera == "forward":
            if cfg.get("camera_yaml"):
                self.cam_K, self.cam_dist, self.cam_size = load_camera_yaml(cfg["camera_yaml"])
            else:
                from localizer import TELLO_DIST, TELLO_IMAGE_SIZE, TELLO_K

                self.cam_K, self.cam_dist, self.cam_size = TELLO_K, TELLO_DIST, TELLO_IMAGE_SIZE
        if self.camera:
            threading.Thread(target=self._camera_loop, daemon=True).start()

    # -- sensing --
    def state(self) -> dict:
        try:
            s = self.tello.get_current_state()
        except Exception:
            s = None
        if s is not None and s is not self._state_obj:  # djitellopy replaces the dict on every packet
            self._state_obj, self._state_time = s, time.time()
        return s or {}

    def link_ok(self) -> bool:
        return time.time() - self._state_time < 2.0

    def _camera_loop(self) -> None:
        import cv2

        try:
            if self.camera == "down":
                from djitellopy import Tello

                self.tello.set_video_direction(Tello.CAMERA_DOWNWARD)  # EDU only: stream the belly camera
            self.tello.streamon()
            reader = self.tello.get_frame_read()
        except Exception as e:  # noqa: BLE001
            print(f"[{self.name}] camera failed: {e}")
            return
        last, localizer = None, None
        while self.alive:
            frame = reader.frame
            if frame is None or frame is last:
                time.sleep(0.01)
                continue
            last = frame
            if localizer is None:  # intrinsics scaled to the resolution actually streamed
                size = (frame.shape[1], frame.shape[0])
                localizer = AprilTagLocalizer(self.arena, scaled_K(self.cam_K, self.cam_size, size), self.cam_dist)
                print(f"[{self.name}] {self.camera} camera streaming at {size[0]}x{size[1]}")
            est = localizer.estimate(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            if est is not None and est["reproj_err"] < 3.0:
                if self.camera == "down":
                    # looking at the floor: the nose is toward the top of the image (-y of the camera)
                    up = -est["R_wc"][:, 1]
                    est["yaw"] = wrap(math.atan2(up[1], up[0]) + self.down_yaw_offset)
                self._fix = (time.time(), est)

    def predict(self, dt: float) -> None:
        if not self.flying:
            return
        self.v += (self.cmd - self.v) * min(dt / self.tau, 1.0)
        self.p += self.v * dt
        tof = self.state().get("tof")
        if tof is not None and 15 < tof < 400:  # height sensor, cm above the floor
            self.p[2] += 0.5 * (tof / 100.0 - self.p[2])

    def correct(self, z: np.ndarray, gain: float) -> None:
        self.p += gain * (z - self.p)
        self.last_fix = time.time()

    def apply_camera_fix(self) -> None:
        fix = self._fix
        if fix is None or fix is self._fix_used:
            return
        self._fix_used = fix
        t, est = fix
        if time.time() - t > 0.3:
            return  # too old to use
        self.correct(np.asarray(est["position"], float), 0.6)
        self.yaw = est["yaw"]
        self.last_yaw_fix = t

    def rc(self, vel: np.ndarray, rc_per_mps: float, rc_limit: int) -> tuple[int, int, int, int]:
        lr, fb, ud = world_to_rc(vel, self.yaw, rc_per_mps, rc_limit)
        yaw_rc = 0
        if time.time() - self.last_yaw_fix < 2.0:  # hold the takeoff heading, only with a fresh camera heading
            err = math.degrees(wrap(self.yaw_hold - self.yaw))
            if abs(err) > 5:
                yaw_rc = int(np.clip(-1.0 * err, -25, 25))  # rc yaw > 0 turns clockwise (heading decreases)
        return lr, fb, ud, yaw_rc


# ----------------------------------------------------------------------------- ground station
class Keys:
    """Non-blocking single-key input (Windows console; elsewhere only Ctrl+C works)."""

    def __init__(self):
        try:
            import msvcrt

            self.msvcrt = msvcrt
        except ImportError:
            self.msvcrt = None

    def get(self) -> str | None:
        if self.msvcrt and self.msvcrt.kbhit():
            return self.msvcrt.getwch().lower()
        return None


def load_drones(path: str) -> dict:
    if not os.path.exists(path):
        with open(path, "w") as f:
            json.dump(EXAMPLE_DRONES, f, indent=2)
        print(f"[gs] wrote an example {path}: edit it for your drones")
    with open(path) as f:
        return json.load(f)


def load_tuned(use: bool) -> dict:
    if use and os.path.exists(TUNED_CFG):
        with open(TUNED_CFG) as f:
            tuned = json.load(f)
        print(f"[gs] using auto-tuned settings ({tuned.get('win_rate', 0):.0%} sim win rate): {TUNED_CFG}")
        return tuned["settings"]
    print("[gs] using the hand-tuned default settings")
    return {}


def make_tello(cfg: dict, world: FakeWorld | None):
    if world is not None:
        return FakeTello(world, cfg["host"])
    from djitellopy import Tello

    port = int(cfg.get("video_port", 11111))
    tello = Tello(host=cfg["host"], vs_udp=port)
    tello.connect()
    if cfg.get("camera") and port != 11111:
        tello.change_vs_udp(port)  # EDU only: send this drone's video to its own port
    return tello


def takeoff_all(defenders: list[Defender]) -> None:
    def go(d: Defender):
        d.tello.takeoff()
        d.flying, d.takeoff_time = True, time.time()

    threads = [threading.Thread(target=go, args=(d,)) for d in defenders]
    for t in threads:
        t.start()
    for t in threads:
        t.join()


def land_all(defenders: list[Defender]) -> None:
    for d in defenders:
        if d.flying:
            try:
                d.tello.send_rc_control(0, 0, 0, 0)
                d.tello.land()
            except Exception as e:  # noqa: BLE001
                print(f"[{d.name}] land failed: {e}")
            d.flying = False


def speed_test(args, drones_cfg: dict, arena: dict, world: FakeWorld | None) -> None:
    """Fly the first drone forward and back at two rc levels and measure its speed (Tello's own vgx/vgy)."""
    cfg = drones_cfg["drones"][0]
    tello = make_tello(cfg, world)
    if world is not None:
        world.add(cfg["host"], arena["defender_starts"][int(cfg["slot"])], math.radians(cfg.get("yaw_deg", 180)))
    print(f"[speed] {cfg['name']} battery {tello.get_battery()}%; needs ~3 m clear in front. Taking off in 3 s...")
    time.sleep(3)
    tello.takeoff()
    ratios = []
    try:
        for rc in (20, 40):
            for sign in (1, -1):
                speeds, t0 = [], time.time()
                while time.time() - t0 < 1.5:
                    tello.send_rc_control(0, sign * rc, 0, 0)
                    if world is not None:
                        world.step(0.05)
                    s = tello.get_current_state()
                    if time.time() - t0 > 0.8:  # past the acceleration
                        speeds.append(math.hypot(s.get("vgx", 0), s.get("vgy", 0)) / 10.0)  # dm/s -> m/s
                    time.sleep(0.05)
                tello.send_rc_control(0, 0, 0, 0)
                time.sleep(1.5)
                v = float(np.median(speeds)) if speeds else 0.0
                print(f"[speed] rc {sign * rc:+d}: {v:.2f} m/s")
                if v > 0.05:
                    ratios.append(rc / v)
    finally:
        tello.send_rc_control(0, 0, 0, 0)
        tello.land()
    if ratios:
        print(f"[speed] forward rc_per_mps = {np.mean(ratios):.1f} (team fit: 137)  -> update {DRONES_CFG}")
    else:
        print("[speed] no speed reading; check the Tello state stream")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sim", action="store_true", help="simulated drones and RealSense, no hardware")
    ap.add_argument("--speed_test", action="store_true", help="measure rc_per_mps with the first drone and quit")
    ap.add_argument("--no_realsense", action="store_true", help="no RealSense: no attackers, camera fixes only")
    ap.add_argument("--drones", default=DRONES_CFG)
    ap.add_argument("--speed_scale", type=float, default=1.0, help="fraction of the sim's top speed (use 0.3 for a first slow test)")
    ap.add_argument("--rc_limit", type=int, default=100, help="hard cap on any rc channel (-100..100); full stick is only ~0.73 m/s")
    ap.add_argument("--rate", type=float, default=20.0, help="control rate [Hz]")
    ap.add_argument("--max_no_fix", type=float, default=1.5, help="hover a defender with no position fix this long [s]")
    ap.add_argument("--lost_land", type=float, default=6.0, help="land a defender with no position fix this long [s]")
    ap.add_argument("--min_battery", type=int, default=25)
    ap.add_argument("--no_tuned", action="store_true", help="ignore config/block_tuned.json")
    ap.add_argument("--relay", action="store_true", help="relay blocking: predicted intercepts, beaten marker retreats")
    ap.add_argument("--kamikaze", action="store_true", help="kamikaze keeper: the goalkeeper rams attackers near the goal")
    ap.add_argument("--zone", type=float, nargs=6, default=None, metavar=("X0", "X1", "Y0", "Y1", "Z0", "Z1"),
                    help="zone 1v1: defenders stay inside this defense box [m], e.g. 0.6 3.0 -2 2 0.6 2.4")
    ap.add_argument("--duration", type=float, default=0.0, help="--sim only: stop after this many seconds (0 = keys)")
    args = ap.parse_args()

    arena = load_arena()
    drones_cfg = load_drones(args.drones)
    rc_per_mps = drones_cfg.get("rc_per_mps", [125.0, 137.0, 167.0])
    world = FakeWorld(arena, float(np.mean(rc_per_mps))) if args.sim else None
    if args.speed_test:
        speed_test(args, drones_cfg, arena, world)
        return

    # -- drones --
    starts = np.array(arena["defender_starts"], float)
    defenders: list[Defender] = []
    for cfg in drones_cfg["drones"]:
        start = starts[int(cfg["slot"])]
        if world is not None:
            world.add(cfg["host"], start, math.radians(cfg.get("yaw_deg", 180)))
        tello = make_tello(cfg, world)
        d = Defender(cfg, start, tello, arena)
        d.battery = tello.get_battery()
        print(f"[gs] {d.name} ({cfg['host']}) slot {d.slot} battery {d.battery}%")
        if d.battery < args.min_battery:
            raise SystemExit(f"[gs] {d.name} battery {d.battery}% < {args.min_battery}%: swap it")
        defenders.append(d)
    assert len({d.slot for d in defenders}) == len(defenders), "two drones share a slot"

    # -- attackers --
    cam = None
    tracker = MultiTargetTracker(arena, max_attackers=NUM_ATTACKERS)
    if not args.sim and not args.no_realsense:
        cam = RealSenseAttackerTracker(arena)
        if cam.T_world_cam is None:
            cam.calibrate()
        print("[gs] building the RealSense background: keep ALL drones out of view (on the floor is fine)...")
        cam.build_background()
        tracker = cam.tracker

    gains = np.broadcast_to(np.asarray(rc_per_mps, float), (3,))
    top = args.rc_limit / gains  # fastest each axis can go within the rc cap [m/s]
    max_xy, max_z = min(1.5 * args.speed_scale, float(top[:2].min())), min(1.0 * args.speed_scale, float(top[2]))
    print(f"[gs] top speed {max_xy:.2f} m/s horizontal, {max_z:.2f} m/s vertical (rc cap {args.rc_limit})")
    nets = [False] * NUM_SLOTS
    for cfg in drones_cfg["drones"]:
        nets[int(cfg["slot"])] = bool(cfg.get("net", False))
    if any(nets):
        print(f"[gs] mesh carriers: {[d['name'] for d in drones_cfg['drones'] if d.get('net')]}")
    ctrl = BlockController(arena, NUM_SLOTS, NUM_ATTACKERS, max_speed_xy=max_xy, max_speed_z=max_z,
                           **{**load_tuned(not args.no_tuned), "net_carriers": nets, "relay": args.relay, "kamikaze": args.kamikaze,
                              "zone": None if args.zone is None else tuple(zip(args.zone[0::2], args.zone[1::2]))})
    print(f"[gs] strategy: {'relay' if args.relay else 'zone 1v1' if args.zone else 'block'}"
          f"{' + kamikaze keeper' if args.kamikaze else ''}"
          f"{' + mesh' if any(nets) else ''}")
    room = arena["room"]
    room_lo = np.array([room[a][0] for a in "xyz"]) + 0.3
    room_hi = np.array([room[a][1] for a in "xyz"]) - 0.3
    room_center = (room_lo + room_hi) / 2

    # -- recording (same format as the sim recordings, for scripts/build_replay.py) --
    stamp = time.strftime("%Y%m%d_%H%M%S")
    name = f"{'simflight' if args.sim else 'flight'}_{stamp}"
    frames = []
    rec_path = os.path.join(REPO, "media", "replay", f"{name}.json")

    keys = Keys()
    print("[gs] keys: g = defend, h = hover, l/SPACE = land all, x x = emergency stop, 1/2/3 = toggle mesh, Ctrl+C = land")
    last_tracks: dict[int, np.ndarray] = {}
    print("[gs] taking off in 3 s...")
    time.sleep(0 if args.sim else 3)
    takeoff_all(defenders)
    mode, x_pressed = "hover", 0.0
    slot_of_track: dict[int, int] = {}
    period = 1.0 / args.rate
    t_prev = time.time()
    t_start = t_prev
    last_print = 0.0
    try:
        while True:
            now = time.time()
            dt = now - t_prev
            if cam is None and dt < period:
                time.sleep(period - dt)
                now = time.time()
                dt = now - t_prev
            t_prev = now
            if world is not None:
                world.step(dt)
                if args.duration and now - t_start > args.duration:
                    break

            # -- keys --
            k = keys.get()
            if k == "g":
                mode = "defend"
                print("[gs] DEFEND")
            elif k == "h":
                mode = "hover"
                print("[gs] HOVER")
            elif k in ("l", " "):
                print("[gs] LAND ALL")
                break
            elif k == "x":
                if now - x_pressed < 1.0:
                    print("[gs] EMERGENCY STOP")
                    for d in defenders:
                        d.tello.emergency()
                        d.flying = False
                    break
                x_pressed = now
                print("[gs] press x again within 1 s for an emergency motor stop")
            elif k in ("1", "2", "3"):
                nets[int(k) - 1] = not nets[int(k) - 1]
                ctrl.net = np.array(nets)
                print(f"[gs] slot {int(k) - 1} mesh {'ON' if nets[int(k) - 1] else 'OFF (used up)'}")
            if args.sim and mode == "hover" and now - t_start > 2.0:
                mode = "defend"  # unattended sim run

            # -- sense: predict, then correct with RealSense blobs near defenders and camera fixes --
            for d in defenders:
                d.predict(dt)
            if cam is not None:
                _, depth, t_cam = cam.grab()
                dets = cam.detector.detect(depth)
            elif world is not None:
                dets, t_cam = world.detections(), now
            else:
                dets, t_cam = [], now
            flying = [d for d in defenders if d.flying and d.alive]
            # defenders and attacker tracks compete for the blobs in one assignment: a blocking defender
            # flies right next to its attacker, so "drop every blob near a defender" would lose the attacker
            Z = np.array([np.asarray(p, float) for p, _ in dets]).reshape(-1, 3)
            att_dets = list(dets)
            if len(Z) and flying:
                lag = 0.0 if tracker._t is None else max(0.0, t_cam - tracker._t)
                owners = [d.p for d in flying] + [tr.x[:3] + tr.x[3:] * lag for tr in tracker.tracks]
                cost = np.linalg.norm(np.array(owners)[:, None] - Z[None], axis=-1)
                for r, d in enumerate(flying):
                    if nets[d.slot]:  # the strip and the drone show up as one tall blob
                        cost[r] = np.minimum(cost[r], mesh_dist(Z, d.p))
                taken = set()
                for r, c in zip(*linear_sum_assignment(cost)):
                    if r < len(flying) and cost[r, c] < 0.5:
                        d = flying[r]
                        z = Z[c].copy()
                        if nets[d.slot]:
                            z[2] = d.p[2]  # the blob center sits down the strip: use it for x/y only
                        d.correct(z, 0.5)
                        taken.add(c)
                # any other blob along a strip is the strip
                for d in flying:
                    if nets[d.slot]:
                        taken |= {c for c in range(len(Z)) if mesh_dist(Z[c:c + 1], d.p)[0] < 0.2}
                att_dets = [det for c, det in enumerate(dets) if c not in taken]
            for d in defenders:
                if d.camera:
                    d.apply_camera_fix()
            tracks = tracker.update(att_dets, t_cam) if (cam or world) else []

            # -- defenders down: no state stream, or on the floor after takeoff --
            for d in flying:
                s = d.state()
                low = s.get("tof", 100) < 20 and now - d.takeoff_time > 3.0
                d.low_since = (d.low_since or now) if low else None
                if not d.link_ok() or (d.low_since and now - d.low_since > 1.0):
                    print(f"[gs] {d.name} is DOWN ({'link lost' if not d.link_ok() else 'on the floor'})")
                    d.alive = False
                    land_all([d])

            # -- controller input: 3 slots, attackers in stable slots by track id --
            def_pos = starts.copy()
            def_vel = np.zeros((NUM_SLOTS, 3))
            def_alive = np.zeros(NUM_SLOTS, bool)
            for d in defenders:
                def_pos[d.slot], def_vel[d.slot], def_alive[d.slot] = d.p, d.v, d.flying and d.alive
            ids = {tr["id"] for tr in tracks}
            # a taped mesh is used up when an attacker track ends right at it (caught and dropped)
            for tid, pos in last_tracks.items():
                if tid in ids:
                    continue
                for d in flying:
                    if nets[d.slot] and mesh_dist(pos[None], d.p)[0] < 0.4:
                        nets[d.slot] = False
                        ctrl.net = np.array(nets)
                        print(f"[gs] {d.name} caught an attacker: mesh used up, back to normal blocking")
            last_tracks = {tr["id"]: tr["pos"].copy() for tr in tracks}
            slot_of_track = {i: s for i, s in slot_of_track.items() if i in ids}
            for tr in tracks:
                if tr["id"] not in slot_of_track:
                    free = [s for s in range(NUM_ATTACKERS) if s not in slot_of_track.values()]
                    if free:
                        slot_of_track[tr["id"]] = free[0]
            att_pos = np.zeros((NUM_ATTACKERS, 3))
            att_vel = np.zeros((NUM_ATTACKERS, 3))
            att_alive = np.zeros(NUM_ATTACKERS, bool)
            for tr in tracks:
                s = slot_of_track.get(tr["id"])
                if s is not None:
                    att_pos[s], att_vel[s], att_alive[s] = tr["pos"], tr["vel"], True

            if mode == "defend":
                vel = ctrl(def_pos[None], def_vel[None], att_pos[None], att_vel[None], att_alive[None], def_alive[None])[0]
            else:
                vel = np.zeros((NUM_SLOTS, 3))

            # -- safety and send --
            for d in flying:
                if not d.alive:
                    continue
                v = vel[d.slot].copy()
                no_fix = now - d.last_fix
                if no_fix > args.lost_land:
                    print(f"[gs] {d.name}: no position fix for {no_fix:.0f} s, landing it")
                    d.alive = False
                    land_all([d])
                    continue
                if no_fix > args.max_no_fix:
                    v[:] = 0.0  # hover in place until a fix comes back
                outside = (d.p < room_lo) | (d.p > room_hi)
                if outside.any():  # geofence: head back toward the room center
                    v = 0.3 * (room_center - d.p) / max(np.linalg.norm(room_center - d.p), 1e-6)
                d.cmd = v
                try:
                    d.tello.send_rc_control(*d.rc(v, rc_per_mps, args.rc_limit))
                except Exception as e:  # noqa: BLE001
                    print(f"[{d.name}] rc failed: {e}")

            frames.append({"d": def_pos.round(3).tolist(), "a": att_pos.round(3).tolist(),
                           "alive": att_alive.tolist(), "dalive": def_alive.tolist()})
            if now - last_print > 1.0:
                last_print = now
                ds = "  ".join(f"{d.name}{'' if d.alive else '(down)'} {np.round(d.p, 2).tolist()} fix {now - d.last_fix:.1f}s"
                               for d in defenders)
                print(f"[gs] {mode:<6} attackers {int(att_alive.sum())}  {ds}")
            if not any(d.flying for d in defenders):
                print("[gs] no defender flying")
                break
    except KeyboardInterrupt:
        print("[gs] Ctrl+C")
    finally:
        land_all(defenders)
        if cam is not None:
            cam.close()
        for d in defenders:
            d.alive = False
            try:
                d.tello.end()
            except Exception:  # noqa: BLE001
                pass
        if frames:
            os.makedirs(os.path.dirname(rec_path), exist_ok=True)
            step_dt = (time.time() - t_start) / max(len(frames), 1)
            with open(rec_path, "w") as f:
                json.dump({"dt": step_dt, "room": [room[a] for a in "xyz"], "goal_zone": [arena["goal"]["zone"][a] for a in "xyz"],
                           "goal_tag_center": next(t["center"] for t in arena["tags"] if t["id"] == arena["goal"]["tag_id"]),
                           "goal_tag_size": next(t["size_m"] for t in arena["tags"] if t["id"] == arena["goal"]["tag_id"]),
                           "tag_radius": 0.2, "contact_mode": "bump", "task": "real flight" if not args.sim else "ground station sim",
                           "controller": name, "envs": [frames]}, f)
            print(f"[gs] recorded {len(frames)} frames to {rec_path}")
        if world is not None:
            print(f"[sim] breaches {world.breaches}, bumps {world.bumps}")


if __name__ == "__main__":
    main()
