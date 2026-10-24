# Copyright (c) 2025, Action Space hackathon team.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Defender policy for the fleet-telemetry setup: every drone's pose comes in, rc commands go out.

No vision or localization here: the fleet telemetry (drone_telemetry wire format v1, UDP JSON, one
datagram per tick with EVERY drone, both teams) already says where everyone is. This file

1. parses a telemetry datagram into one room frame (config/arena.json, or ARENA_JSON=arena_gate):
   ``pose`` is the camera position in the frame of the AprilTag it saw (x right, y down, z into the
   wall, yaw 0 = facing the tag), and arena.json says where each tag is, so pose -> room position;
   height comes from ``tof`` / ``h`` (pose y can flip on small tags); velocities are differenced
   from positions (the Tello's own vgx/vgy are in its own axes);
2. runs the tuned BlockController (tello/controllers.py, optionally relay / taped mesh) on the
   whole snapshot. Every defender gets the same snapshot and the controller is deterministic, so
   per-drone copies of this policy agree on who marks whom without talking to each other;
3. turns each defender's room-frame velocity into Tello rc (left_right, forward_back, up_down, yaw)
   with the team's motion fit (~0.73 m/s at full stick).

Use it three ways:

* in the team's runner (swarm_ops): copy this file and the tello/ folder's controllers.py next to
  it, register ``"defend": DefensePolicy`` in swarm_ops/policies/__init__.py, then
  ``python swarm_policies.py --run defend:5,6,7 --fly``. ``step(d, t)`` returns ``(a, b, c, d)``.
* as a library: ``policy.update(datagram_dict)`` then ``policy.rc_for(name)``.
* standalone, dry run: ``python tello/defense_policy.py --port 9100`` prints the rc it would send.
  ``python tello/defense_policy.py --selftest`` flies a simulated 3v2 game through the same
  telemetry format end to end (no hardware, no Isaac).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import socket
import sys
import threading
import time

import numpy as np

try:  # works both as ``tello.defense_policy`` and as a plain script / copied file
    from .controllers import BlockController, tuned_settings_path
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from controllers import BlockController, tuned_settings_path  # noqa: E402


def load_arena(path: str | None = None) -> dict:
    """Arena map JSON (no OpenCV needed): path, else $ARENA_JSON name in config/, else config/arena.json."""
    if path is None:
        name = os.environ.get("ARENA_JSON", "arena")
        path = name if os.path.isabs(name) else os.path.join(REPO, "config", name if name.endswith(".json") else name + ".json")
    with open(path) as f:
        return json.load(f)

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
TUNED_CFG = tuned_settings_path(REPO)  # tuned at real Tello speed for this arena map
RC_PER_MPS = np.array([125.0, 137.0, 167.0])  # team motion fit: right, forward, up (rc per m/s)
STALE_S = 1.0  # a pose older than this is not used for an attacker (treated as not seen)
NUM_SLOTS, NUM_ATTACKERS = 3, 2


def wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


class TagFrames:
    """Convert telemetry poses (camera in a tag's frame) into the room frame of arena.json."""

    def __init__(self, arena: dict):
        self.tags = {}
        for t in arena["tags"]:
            n = np.asarray(t["normal"], float)
            u = np.asarray(t["up"], float)
            n, u = n / np.linalg.norm(n), u / np.linalg.norm(u)
            r = np.cross(u, n)
            # tag frame axes in the room: x = right, y = down, z = into the wall
            self.tags[int(t["id"])] = (np.asarray(t["center"], float), r, -u, -n)

    def to_room(self, pose: dict):
        """(position (3,), yaw [rad]) in the room, or None for an unknown tag."""
        frame = self.tags.get(int(pose["tag"]))
        if frame is None:
            return None
        c, ex, ey, ez = frame
        x, y, z = pose["xyz"]
        pos = c + x * ex + y * ey + z * ez
        yaw = math.radians(pose.get("yaw", 0.0))
        heading = math.cos(yaw) * ez + math.sin(yaw) * ex  # 0 = looking at the tag, + = toward its right
        return pos, math.atan2(heading[1], heading[0])


class MapFrames:
    """Pose conversion through the TEAM'S MEASURED tag map (config/team_map.yaml, from cal_and_map).

    map.yaml: every tag's pose (R, t) in tag 10's frame, map convention x right, y up, z out of the tag
    face. Telemetry poses use the camera convention (x right, y down, z into the wall), so a pose
    (x, y, z) in tag k is (x, -y, -z) in k's map frame, then R_k p + t_k in tag 10's frame. That frame
    becomes the room frame of arena.json: x_room = z (out of the goal wall), y_room = x, z_room = y +
    height of tag 10 above the floor (from the floor tags, whose faces point up).
    """

    def __init__(self, path: str, tag10_height: float | None = None):
        import yaml

        with open(path) as f:
            m = yaml.safe_load(f)
        self.tags = {int(k): (np.asarray(v["R"], float), np.asarray(v["t"], float)) for k, v in m["tags"].items()}
        if tag10_height is None:
            floor = [t[1] for R, t in self.tags.values() if R[1, 2] > 0.9]  # faces pointing up
            tag10_height = -float(np.median(floor)) if floor else 1.22
        self.h10 = tag10_height

    def _room(self, p_map: np.ndarray) -> np.ndarray:
        return np.array([p_map[2], p_map[0], p_map[1] + self.h10])

    def to_room(self, pose: dict):
        hit = self.tags.get(int(pose["tag"]))
        if hit is None:
            return None
        R, t = hit
        x, y, z = pose["xyz"]
        pos = self._room(R @ np.array([x, -y, -z]) + t)
        yaw = math.radians(pose.get("yaw", 0.0))
        heading_tag = math.cos(yaw) * np.array([0.0, 0.0, -1.0]) + math.sin(yaw) * np.array([1.0, 0.0, 0.0])
        h = R @ heading_tag
        h_room = np.array([h[2], h[0], h[1]])
        return pos, math.atan2(h_room[1], h_room[0])


class DefensePolicy:
    """Policy for the defenders; compatible with swarm_ops' Policy (setup / step / describe)."""

    name = "defend"
    bus = None

    def __init__(self, port: int | None = 9100, relay: bool = False, nets=None, use_tuned: bool = True,
                 rc_limit: int = 100, max_speed=None, arena: dict | None = None, use_truth: bool = False,
                 tag_map: str | None = None, **ctrl_options):
        self.use_truth = use_truth  # drone_gui --sim also sends the true pose; use it to test the policy alone
        self.arena = arena or load_arena()
        # positions through the team's measured map when given (config/team_map.yaml), else arena.json's tags
        self.frames = MapFrames(tag_map) if tag_map else TagFrames(self.arena)
        settings = {}
        if use_tuned and os.path.exists(TUNED_CFG):
            with open(TUNED_CFG) as f:
                settings = json.load(f)["settings"]
        top = rc_limit / RC_PER_MPS  # fastest each axis can go [m/s]
        vmax_xy, vmax_z = (float(top[:2].min()), float(top[2])) if max_speed is None else max_speed
        self.ctrl = BlockController(self.arena, NUM_SLOTS, NUM_ATTACKERS, max_speed_xy=vmax_xy, max_speed_z=vmax_z,
                                    **{**settings, "relay": relay, "net_carriers": nets, **ctrl_options})
        self.rc_limit = rc_limit
        self.starts = np.asarray(self.arena["defender_starts"], float)
        self.lock = threading.Lock()
        self.track: dict[str, dict] = {}  # per drone name: last position/time/velocity
        self.cmd: dict[str, tuple] = {}  # per defender name: rc
        self.jobs: dict[str, str] = {}
        self.yaw: dict[str, float] = {}
        self.seq = None
        self.last_msg_t = 0.0
        if port:
            threading.Thread(target=self._listen, args=(port,), daemon=True).start()

    # -- telemetry ---------------------------------------------------------------------------------
    def _listen(self, port: int) -> None:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind(("0.0.0.0", port))
        while True:
            data, _ = sock.recvfrom(65535)
            try:
                self.update(json.loads(data))
            except (ValueError, KeyError) as e:
                print(f"[defend] bad telemetry: {e}")

    def _state(self, d: dict, now: float):
        """Room position, velocity, yaw and freshness of one telemetry entry (None if never located)."""
        pose = (d.get("truth") if self.use_truth else None) or d.get("pose") or None
        name = d["name"]
        tr = self.track.get(name)
        if pose is None:
            return None if tr is None else (tr["p"], tr["v"], tr["yaw"], False)
        conv = self.frames.to_room(pose)
        if conv is None:
            return None if tr is None else (tr["p"], tr["v"], tr["yaw"], False)
        p, yaw = conv
        tof = d.get("tof")
        if tof is not None and 0.1 < tof < 4.0:
            p[2] = tof  # height above the floor: steadier than pose y
        t_meas = now - float(pose.get("age", 0.0))
        fresh = pose.get("age", 0.0) < STALE_S
        if tr is None:
            tr = {"p": p, "v": np.zeros(3), "t": t_meas, "yaw": yaw}
        elif t_meas > tr["t"] + 1e-3:
            v = (p - tr["p"]) / (t_meas - tr["t"])
            tr = {"p": p, "v": 0.6 * tr["v"] + 0.4 * np.clip(v, -3, 3), "t": t_meas, "yaw": yaw}
        self.track[name] = tr
        return tr["p"], tr["v"], tr["yaw"], fresh

    def update(self, msg: dict) -> None:
        """Take one telemetry datagram (dict) and recompute every defender's rc."""
        if msg.get("v") != 1:
            return
        now = float(msg.get("t", time.time()))
        drones = msg["drones"]
        defenders = sorted((d for d in drones if d.get("team") == "defense"), key=lambda d: d["name"])[:NUM_SLOTS]
        attackers = sorted((d for d in drones if d.get("team") == "attack"), key=lambda d: d["name"])[:NUM_ATTACKERS]
        def_pos, def_vel = self.starts.copy(), np.zeros((NUM_SLOTS, 3))
        def_alive = np.zeros(NUM_SLOTS, bool)
        att_pos, att_vel = np.zeros((NUM_ATTACKERS, 3)), np.zeros((NUM_ATTACKERS, 3))
        att_alive = np.zeros(NUM_ATTACKERS, bool)
        names = []
        for i, d in enumerate(defenders):
            names.append(d["name"])
            s = self._state(d, now)
            if s is None:
                continue
            def_pos[i], def_vel[i], self.yaw[d["name"]], _ = s
            def_alive[i] = bool(d.get("link", True)) and bool(d.get("fly", True))
        for j, d in enumerate(attackers):
            s = self._state(d, now)
            if s is None:
                continue
            att_pos[j], att_vel[j], _, fresh = s
            att_alive[j] = fresh and bool(d.get("fly", True))
        vel = self.ctrl(def_pos[None], def_vel[None], att_pos[None], att_vel[None], att_alive[None], def_alive[None])[0]
        jobs = self.ctrl.describe(0)
        with self.lock:
            for i, name in enumerate(names):
                self.cmd[name] = self._rc(vel[i], self.yaw.get(name, math.pi)) if def_alive[i] else (0, 0, 0, 0)
                self.jobs[name] = jobs[i]
            self.seq = msg.get("seq")
            self.last_msg_t = time.time()

    def _rc(self, v: np.ndarray, yaw: float) -> tuple[int, int, int, int]:
        fwd = np.array([math.cos(yaw), math.sin(yaw)])
        right = np.array([math.sin(yaw), -math.cos(yaw)])
        rc = np.array([v[:2] @ right, v[:2] @ fwd, v[2]]) * RC_PER_MPS
        lr, fb, ud = (int(x) for x in np.clip(np.round(rc), -self.rc_limit, self.rc_limit))
        return lr, fb, ud, 0

    def fresh(self, name: str) -> bool:
        """Telemetry arrived within the last second and included this defender."""
        with self.lock:
            return time.time() - self.last_msg_t <= 1.0 and name in self.cmd

    def rc_for(self, name: str):
        """Latest rc for a defender by telemetry name (hover if telemetry stopped)."""
        with self.lock:
            if time.time() - self.last_msg_t > 1.0:
                return (0, 0, 0, 0)  # no fresh telemetry: hold position
            return self.cmd.get(name, (0, 0, 0, 0))

    # -- swarm_ops Policy interface ----------------------------------------------------------------
    @staticmethod
    def telemetry_name(d) -> str:
        """swarm_ops Drone (num) -> telemetry name (TELLO-<num>)."""
        return getattr(d, "telemetry_name", None) or f"TELLO-{d.num}"

    def setup(self, drones):
        self.drones = list(drones)

    def step(self, d, t):
        return self.rc_for(self.telemetry_name(d))

    def describe(self, d):
        return f"defend: {self.jobs.get(self.telemetry_name(d), '')}"


# ----------------------------------------------------------------------------------------- self test
class _FakeFleet:
    """3 defenders + 2 attackers flying in the room, published in the telemetry v1 format (pose in tag 10's frame)."""

    def __init__(self, arena: dict, seed: int = 0):
        self.rng = np.random.default_rng(seed)
        self.frames = TagFrames(arena)
        gid = int(arena["goal"]["tag_id"])
        self.tag = gid
        zone = arena["goal"]["zone"]
        self.zlo = np.array([zone[a][0] for a in "xyz"])
        self.zhi = np.array([zone[a][1] for a in "xyz"])
        self.goal = (self.zlo + self.zhi) / 2
        self.spawn = arena["attacker_spawn"]
        starts = np.asarray(arena["defender_starts"], float)
        self.d = [{"name": f"TELLO-{5 + i}", "p": starts[i] + [0, 0, 1.0], "v": np.zeros(3), "yaw": math.pi,
                   "cmd": np.zeros(3)} for i in range(3)]
        self.a = [{"name": f"TELLO-{1 + j}", "p": None, "v": np.zeros(3), "stun": 0.0, "off": 0.0} for j in range(2)]
        for a in self.a:
            self._respawn(a)
        self.breaches = self.bumps = 0
        self.seq = 0

    def _respawn(self, a):
        s = self.spawn
        a["p"] = np.array([self.rng.uniform(*s["x"]), self.rng.uniform(*s["y"]), self.rng.uniform(*s["z"])])
        a["v"][:] = 0
        a["off"] = self.rng.uniform(-1.5, 1.5)

    def apply_rc(self, name, rc):
        d = next(x for x in self.d if x["name"] == name)
        lr, fb, ud, _ = rc
        fwd = np.array([math.cos(d["yaw"]), math.sin(d["yaw"])])
        right = np.array([math.sin(d["yaw"]), -math.cos(d["yaw"])])
        d["cmd"][:2] = lr / RC_PER_MPS[0] * right + fb / RC_PER_MPS[1] * fwd
        d["cmd"][2] = ud / RC_PER_MPS[2]

    def step(self, dt):
        for d in self.d:
            d["v"] += (d["cmd"] - d["v"]) * min(dt / 0.3, 1.0)
            d["p"] = d["p"] + d["v"] * dt
        for a in self.a:
            if a["stun"] > 0:
                a["stun"] -= dt
            else:
                dist = np.linalg.norm(self.goal - a["p"])
                aim = self.goal + np.array([0.0, a["off"] * min(dist / 4, 1.0), 0.0])
                want = (aim - a["p"]) / max(np.linalg.norm(aim - a["p"]), 1e-6) * 0.6  # Tellos are slow
                a["v"] += (want - a["v"]) * min(dt / 0.3, 1.0)
            for d in self.d:
                gap = a["p"] - d["p"]
                if np.linalg.norm(gap) < 0.22:
                    a["v"] = gap / max(np.linalg.norm(gap), 1e-6) * 0.8
                    a["stun"] = 0.7
                    self.bumps += 1
            a["p"] = a["p"] + a["v"] * dt
            if np.all((a["p"] >= self.zlo) & (a["p"] <= self.zhi)):
                self.breaches += 1
                self._respawn(a)

    def _pose(self, p, yaw):
        """Room position/yaw -> camera pose in the goal tag's frame (the inverse of TagFrames.to_room)."""
        c, ex, ey, ez = self.frames.tags[self.tag]
        rel = p - c
        heading = np.array([math.cos(yaw), math.sin(yaw), 0.0])
        tag_yaw = math.degrees(math.atan2(heading @ ex, heading @ ez))
        return {"tag": self.tag, "xyz": [round(float(rel @ ex), 3), round(float(rel @ ey), 3), round(float(rel @ ez), 3)],
                "yaw": round(tag_yaw, 1), "age": 0.03}

    def datagram(self, t):
        self.seq += 1
        entries = []
        for team, group in (("defense", self.d), ("attack", self.a)):
            for x in group:
                p = x["p"] + self.rng.normal(0, 0.03, 3)
                entries.append({"name": x["name"], "team": team, "link": True, "fly": True, "bat": 80,
                                "h": round(float(p[2]), 2), "tof": round(float(p[2]), 2),
                                "vel": [0.0, 0.0, 0.0], "pose": self._pose(p, x.get("yaw", 0.0))})
        return {"v": 1, "src": "sim", "seq": self.seq, "t": t, "hz": 20.0, "drones": entries}


def selftest(seconds: float, relay: bool, kamikaze: bool = False) -> None:
    arena = load_arena()
    fleet = _FakeFleet(arena)
    pol = DefensePolicy(port=None, relay=relay, arena=arena, kamikaze=kamikaze)
    dt, t = 0.05, time.time()
    last = -1.0
    for k in range(int(seconds / dt)):
        now = t + k * dt
        msg = fleet.datagram(now)
        pol.update(msg)
        pol.last_msg_t = time.time()
        for d in fleet.d:
            fleet.apply_rc(d["name"], pol.rc_for(d["name"]))
        fleet.step(dt)
        if k * dt - last >= 5.0:
            last = k * dt
            print(f"[selftest] t {k * dt:5.1f}s  " + "  ".join(f"{n}: {pol.jobs.get(n, '')} rc {pol.cmd.get(n)}"
                                                         for n in sorted(pol.cmd)))
    print(f"[selftest] {seconds:.0f} s of play: breaches {fleet.breaches}, bumps {fleet.bumps} "
          f"(attackers respawn after every breach)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=9100, help="telemetry UDP port to listen on")
    ap.add_argument("--relay", action="store_true", help="relay blocking strategy")
    ap.add_argument("--nets", type=str, default=None, help="mesh carriers by defender order, e.g. 1,1,1")
    ap.add_argument("--map", default=None, help="team tag map (e.g. config/team_map.yaml) for pose -> room")
    ap.add_argument("--kamikaze", action="store_true", help="kamikaze keeper strategy (rams attackers near the goal)")
    ap.add_argument("--use_truth", action="store_true", help="with drone_gui --sim: use the true pose it sends")
    ap.add_argument("--selftest", type=float, nargs="?", const=60.0, default=None, help="simulate N seconds, no hardware")
    args = ap.parse_args()
    if args.selftest:
        selftest(args.selftest, args.relay, args.kamikaze)
        return
    nets = None if args.nets is None else [bool(int(x)) for x in args.nets.split(",")]
    pol = DefensePolicy(port=args.port, relay=args.relay, nets=nets, use_truth=args.use_truth, kamikaze=args.kamikaze,
                        tag_map=args.map)
    print(f"[defend] listening for telemetry on :{args.port} (dry run: printing rc only)")
    while True:
        time.sleep(0.5)
        with pol.lock:
            if pol.cmd:
                print(f"seq {pol.seq}  " + "  ".join(f"{n}: {pol.jobs.get(n, '')} rc {pol.cmd[n]}" for n in sorted(pol.cmd)), flush=True)


if __name__ == "__main__":
    main()
