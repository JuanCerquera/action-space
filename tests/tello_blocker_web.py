# Copyright (c) 2026, Action Space team.
# SPDX-License-Identifier: Apache-2.0
"""Blocker test panel: 3 simulated defenders vs 2 draggable attackers, with the real Tello on any slot.

All three defenders are simulated. Attach the real Tello to one slot (D1-D3)
and that defender is flown by the drone instead; switch slots at any time,
even in flight (the new slot takes the drone's position). Each defender is
paired with one attacker, flies into its path to the table and turns to face
it; the spare defender keeps goal near the table, facing the attacker
closest to it. Runs without a drone; press Connect when on the Tello's Wi-Fi.

    python tests/tello_blocker_web.py [--connect]

and open http://127.0.0.1:8765 (opened automatically).

World frame: metres, table at the origin, attackers come from +x; yaw is
counterclockwise from +x. Edit TABLE / FENCE / spawns to match the AprilTag
frame. The real drone's position comes from POST /pose {"x", "y", "yaw"?}
(world frame, radians) while it is fresh, otherwise it is dead-reckoned from
the commands sent (uncalibrated rc_per_mps). Its heading is the Tello IMU yaw,
anchored at takeoff and at each /pose yaw. Before takeoff, place the drone on
its slot's spot facing +x.

Record saves camera frames and telemetry to recordings/<time>/ (see Recorder).

    t take off   l land   b blocker on/off   x EMERGENCY (cuts motors)
"""

import argparse
import json
import math
import sys
import threading
import time
import webbrowser
from dataclasses import dataclass, field, replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
from djitellopy import Tello

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from defense.attack_script import PLANS as SCRIPT, scripted_position  # noqa: E402
from defense.blocker import BlockerConfig, min_pairwise_distance, rc_to_velocity, to_rc  # noqa: E402
from defense.team import find_hits, plan_team  # noqa: E402
from tello_web import Drone, Recorder, VideoReader, jpeg  # noqa: E402

Z = 1.0  # planar test: all points share one height; vertical is not commanded
TABLE = np.array((0.0, 0.0, Z))
FENCE = ((0.4, 2.5), (-1.2, 1.2))  # x and y limits for block targets, m
DEFENDER_SPAWNS = ((1.0, -0.7), (1.0, 0.0), (1.0, 0.7))  # all facing +x
ATTACKER_STARTS = ((3.5, -0.8), (3.5, 0.8))  # 2 attackers: the spare defender keeps goal
LOOP_S = 0.1  # 10 Hz control loop
ATTACKER_STALE_S = 0.3  # attacker velocity drops to zero when not dragged
POSE_STALE_S = 0.5  # fall back to dead reckoning when /pose is older than this
SIM_TAU_S = 0.3  # simulated drones reach the commanded velocity with this time constant
IMU_YAW_CLOCKWISE = True  # Tello IMU yaw grows turning clockwise: CHECK by turning the landed drone by hand
RECORD_DIR = Path(__file__).resolve().parents[1] / "recordings"

CFG = BlockerConfig(max_speed=0.6, gain=1.5, defender_speed=0.6)  # max speed is set live on the page
REAL_VERTICAL = False  # the real Tello holds its height (no height estimate wired in yet); sims climb and descend


def wrap(angle: float) -> float:
    return (angle + math.pi) % (2 * math.pi) - math.pi


@dataclass
class Defender:
    name: str
    pos: np.ndarray
    yaw: float = 0.0
    vel: np.ndarray = field(default_factory=lambda: np.zeros(3))  # simulated velocity
    target: np.ndarray = None
    rc: tuple = (0, 0, 0, 0)
    alive: bool = True  # False once it has bumped an attacker
    role: str = "block"  # block / keeper / intercept / out


class Game:
    """Simulated defenders and attackers, the real Tello on one slot, and the control loop."""

    def __init__(self, cfg: BlockerConfig):
        self.cfg = cfg
        self.lock = threading.Lock()
        self.enabled = False
        self.face = True
        self.avoid = True  # teammate avoidance in the velocity command (targets are always separated)
        self.attack_mode = False  # the spare defender rams the attacker nearest the table instead of keeping goal
        self.min_dist = float("inf")  # closest pair of defenders now, m
        self.min_dist_seen = float("inf")  # closest pair since the last reset while the blocker was on
        self.message = "simulation only: press Connect to add the real drone"
        self.tello = None
        self.drone = None  # tello_web.Drone once connected
        self.connecting = False
        self.slot = None  # defender index the real drone flies
        self.real_pos = None  # real drone world position estimate
        self.yaw_anchor = None  # (imu yaw deg, world yaw rad) pair the heading is measured from
        self.pose_time = 0.0  # last /pose
        self.recorder = None
        self.defenders = []
        self.reset()

    def reset(self):
        """Everyone back in play: simulated defenders to their spawns (the real slot keeps its pose), attackers to their starts."""
        with self.lock:
            old = self.defenders
            self.defenders = [Defender(f"D{i + 1}", np.array((x, y, Z))) for i, (x, y) in enumerate(DEFENDER_SPAWNS)]
            if self.slot is not None and self.real_pos is not None and old:
                keep = old[self.slot]
                keep.alive, keep.role = True, "block"
                self.defenders[self.slot] = keep
            self.min_dist_seen = float("inf")
        self.reset_attackers()

    def reset_attackers(self):
        with self.lock:
            self.attackers = [np.array((x, y, plan["z0"])) for (x, y), plan in zip(ATTACKER_STARTS, SCRIPT)]
            self.attacker_vel = [np.zeros(3) for _ in self.attackers]
            self.attacker_time = [0.0] * len(self.attackers)
            self.attacker_status = ["active"] * len(self.attackers)  # active / down (bumped) / scored
            self.script_t = None  # scripted attack clock, None when not running
            self.scripted = [False] * len(self.attackers)
            self.assignment = None
            for d in self.defenders:  # knocked-out defenders return when the attackers do
                d.alive = True

    # --- real drone -------------------------------------------------------

    def connect(self):
        if self.drone is not None or self.connecting:
            return
        self.connecting = True
        self.message = "connecting to the Tello..."

        def worker():
            try:
                self.tello = self.tello or Tello()
                self.tello.connect()
                self.tello.streamon()
                drone = Drone(self.tello, speed=30, reader=VideoReader())
                drone.message = ""
                threading.Thread(target=drone.watchdog, daemon=True).start()
                self.drone = drone
                self.message = f"connected, battery {self.tello.get_battery()}%: attach it to a slot"
            except Exception as exc:
                self.message = f"connect failed ({exc}): on the Tello's Wi-Fi?"
            self.connecting = False

        threading.Thread(target=worker, daemon=True).start()

    def imu_yaw(self):
        try:
            return float(self.drone.tello.get_yaw())
        except Exception:
            return None

    def _world_yaw(self, imu):
        imu_ref, world_ref = self.yaw_anchor
        turn = math.radians(imu - imu_ref)
        return wrap(world_ref - turn if IMU_YAW_CLOCKWISE else world_ref + turn)

    def _hover(self):
        drone = self.drone
        if drone is None:
            return
        with drone.lock:
            if drone.flying and not drone.busy:
                drone.tello.send_rc_control(0, 0, 0, 0)
            drone.moving = False

    def attach(self, slot):
        with self.lock:
            if slot is not None and self.real_pos is not None and self.drone is not None and self.drone.flying:
                d = self.defenders[slot]  # the new slot takes over where the drone is
                d.pos, d.vel = self.real_pos.copy(), np.zeros(3)
            self.slot = slot
        if slot is None:
            self._hover()

    def takeoff(self):
        if self.drone is None:
            return
        if self.slot is None:
            self.message = "attach the real drone to a slot first (and place it on that spot, facing +x)"
            return
        imu = self.imu_yaw()
        with self.lock:
            d = self.defenders[self.slot]
            self.real_pos = d.pos.copy()
            self.yaw_anchor = (imu if imu is not None else 0.0, d.yaw)
            self.message = ""
        self.drone.command("takeoff")

    def set_pose(self, x, y, yaw=None):
        imu = self.imu_yaw()
        with self.lock:
            self.real_pos = np.array((float(x), float(y), Z))
            if yaw is not None and imu is not None:
                self.yaw_anchor = (imu, float(yaw))
            self.pose_time = time.monotonic()

    def set_recording(self, on: bool):
        if on and self.recorder is None:
            if self.drone is None or self.drone.reader is None:
                self.message = "connect the drone to record its camera"
                return
            self.recorder = Recorder(RECORD_DIR / time.strftime("%Y%m%d-%H%M%S"), self._record_state)
            self.drone.reader.recorder = self.recorder
            self.message = f"recording to {self.recorder.directory}"
        elif not on and self.recorder is not None:
            if self.drone is not None and self.drone.reader is not None:
                self.drone.reader.recorder = None
            self.recorder.close()
            self.message = f"saved {self.recorder.frames} frames to {self.recorder.directory}"
            self.recorder = None

    def _record_state(self) -> dict:
        tello_state = dict(self.drone.tello.get_current_state()) if self.drone else {}
        with self.lock:
            return {
                "tello": tello_state,  # raw SDK state: yaw/pitch/roll deg, vgx.. dm/s, h/tof cm, agx.. etc.
                "pose_est": None if self.real_pos is None else self.real_pos.tolist(),
                "yaw_est": self.defenders[self.slot].yaw if self.slot is not None else None,
                "pose_from_tags": time.monotonic() - self.pose_time < POSE_STALE_S,
                "flying": bool(self.drone and self.drone.flying),
            }

    # --- simulation and control -------------------------------------------

    def set_attacker(self, i: int, x: float, y: float):
        now = time.monotonic()
        with self.lock:
            if self.attacker_status[i] != "active":
                return
            new = np.array((float(x), float(y), self.attackers[i][2]))  # dragging keeps its height
            self.scripted[i] = False  # dragging takes it off the script
            dt = now - self.attacker_time[i]
            if dt < ATTACKER_STALE_S:
                self.attacker_vel[i] = 0.7 * self.attacker_vel[i] + 0.3 * (new - self.attackers[i]) / max(dt, 1e-3)
            self.attackers[i] = new
            self.attacker_time[i] = now

    def set_enabled(self, on: bool):
        with self.lock:
            self.enabled = on
        if not on:
            self._hover()

    def set_max_speed(self, speed: float):
        speed = float(np.clip(speed, 0.1, 1.0))
        with self.lock:
            self.cfg = replace(self.cfg, max_speed=speed, defender_speed=speed)

    def loop(self):
        while True:
            time.sleep(LOOP_S)
            try:
                self.step(LOOP_S)
            except Exception as exc:  # keep the loop alive; show what went wrong
                self.message = f"control loop error: {exc}"
                print(f"[ERROR] {exc}")

    def start_script(self, on: bool):
        """Scripted attack: attackers back to their starts, then weave toward the table (see SCRIPT)."""
        self.reset_attackers()
        with self.lock:
            self.script_t = 0.0 if on else None
            self.scripted = [on] * len(self.attackers)

    def _advance_script(self, dt: float, now: float):
        self.script_t += dt
        for j, plan in enumerate(SCRIPT):
            if not self.scripted[j] or self.attacker_status[j] != "active":
                continue
            pos, arrived = scripted_position(ATTACKER_STARTS[j], TABLE, self.script_t, plan)
            self.attacker_vel[j] = (pos - self.attackers[j]) / dt
            self.attackers[j] = pos
            self.attacker_time[j] = now
            if arrived:
                self.attacker_status[j] = "scored"
                self.message = f"A{j + 1} reached the table"

    def _check_hits(self, cfg: BlockerConfig) -> bool:
        """Defender-attacker contact knocks both out. Returns True if the real drone was knocked out."""
        real_out = False
        hits = find_hits([d.pos for d in self.defenders], [d.alive for d in self.defenders], self.attackers,
                         [s == "active" for s in self.attacker_status], cfg)
        for i, j in hits:
            d = self.defenders[i]
            d.alive, d.role, d.rc, d.target = False, "out", (0, 0, 0, 0), None
            self.attacker_status[j] = "down"
            real_out |= i == self.slot and self.drone is not None
            left = sum(s == "active" for s in self.attacker_status)
            self.message = f"{d.name} bumped A{j + 1}: both out" + ("" if left else " - all attackers down")
            if real_out:
                self.message += f" ({d.name} is the real drone: it hovers, land it)"
        return real_out

    def step(self, dt: float):
        drone = self.drone
        imu = self.imu_yaw() if drone is not None else None
        now = time.monotonic()
        real_rc = None
        with self.lock:
            cfg = self.cfg
            if self.script_t is not None:
                self._advance_script(dt, now)
            for j in range(len(self.attackers)):
                if now - self.attacker_time[j] > ATTACKER_STALE_S:
                    self.attacker_vel[j] = np.zeros(3)
            real = self.slot if drone is not None and self.real_pos is not None else None
            real_flying = real is not None and drone.flying and not drone.busy
            if real is not None:
                d = self.defenders[real]
                d.pos = self.real_pos.copy()
                if imu is not None and self.yaw_anchor is not None:
                    d.yaw = self._world_yaw(imu)

            # Roles over the drones still in play: block an attacker, keep goal, or (attack mode) intercept.
            plans = plan_team([d.pos for d in self.defenders], [d.yaw for d in self.defenders],
                              [d.alive for d in self.defenders], self.attackers, self.attacker_vel,
                              [s == "active" for s in self.attacker_status], TABLE, cfg, fence=FENCE,
                              previous=self.assignment, attack_mode=self.attack_mode, avoid=self.avoid,
                              face=self.face)
            self.assignment = [p.attacker for p in plans]
            self.min_dist = min_pairwise_distance([d.pos for d in self.defenders if d.alive])
            if self.enabled:
                self.min_dist_seen = min(self.min_dist_seen, self.min_dist)

            for i, (d, plan) in enumerate(zip(self.defenders, plans)):
                if not d.alive:
                    continue
                d.role, d.target = plan.role, plan.target
                if self.enabled and (i != real or real_flying):
                    d.rc = to_rc(plan.velocity, d.yaw, cfg, plan.yaw_rate)
                else:
                    d.rc = (0, 0, 0, 0)
                if i == real:
                    if not REAL_VERTICAL:
                        d.rc = (d.rc[0], d.rc[1], 0, d.rc[3])
                    real_rc = d.rc if self.enabled and real_flying else None
                else:
                    self._simulate(d, cfg, dt)
            real_out = self.enabled and self._check_hits(cfg)
        if real_out:
            self._hover()
            return
        if real_rc is None:
            return
        with drone.lock:
            if not drone.flying or drone.busy:
                return
            drone.tello.send_rc_control(*real_rc)
            drone.moving = any(real_rc)
            drone.last_rc = time.monotonic()
        with self.lock:
            if time.monotonic() - self.pose_time > POSE_STALE_S:  # no fresh tag pose: dead-reckon
                vel, _ = rc_to_velocity(real_rc, self.defenders[real].yaw, cfg)
                vel[2] = 0.0
                self.real_pos = self.real_pos + vel * dt

    @staticmethod
    def _simulate(d: Defender, cfg: BlockerConfig, dt: float):
        vel, turn = rc_to_velocity(d.rc, d.yaw, cfg)
        d.vel = d.vel + (vel - d.vel) * min(1.0, dt / SIM_TAU_S)
        d.pos = d.pos + d.vel * dt
        d.yaw = wrap(d.yaw + turn * dt)

    def state(self) -> dict:
        drone = self.drone
        real = {"connected": drone is not None, "connecting": self.connecting}
        if drone is not None:
            try:
                real.update(drone.status())
                tello = drone.tello
                real["measured_raw"] = [tello.get_speed_x(), tello.get_speed_y()]  # for calibrating rc_per_mps
            except Exception as exc:
                real["message"] = f"no telemetry ({exc})"
            real["imu_yaw"] = self.imu_yaw()
            real["video_age"] = None if drone.reader is None else min(drone.reader.age(), 1e6)
        with self.lock:
            return {
                "enabled": self.enabled,
                "face": self.face,
                "avoid": self.avoid,
                "min_dist": None if math.isinf(self.min_dist) else self.min_dist,
                "min_dist_seen": None if math.isinf(self.min_dist_seen) else self.min_dist_seen,
                "min_separation": self.cfg.min_separation,
                "max_speed": self.cfg.max_speed,
                "rc_per_mps": self.cfg.rc_per_mps,
                "slot": self.slot,
                "pose_from_tags": time.monotonic() - self.pose_time < POSE_STALE_S,
                "recording": None if self.recorder is None else {"dir": str(self.recorder.directory),
                                                                 "frames": self.recorder.frames},
                "message": self.message,
                "attack_mode": self.attack_mode,
                "script": self.script_t is not None,
                "attacker_status": self.attacker_status,
                "defenders": [{"name": d.name, "pos": d.pos.tolist(), "yaw": d.yaw, "rc": list(d.rc), "role": d.role,
                               "alive": d.alive, "target": None if d.target is None else d.target.tolist()}
                              for d in self.defenders],
                "assignment": self.assignment,
                "attackers": [a.tolist() for a in self.attackers],
                "table": TABLE.tolist(),
                "fence": FENCE,
                "real": real,
            }


PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Tello Blocker Test</title>
<style>
  body { margin: 0; background: #111; color: #eee; font: 15px system-ui, sans-serif; }
  main { max-width: 1100px; margin: 0 auto; padding: 16px; }
  .cols { display: flex; gap: 16px; flex-wrap: wrap; align-items: flex-start; }
  #map { background: #1b1f26; border-radius: 8px; touch-action: none; cursor: crosshair; max-width: 100%; }
  .cam { flex: 1; min-width: 280px; }
  #video { width: 100%; aspect-ratio: 4 / 3; background: #000; border-radius: 8px; object-fit: contain; display: block; }
  #videoinfo { color: #999; margin-top: 4px; font-variant-numeric: tabular-nums; }
  #videoinfo.bad { color: #f66; }
  .view3d { flex: 1 1 100%; }
  #view3d { width: 100%; background: #1b1f26; border-radius: 8px; touch-action: none; cursor: grab; display: block; }
  .hint { color: #999; margin-top: 4px; }
  #status { display: flex; gap: 16px; flex-wrap: wrap; margin: 12px 0; font-variant-numeric: tabular-nums; }
  .message { color: #fc6; }
  .row { display: flex; gap: 8px; flex-wrap: wrap; align-items: center; margin: 8px 0; }
  button, select { font: inherit; padding: 12px 16px; border: 0; border-radius: 6px; background: #2a2f38; color: #eee; cursor: pointer; }
  #takeoff { background: #1f6b3a; } #land { background: #6b5a1f; }
  #blocker.on, #face.on, #avoid.on, #script.on { background: #2f5fd0; }
  #attack.on { background: #b5471f; }
  .bad { color: #f66; }
  #record.on { background: #c22; }
  #emergency { background: #a11; margin-left: auto; font-weight: 700; }
  input[type=range] { width: 220px; }
  p.help { color: #999; }
</style></head>
<body><main>
  <div class="cols">
    <canvas id="map" width="500" height="500"></canvas>
    <div class="cam"><img id="video" alt="Tello camera feed"><div id="videoinfo">no drone</div></div>
    <div class="view3d"><canvas id="view3d" width="1000" height="460"></canvas>
      <div class="hint">3D view: drag to orbit, scroll to zoom, double-click to reset</div></div>
  </div>
  <div id="status"></div>
  <div class="row">
    <button id="connect">Connect</button>
    <label>Real drone on <select id="slot"><option value="">none</option>
      <option value="0">D1</option><option value="1">D2</option><option value="2">D3</option></select></label>
    <button id="takeoff">Take off <kbd>t</kbd></button>
    <button id="land">Land <kbd>l</kbd></button>
    <button id="emergency">EMERGENCY <kbd>x</kbd></button>
  </div>
  <div class="row">
    <button id="blocker">Blocker OFF <kbd>b</kbd></button>
    <button id="face">Face attacker</button>
    <button id="avoid" title="Teammates push each other apart when closer than the avoid radius">Avoid teammates</button>
    <button id="attack" title="The spare defender rams the attacker nearest the table; a bump knocks both out">Keeper: guard</button>
    <button id="script" title="Attackers restart and weave toward the table on their own">Run scripted attack</button>
    <label>Max speed <input id="speed" type="range" min="0.1" max="1" step="0.05"> <span id="speedval"></span></label>
  </div>
  <div class="row">
    <button id="record">Record</button>
    <button id="reset">Reset sim</button>
    <button id="restart_video" title="Restart the camera stream (safe in flight)">Restart video</button>
    <button id="reconnect" title="Redo the connection and video, e.g. after a battery swap">Reconnect</button>
  </div>
  <p class="help">Drag an attacker (red) on the map. Map up = world +x, toward the attackers. Triangles = defenders
    (nose = heading), gold = the real drone, rings = block targets, faint lines = who blocks whom, dashed box = fence.
    Before takeoff place the drone on its slot's spot, facing up the map.</p>
</main>
<script>
const S = 100, X_MAX = 4.5, Y_MAX = 2.5;  // px per metre; map top edge x, left edge y
const toPx = (x, y) => [(Y_MAX - y) * S, (X_MAX - x) * S];
const toWorld = (px, py) => [X_MAX - py / S, Y_MAX - px / S];
const canvas = document.getElementById("map"), ctx = canvas.getContext("2d");
const $ = id => document.getElementById(id);
let game = null, drag = null, lastSent = 0;

const post = (path, body) => fetch(path, { method: "POST", headers: { "Content-Type": "application/json" },
                                           body: JSON.stringify(body || {}) }).catch(() => {});
const command = name => post("/cmd/" + name);
const toggle = (path, key) => game && post(path, { on: !game[key] });

function ring(x, y, r, color) {
  const [px, py] = toPx(x, y);
  ctx.beginPath(); ctx.arc(px, py, r, 0, 2 * Math.PI); ctx.strokeStyle = color; ctx.lineWidth = 2; ctx.stroke();
}
function line(a, b, color, dash) {
  const [ax, ay] = toPx(a[0], a[1]), [bx, by] = toPx(b[0], b[1]);
  ctx.setLineDash(dash || []); ctx.strokeStyle = color; ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(ax, ay); ctx.lineTo(bx, by); ctx.stroke(); ctx.setLineDash([]);
}
function label(x, y, text, color) {
  const [px, py] = toPx(x, y); ctx.fillStyle = color; ctx.font = "12px system-ui"; ctx.fillText(text, px + 12, py - 10);
}
function drone(d, real) {
  const [px, py] = toPx(d.pos[0], d.pos[1]);
  const fx = -Math.sin(d.yaw), fy = -Math.cos(d.yaw);  // world heading on screen (up = +x, left = +y)
  ctx.beginPath();
  ctx.moveTo(px + fx * 14, py + fy * 14);
  ctx.lineTo(px - fx * 8 + fy * 8, py - fy * 8 - fx * 8);
  ctx.lineTo(px - fx * 8 - fy * 8, py - fy * 8 + fx * 8);
  ctx.closePath(); ctx.fillStyle = d.alive ? "#4c8dff" : "#555"; ctx.fill();
  if (real) { ctx.strokeStyle = "#f5c542"; ctx.lineWidth = 3; ctx.stroke(); }
}

function draw() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.strokeStyle = "#2a2f38"; ctx.lineWidth = 1;
  for (let m = -10; m <= 10; m++) {
    const [gx] = toPx(0, m), [, gy] = toPx(m, 0);
    ctx.beginPath(); ctx.moveTo(gx, 0); ctx.lineTo(gx, canvas.height); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(0, gy); ctx.lineTo(canvas.width, gy); ctx.stroke();
  }
  if (!game) return;
  const [[x0, x1], [y0, y1]] = game.fence, [fx, fy] = toPx(x1, y1), [fx2, fy2] = toPx(x0, y0);
  ctx.setLineDash([6, 4]); ctx.strokeStyle = "#888"; ctx.strokeRect(fx, fy, fx2 - fx, fy2 - fy); ctx.setLineDash([]);
  const [tx, ty] = toPx(game.table[0], game.table[1]);
  ctx.fillStyle = "#2c8a4a"; ctx.fillRect(tx - 25, ty - 15, 50, 30);
  game.attackers.forEach((a, i) => {
    const status = game.attacker_status[i], live = status === "active";
    if (live) line(a, game.table, "#a33", [3, 3]);
    const [ax, ay] = toPx(a[0], a[1]); ctx.beginPath(); ctx.arc(ax, ay, 10, 0, 2 * Math.PI);
    ctx.fillStyle = live ? "#e33" : "#555"; ctx.fill();
    label(a[0], a[1], `A${i + 1} ${live ? a[2].toFixed(1) + " m" : status.toUpperCase()}`, live ? "#e88" : "#888");
  });
  game.defenders.forEach((d, i) => {
    const real = game.slot === i && game.real.connected;
    const role = game.assignment ? game.assignment[i] : undefined;
    if (d.alive && d.target) {
      if (role !== undefined && role !== null) line(d.target, game.attackers[role], "#4c8dff55");
      line(d.pos, d.target, d.role === "intercept" ? "#ff7a45" : "#4c8dff", [2, 3]);
      ring(d.target[0], d.target[1], 12, real ? "#f5c542" : d.role === "intercept" ? "#ff7a45" : "#4c8dff");
    }
    drone(d, real);
    label(d.pos[0], d.pos[1], d.name + ROLE_TAG[d.role] + (real ? " (real)" : ""), !d.alive ? "#888" : real ? "#f5c542" : "#9bf");
  });
}

// --- 3D view: a small perspective renderer on a 2D canvas (no libraries: the Tello Wi-Fi has no internet) ---
const c3 = $("view3d"), g3 = c3.getContext("2d");
const VIEW0 = { az: Math.PI + 0.6, el: 0.45, r: 6.5, center: [1.8, 0, 0.5] };  // from behind the table
const view = { ...VIEW0 };
const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const unit = a => { const n = Math.hypot(...a); return a.map(v => v / n); };
let cam = null;

function setupCamera() {
  const c = view.center, ce = Math.cos(view.el);
  const eye = [c[0] + view.r * ce * Math.cos(view.az), c[1] + view.r * ce * Math.sin(view.az), c[2] + view.r * Math.sin(view.el)];
  const f = unit(sub(c, eye)), right = unit(cross(f, [0, 0, 1])), up = cross(right, f);
  cam = { eye, f, right, up, F: c3.height * 1.3 };
}
function P(p) {  // world point -> [screen x, screen y, depth], null if behind the camera
  const d = sub(p, cam.eye), z = dot(d, cam.f);
  if (z < 0.1) return null;
  return [c3.width / 2 + cam.F * dot(d, cam.right) / z, c3.height / 2 - cam.F * dot(d, cam.up) / z, z];
}
function seg3(a, b, color, width, dash) {
  const pa = P(a), pb = P(b);
  if (!pa || !pb) return;
  g3.setLineDash(dash || []); g3.strokeStyle = color; g3.lineWidth = width || 1;
  g3.beginPath(); g3.moveTo(pa[0], pa[1]); g3.lineTo(pb[0], pb[1]); g3.stroke(); g3.setLineDash([]);
}
function poly3(points, stroke, fill, width) {
  const ps = points.map(P);
  if (ps.some(p => !p)) return;
  g3.beginPath(); ps.forEach((p, i) => i ? g3.lineTo(p[0], p[1]) : g3.moveTo(p[0], p[1])); g3.closePath();
  if (fill) { g3.fillStyle = fill; g3.fill(); }
  if (stroke) { g3.strokeStyle = stroke; g3.lineWidth = width || 1; g3.stroke(); }
}
function ring3(c, r, color) {
  poly3([...Array(24).keys()].map(k => [c[0] + r * Math.cos(k * Math.PI / 12), c[1] + r * Math.sin(k * Math.PI / 12), c[2]]), color, null, 2);
}
function ball3(p, radius, color, outline) {
  const s = P(p);
  if (!s) return;
  g3.beginPath(); g3.arc(s[0], s[1], Math.max(3, cam.F * radius / s[2]), 0, 2 * Math.PI);
  g3.fillStyle = color; g3.fill();
  if (outline) { g3.strokeStyle = outline; g3.lineWidth = 3; g3.stroke(); }
}
function text3(p, text, color) {
  const s = P(p);
  if (!s) return;
  g3.fillStyle = color; g3.font = "13px system-ui"; g3.fillText(text, s[0] + 10, s[1] - 10);
}

function draw3d() {
  setupCamera();
  g3.clearRect(0, 0, c3.width, c3.height);
  for (let v = -2.5; v <= 2.5001; v += 0.5) seg3([-0.5, v, 0], [4.5, v, 0], "#2a2f38");  // floor grid
  for (let v = -0.5; v <= 4.5001; v += 0.5) seg3([v, -2.5, 0], [v, 2.5, 0], "#2a2f38");
  seg3([0, 0, 0], [0.5, 0, 0], "#c55", 2); seg3([0, 0, 0], [0, 0.5, 0], "#5c5", 2);  // world axes at the table
  if (!game) return;
  const z = game.table[2], [[x0, x1], [y0, y1]] = game.fence;
  poly3([[x0, y0, 0], [x1, y0, 0], [x1, y1, 0], [x0, y1, 0]], "#666");  // fence on the floor and in the flight plane
  poly3([[x0, y0, z], [x1, y0, z], [x1, y1, z], [x0, y1, z]], "#88888855");
  [[x0, y0], [x1, y0], [x1, y1], [x0, y1]].forEach(([x, y]) => seg3([x, y, 0], [x, y, z], "#88888855"));
  const [tx, ty] = game.table, tw = 0.25, td = 0.15;  // protected table: a box up to flight height
  const top = [[tx - tw, ty - td, z], [tx + tw, ty - td, z], [tx + tw, ty + td, z], [tx - tw, ty + td, z]];
  top.forEach(([x, y]) => seg3([x, y, 0], [x, y, z], "#2c8a4a", 2));
  poly3(top.map(([x, y]) => [x, y, 0]), "#2c8a4a"); poly3(top, "#2c8a4a", "#2c8a4a88", 2);

  const items = [];  // painter's algorithm: draw far objects first
  game.attackers.forEach((a, i) => items.push({ p: a, draw: () => {
    const live = active(i);
    if (live) seg3(a, game.table, "#a33", 1, [4, 4]);
    seg3(a, [a[0], a[1], 0], live ? "#a3333388" : "#55555588", 1, [2, 3]);
    ball3(a, 0.09, live ? "#e33" : "#555");
    text3(a, "A" + (i + 1) + (live ? "" : " " + game.attacker_status[i].toUpperCase()), live ? "#e88" : "#888");
  } }));
  game.defenders.forEach((d, i) => items.push({ p: d.pos, draw: () => {
    const real = game.slot === i && game.real.connected;
    const color = !d.alive ? "#666" : real ? "#f5c542" : d.role === "intercept" ? "#ff7a45" : "#4c8dff";
    if (d.alive && d.target) { ring3(d.target, 0.12, color); seg3(d.pos, d.target, color, 1, [2, 3]); }
    seg3(d.pos, [d.pos[0], d.pos[1], 0], "#4c8dff88", 1, [2, 3]);
    const nose = [d.pos[0] + 0.3 * Math.cos(d.yaw), d.pos[1] + 0.3 * Math.sin(d.yaw), d.pos[2]];
    seg3(d.pos, nose, color, 3);
    ball3(d.pos, 0.08, d.alive ? "#4c8dff" : "#666", real ? "#f5c542" : null);
    text3(d.pos, d.name + ROLE_TAG[d.role] + (real ? " (real)" : ""), !d.alive ? "#888" : real ? "#f5c542" : "#9bf");
  } }));
  items.map(it => ({ ...it, depth: (P(it.p) || [0, 0, 1e9])[2] })).sort((a, b) => b.depth - a.depth).forEach(it => it.draw());
}

let orbit = null;
c3.addEventListener("pointerdown", ev => { c3.setPointerCapture(ev.pointerId); orbit = [ev.clientX, ev.clientY]; c3.style.cursor = "grabbing"; });
c3.addEventListener("pointermove", ev => {
  if (!orbit) return;
  view.az -= (ev.clientX - orbit[0]) * 0.008;
  view.el = Math.min(1.5, Math.max(0.05, view.el + (ev.clientY - orbit[1]) * 0.008));
  orbit = [ev.clientX, ev.clientY]; draw3d();
});
["pointerup", "pointercancel"].forEach(t => c3.addEventListener(t, () => { orbit = null; c3.style.cursor = "grab"; }));
c3.addEventListener("wheel", ev => {
  ev.preventDefault(); view.r = Math.min(15, Math.max(2, view.r * (ev.deltaY > 0 ? 1.1 : 1 / 1.1))); draw3d();
}, { passive: false });
c3.addEventListener("dblclick", () => { Object.assign(view, VIEW0); draw3d(); });

function sendAttacker(ev, force) {
  const r = canvas.getBoundingClientRect();
  const [x, y] = toWorld((ev.clientX - r.left) * canvas.width / r.width, (ev.clientY - r.top) * canvas.height / r.height);
  if (drag === null && game) {  // grab the nearest attacker
    const live = game.attackers.map((a, i) => i).filter(active);
    drag = live.length ? live.reduce((best, i) => Math.hypot(game.attackers[i][0] - x, game.attackers[i][1] - y) <
      Math.hypot(game.attackers[best][0] - x, game.attackers[best][1] - y) ? i : best) : null;
  }
  if (drag === null) return;
  game.attackers[drag] = [x, y, game.attackers[drag][2]];
  const now = performance.now();
  if (force || now - lastSent > 80) { lastSent = now; post("/attacker", { i: drag, x, y }); }
  draw();
}
let dragging = false;
canvas.addEventListener("pointerdown", ev => { canvas.setPointerCapture(ev.pointerId); dragging = true; drag = null; sendAttacker(ev, true); });
canvas.addEventListener("pointermove", ev => dragging && sendAttacker(ev));
["pointerup", "pointercancel"].forEach(t => canvas.addEventListener(t, ev => {
  if (dragging) sendAttacker(ev, true); dragging = false; drag = null;
}));

["connect", "takeoff", "land", "emergency", "reconnect", "restart_video"].forEach(id => $(id).onclick = () => command(id));
$("blocker").onclick = () => toggle("/blocker", "enabled");
$("face").onclick = () => toggle("/face", "face");
$("avoid").onclick = () => toggle("/avoid", "avoid");
$("attack").onclick = () => toggle("/attack", "attack_mode");
$("script").onclick = () => toggle("/script", "script");
const ROLE_TAG = { keeper: " GK", intercept: " ATTACK", out: " OUT", block: "" };
const active = j => game.attacker_status[j] === "active";
$("record").onclick = () => game && post("/record", { on: !game.recording });
$("reset").onclick = () => post("/reset");
$("slot").onchange = ev => post("/attach", { slot: ev.target.value === "" ? null : +ev.target.value });
const speedText = v => `${v.toFixed(2)} m/s (rc ${Math.round(v * (game ? game.rc_per_mps : 100))})`;
$("speed").oninput = ev => { $("speedval").textContent = speedText(+ev.target.value); post("/config", { max_speed: +ev.target.value }); };
document.addEventListener("keydown", ev => {
  if (ev.repeat || ev.target.tagName === "SELECT" || ev.target.tagName === "INPUT") return;
  const k = ev.key.toLowerCase();
  if (k === "x") command("emergency"); else if (k === "t") command("takeoff");
  else if (k === "l") command("land"); else if (k === "b") toggle("/blocker", "enabled");
});

// Camera: fetch single JPEGs in a loop (an MJPEG <img> can freeze silently).
const video = $("video");
const nextFrame = () => {
  if (game && game.real.connected) video.src = "/frame.jpg?" + Date.now(); else setTimeout(nextFrame, 1000);
};
video.onload = () => setTimeout(nextFrame, 30);
video.onerror = () => setTimeout(nextFrame, 500);
nextFrame();

async function poll() {
  try {
    const g = await (await fetch("/game")).json();
    if (dragging && drag !== null && game) g.attackers[drag] = game.attackers[drag];  // keep the local drag smooth
    game = g;
    const r = g.real;
    $("blocker").textContent = `Blocker ${g.enabled ? "ON" : "OFF"}`; $("blocker").classList.toggle("on", g.enabled);
    $("face").classList.toggle("on", g.face);
    $("avoid").classList.toggle("on", g.avoid);
    $("attack").textContent = g.attack_mode ? "Keeper: ATTACK" : "Keeper: guard"; $("attack").classList.toggle("on", g.attack_mode);
    $("script").textContent = g.script ? "Stop scripted attack" : "Run scripted attack"; $("script").classList.toggle("on", g.script);
    $("record").textContent = g.recording ? `Stop recording (${g.recording.frames} frames)` : "Record";
    $("record").classList.toggle("on", !!g.recording);
    $("connect").textContent = r.connected ? "Connected" : r.connecting ? "Connecting..." : "Connect";
    if (document.activeElement !== $("slot")) $("slot").value = g.slot === null ? "" : String(g.slot);
    if (document.activeElement !== $("speed")) { $("speed").value = g.max_speed; $("speedval").textContent = speedText(g.max_speed); }
    let s = "";
    if (r.connected) {
      const state = r.busy ? "busy" : r.flying ? "FLYING" : "landed";
      s += `<b>${state}</b><span>battery ${r.battery}%</span><span>height ${r.height} cm</span><span>temp ${r.temp}</span>` +
           `<span>imu yaw ${r.imu_yaw}&deg;</span>`;
      if (g.slot !== null) {
        const d = g.defenders[g.slot];
        s += `<span>world yaw ${(d.yaw * 180 / Math.PI).toFixed(0)}&deg;</span>` +
             `<span>pose ${g.pose_from_tags ? "tags" : "dead-reckoned"} (${d.pos[0].toFixed(2)}, ${d.pos[1].toFixed(2)})</span>` +
             `<span>rc ${d.rc.join(" ")}</span>`;
      }
      s += `<span>measured raw vx,vy ${r.measured_raw}</span>`;
      const age = r.video_age;
      $("videoinfo").textContent = age === null || age > 1e5 ? "no video frames yet (firewall on UDP 11111?)" :
        age > 1 ? `video stalled ${age.toFixed(1)} s` : `video ok (${(age * 1000).toFixed(0)} ms old)`;
      $("videoinfo").classList.toggle("bad", age === null || age > 1);
    } else {
      s += "<b>simulation only</b>";
      $("videoinfo").textContent = "no drone"; $("videoinfo").classList.remove("bad");
    }
    const sep = v => v === null ? "-" : v.toFixed(2);
    const tight = v => v !== null && v < g.min_separation ? ' class="bad"' : "";
    s += `<span${tight(g.min_dist)}>closest pair ${sep(g.min_dist)} m</span>` +
         `<span${tight(g.min_dist_seen)}>closest since reset ${sep(g.min_dist_seen)} m</span>`;
    $("status").innerHTML = s + `<span class="message" id="m1"></span><span class="message" id="m2"></span>`;
    $("m1").textContent = g.message || ""; $("m2").textContent = r.message || "";
  } catch (e) {
    $("status").textContent = "server not responding";
  }
  draw();
  draw3d();
}
setInterval(poll, 100); poll();
</script></body></html>
"""


def make_handler(game: Game):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):  # keep the terminal quiet; the page polls every 100 ms
            pass

        def _send(self, code: int, body: bytes, content_type: str):
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = self.path.split("?")[0]
            if path == "/":
                self._send(200, PAGE.encode(), "text/html; charset=utf-8")
            elif path == "/game":
                self._send(200, json.dumps(game.state()).encode(), "application/json")
            elif path == "/frame.jpg":
                drone = game.drone
                frame = drone.reader.frame if drone is not None and drone.reader is not None else None
                data = jpeg(frame) if frame is not None else None
                if data:
                    self._send(200, data, "image/jpeg")
                else:
                    self._send(404, b"no frame", "text/plain")
            else:
                self._send(404, b"not found", "text/plain")

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            path = self.path
            drone = game.drone
            if path == "/attacker":
                game.set_attacker(int(body["i"]), body["x"], body["y"])
            elif path == "/blocker":
                game.set_enabled(bool(body.get("on")))
            elif path == "/face":
                game.face = bool(body.get("on"))
            elif path == "/avoid":
                game.avoid = bool(body.get("on"))
            elif path == "/attack":
                game.attack_mode = bool(body.get("on"))
            elif path == "/script":
                game.start_script(bool(body.get("on")))
            elif path == "/config":
                game.set_max_speed(body["max_speed"])
            elif path == "/attach":
                slot = body.get("slot")
                game.attach(None if slot is None else int(slot))
            elif path == "/pose":
                game.set_pose(body["x"], body["y"], body.get("yaw"))
            elif path == "/record":
                game.set_recording(bool(body.get("on")))
            elif path == "/reset":
                game.reset()
            elif path == "/cmd/connect":
                game.connect()
            elif path == "/cmd/takeoff":
                game.takeoff()
            elif path.startswith("/cmd/"):
                if drone is not None:
                    drone.command(path[len("/cmd/"):])
                    if path in ("/cmd/land", "/cmd/emergency"):
                        game._hover()
            else:
                self._send(404, b"not found", "text/plain")
                return
            self._send(200, b"{}", "application/json")

    return Handler


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--connect", action="store_true", help="connect to the Tello at startup")
    parser.add_argument("--max-speed", type=float, default=CFG.max_speed, help="m/s cap (also set on the page)")
    parser.add_argument("--gain", type=float, default=CFG.gain, help="1/s; m/s per metre of error")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true", help="do not open the page automatically")
    args = parser.parse_args()

    game = Game(replace(CFG, max_speed=args.max_speed, defender_speed=args.max_speed, gain=args.gain))
    threading.Thread(target=game.loop, daemon=True).start()
    if args.connect:
        game.connect()

    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(game))
    server.daemon_threads = True
    url = f"http://127.0.0.1:{args.port}"
    print(f"[INFO] Blocker test page at {url} (Ctrl+C lands and quits)")
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("[WARN] Interrupted; landing.")
    finally:
        server.server_close()
        game.set_enabled(False)
        game.set_recording(False)
        drone = game.drone
        if drone is not None:
            tello = drone.tello
            if drone.flying:
                try:
                    tello.send_rc_control(0, 0, 0, 0)
                    tello.land()
                except Exception:
                    tello.emergency()  # cuts motors: last resort only
            tello.streamoff()
            tello.end()
    return 0


if __name__ == "__main__":
    sys.exit(main())
