# Copyright (c) 2026, Action Space team.
# SPDX-License-Identifier: Apache-2.0
"""Real-hardware test: the blocker policy on one Tello against a virtual attacker.

Drag the red attacker on the map; with Blocker ON the drone flies to stay
between it and the green table. Connect to the Tello's Wi-Fi first, then:

    python tests/tello_blocker_web.py

and open http://127.0.0.1:8765 (opened automatically).

Position is dead-reckoned from the commands we send (origin = takeoff spot,
+x = the drone's forward at takeoff, +y = its left), so this tests the policy
and control loop, not localization. Real motion differs by the rc_per_mps
calibration error; compare commanded vs measured speed on the page. Heading
and height are held. The block target is kept inside FENCE.

    t take off   l land   b blocker on/off   x EMERGENCY (cuts motors)
"""

import argparse
import json
import sys
import threading
import time
import webbrowser
from http.server import ThreadingHTTPServer
from pathlib import Path

import numpy as np
from djitellopy import Tello

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from defense.blocker import BlockerConfig, block_point, to_rc, velocity_command  # noqa: E402
from tello_web import Drone, make_handler  # noqa: E402

Z = 1.0  # planar test: all points share one height; vertical is not commanded
TABLE = np.array((-1.0, 0.0, Z))  # 1 m behind the takeoff spot
ATTACKER_START = np.array((2.5, 0.0, Z))
FENCE = ((-0.6, 1.5), (-1.0, 1.0))  # x and y limits for the block target, m
LOOP_S = 0.1  # 10 Hz control loop
ATTACKER_STALE_S = 0.3  # attacker velocity drops to zero when not dragged

CFG = BlockerConfig(max_speed=0.6, gain=1.5)  # override with --max-speed / --gain


class Game:
    """Virtual attacker, dead-reckoned defender, and the blocker control loop."""

    def __init__(self, drone: Drone, cfg: BlockerConfig):
        self.drone = drone
        self.cfg = cfg
        self.lock = threading.Lock()
        self.enabled = False
        self.pos = np.array((0.0, 0.0, Z))
        self.target = self.pos.copy()
        self.attacker = ATTACKER_START.copy()
        self.attacker_vel = np.zeros(3)
        self.attacker_time = 0.0
        self.cmd_speed = 0.0

    def reset_origin(self):
        with self.lock:
            self.pos = np.array((0.0, 0.0, Z))

    def set_attacker(self, x: float, y: float):
        now = time.monotonic()
        new = np.array((float(x), float(y), Z))
        with self.lock:
            dt = now - self.attacker_time
            if dt < ATTACKER_STALE_S:
                self.attacker_vel = 0.7 * self.attacker_vel + 0.3 * (new - self.attacker) / max(dt, 1e-3)
            self.attacker = new
            self.attacker_time = now

    def set_enabled(self, on: bool):
        with self.lock:
            self.enabled = on
        if not on:
            with self.drone.lock:
                if self.drone.flying and not self.drone.busy:
                    self.drone.tello.send_rc_control(0, 0, 0, 0)
                self.drone.moving = False

    def loop(self):
        while True:
            time.sleep(LOOP_S)
            with self.lock:
                if time.monotonic() - self.attacker_time > ATTACKER_STALE_S:
                    self.attacker_vel = np.zeros(3)
                target = block_point(self.attacker, self.attacker_vel, TABLE, self.pos, self.cfg)
                target[0] = np.clip(target[0], *FENCE[0])
                target[1] = np.clip(target[1], *FENCE[1])
                self.target = target
                pos = self.pos.copy()
                enabled = self.enabled
            if not enabled:
                continue
            with self.drone.lock:
                if not self.drone.flying or self.drone.busy:
                    continue
                lr, fb, _, _ = to_rc(velocity_command(pos, target, self.cfg), 0.0, self.cfg)
                self.drone.tello.send_rc_control(lr, fb, 0, 0)
                self.drone.moving = bool(lr or fb)
                self.drone.last_rc = time.monotonic()
            vel = np.array((fb, -lr, 0.0)) / self.cfg.rc_per_mps  # yaw held at 0: forward = +x, right = -y
            with self.lock:
                self.pos += vel * LOOP_S
                self.cmd_speed = float(np.linalg.norm(vel))

    def state(self) -> dict:
        with self.lock:
            game = {
                "enabled": self.enabled,
                "pos": self.pos.tolist(),
                "target": self.target.tolist(),
                "attacker": self.attacker.tolist(),
                "table": TABLE.tolist(),
                "fence": FENCE,
                "cmd_speed": round(self.cmd_speed, 2),
            }
        tello = self.drone.tello
        # Raw Tello velocity readings, for calibrating rc_per_mps (units per SDK: likely dm/s).
        game["measured_raw"] = [tello.get_speed_x(), tello.get_speed_y()]
        return {**self.drone.status(), **game}


PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Tello Blocker Test</title>
<style>
  body { margin: 0; background: #111; color: #eee; font: 15px system-ui, sans-serif; }
  main { max-width: 1100px; margin: 0 auto; padding: 16px; }
  .cols { display: flex; gap: 16px; flex-wrap: wrap; align-items: flex-start; }
  #map { background: #1b1f26; border-radius: 8px; touch-action: none; cursor: crosshair; max-width: 100%; }
  #video { flex: 1; min-width: 280px; aspect-ratio: 4 / 3; background: #000; border-radius: 8px; object-fit: contain; }
  #status { display: flex; gap: 16px; flex-wrap: wrap; margin: 12px 0; font-variant-numeric: tabular-nums; }
  #message { color: #fc6; }
  .row { display: flex; gap: 8px; flex-wrap: wrap; }
  button { font: inherit; padding: 12px 16px; border: 0; border-radius: 6px; background: #2a2f38; color: #eee; cursor: pointer; }
  #takeoff { background: #1f6b3a; } #land { background: #6b5a1f; }
  #blocker.on { background: #2f5fd0; }
  #emergency { background: #a11; margin-left: auto; font-weight: 700; }
  p.help { color: #999; }
</style></head>
<body><main>
  <div class="cols">
    <canvas id="map" width="400" height="500"></canvas>
    <img id="video" src="/video" alt="Tello camera feed">
  </div>
  <div id="status"></div>
  <div class="row">
    <button id="takeoff">Take off <kbd>t</kbd></button>
    <button id="land">Land <kbd>l</kbd></button>
    <button id="blocker">Blocker OFF <kbd>b</kbd></button>
    <button id="reconnect" title="Redo the connection and video, e.g. after a battery swap">Reconnect</button>
    <button id="emergency">EMERGENCY <kbd>x</kbd></button>
  </div>
  <p class="help">Drag anywhere on the map to move the red attacker. Map up = drone's forward at takeoff.
    Blue dot = estimated drone position, blue ring = block target, dashed box = fence, green = table.</p>
</main>
<script>
const S = 100, X_MAX = 3.5, Y_MAX = 2.0;  // px per metre; map top edge x, left edge y
const toPx = (x, y) => [(Y_MAX - y) * S, (X_MAX - x) * S];
const toWorld = (px, py) => [X_MAX - py / S, Y_MAX - px / S];
const canvas = document.getElementById("map"), ctx = canvas.getContext("2d");
let game = null, dragging = false, lastSent = 0;

const post = (path, body) => fetch(path, { method: "POST", headers: { "Content-Type": "application/json" },
                                           body: JSON.stringify(body || {}) }).catch(() => {});
const command = name => post("/cmd/" + name);
const toggleBlocker = () => game && post("/blocker", { on: !game.enabled });

function dot(x, y, r, color, ring) {
  const [px, py] = toPx(x, y);
  ctx.beginPath(); ctx.arc(px, py, r, 0, 2 * Math.PI);
  if (ring) { ctx.strokeStyle = color; ctx.lineWidth = 2; ctx.stroke(); } else { ctx.fillStyle = color; ctx.fill(); }
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
  const [ax, ay] = toPx(game.attacker[0], game.attacker[1]);
  ctx.strokeStyle = "#a33"; ctx.setLineDash([3, 3]); ctx.beginPath(); ctx.moveTo(ax, ay); ctx.lineTo(tx, ty); ctx.stroke(); ctx.setLineDash([]);
  dot(game.attacker[0], game.attacker[1], 10, "#e33");
  dot(game.target[0], game.target[1], 12, "#4c8dff", true);
  dot(game.pos[0], game.pos[1], 8, "#4c8dff");
}

function sendAttacker(ev, force) {
  const r = canvas.getBoundingClientRect();
  const [x, y] = toWorld((ev.clientX - r.left) * canvas.width / r.width, (ev.clientY - r.top) * canvas.height / r.height);
  if (game) game.attacker = [x, y, game.attacker[2]];
  const now = performance.now();
  if (force || now - lastSent > 80) { lastSent = now; post("/attacker", { x, y }); }
  draw();
}
canvas.addEventListener("pointerdown", ev => { dragging = true; canvas.setPointerCapture(ev.pointerId); sendAttacker(ev, true); });
canvas.addEventListener("pointermove", ev => dragging && sendAttacker(ev));
["pointerup", "pointercancel"].forEach(t => canvas.addEventListener(t, ev => { if (dragging) sendAttacker(ev, true); dragging = false; }));

document.getElementById("takeoff").onclick = () => command("takeoff");
document.getElementById("land").onclick = () => command("land");
document.getElementById("emergency").onclick = () => command("emergency");
document.getElementById("blocker").onclick = toggleBlocker;
document.getElementById("reconnect").onclick = () => command("reconnect");
document.addEventListener("keydown", ev => {
  if (ev.repeat) return;
  const k = ev.key.toLowerCase();
  if (k === "x") command("emergency"); else if (k === "t") command("takeoff");
  else if (k === "l") command("land"); else if (k === "b") toggleBlocker();
});

async function poll() {
  try {
    const g = await (await fetch("/game")).json();
    if (dragging && game) g.attacker = game.attacker;  // keep the local drag smooth
    game = g;
    const b = document.getElementById("blocker");
    b.textContent = `Blocker ${g.enabled ? "ON" : "OFF"}`; b.classList.toggle("on", g.enabled);
    const state = g.busy ? "busy" : g.flying ? "FLYING" : "landed";
    document.getElementById("status").innerHTML =
      `<b>${state}</b><span>battery ${g.battery}%</span><span>height ${g.height} cm</span><span>temp ${g.temp}</span>` +
      `<span>cmd ${g.cmd_speed} m/s</span><span>measured raw vx,vy ${g.measured_raw}</span>` +
      `<span id="message">${g.message}</span>`;
  } catch (e) {
    document.getElementById("status").textContent = "server not responding";
  }
  draw();
}
setInterval(poll, 100); poll();
</script></body></html>
"""


def make_game_handler(drone: Drone, game: Game):
    Base = make_handler(drone)

    class Handler(Base):
        def do_GET(self):
            if self.path == "/":
                self._send(200, PAGE.encode(), "text/html; charset=utf-8")
            elif self.path == "/game":
                self._send(200, json.dumps(game.state()).encode(), "application/json")
            else:
                super().do_GET()

        def do_POST(self):
            if self.path == "/rc":  # manual moves would break the dead-reckoned position
                self.rfile.read(int(self.headers.get("Content-Length", 0)))
                self._send(200, b"{}", "application/json")
                return
            if self.path in ("/attacker", "/blocker"):
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
                if self.path == "/attacker":
                    game.set_attacker(body["x"], body["y"])
                else:
                    game.set_enabled(bool(body.get("on")))
                self._send(200, b"{}", "application/json")
                return
            if self.path == "/cmd/takeoff" and not drone.flying:
                game.reset_origin()
            elif self.path in ("/cmd/land", "/cmd/emergency"):
                game.set_enabled(False)
            super().do_POST()

    return Handler


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--max-speed", type=float, default=CFG.max_speed, help="m/s cap (rc = m/s x rc_per_mps)")
    parser.add_argument("--gain", type=float, default=CFG.gain, help="1/s; m/s per metre of error")
    args = parser.parse_args()
    cfg = BlockerConfig(max_speed=args.max_speed, gain=args.gain, defender_speed=args.max_speed)

    tello = Tello()
    tello.connect()
    print(f"[INFO] Connected. Battery {tello.get_battery()}%")
    tello.streamon()
    reader = tello.get_frame_read()

    drone = Drone(tello, speed=30, reader=reader)
    game = Game(drone, cfg)
    threading.Thread(target=drone.watchdog, daemon=True).start()
    threading.Thread(target=game.loop, daemon=True).start()

    port = 8765
    server = ThreadingHTTPServer(("127.0.0.1", port), make_game_handler(drone, game))
    server.daemon_threads = True
    url = f"http://127.0.0.1:{port}"
    print(f"[INFO] Blocker test page at {url} (Ctrl+C lands and quits)")
    webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("[WARN] Interrupted; landing.")
    finally:
        server.server_close()
        game.set_enabled(False)
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
