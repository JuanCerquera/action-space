# Copyright (c) 2026, Action Space team.
# SPDX-License-Identifier: Apache-2.0
"""Real-hardware check: fly a Tello from a local web page with a live camera feed.

Connect this machine to the Tello's Wi-Fi first (inbound UDP 8890 and
11111 must be allowed through the firewall), then:

    python tests/tello_web.py --speed 30

and open http://127.0.0.1:8765 (opened automatically). The server listens
on localhost only. Hold keys or on-screen buttons to move; the drone stops
when they are released, or if the page stops sending for HOLD_S seconds.

    t take off   l land   w/s forward/back   a/d left/right
    r/f up/down  q/e turn   x EMERGENCY (cuts all motors, the drone drops)

Ctrl+C in the terminal lands the drone and stops the server.
"""

import argparse
import json
import sys
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
from djitellopy import Tello


MIN_BATTERY = 20  # percent; refuse takeoff below this
HOLD_S = 0.3  # stop moving if the page sends no rc update within this window
KEEPALIVE_S = 1.0  # send a zero rc this often while hovering, so the Tello does not auto-land
GROUNDED_S = 3.0  # height 0 for this long while "flying" means the drone landed on its own
FRAME_S = 0.04  # ~25 fps video to the browser


class Drone:
    """Tello state shared between HTTP handler threads."""

    def __init__(self, tello: Tello, speed: int, reader=None):
        self.tello = tello
        self.reader = reader  # video frame reader; replaced on reconnect
        self.speed = speed
        self.flying = False
        self.busy = False  # True while takeoff/land waits for the drone's reply
        self.moving = False
        self.last_rc = 0.0
        self.message = "press Take off (t)"
        self.lock = threading.Lock()

    def command(self, name: str):
        with self.lock:
            if name == "emergency":
                self.tello.emergency()
                self.flying = False
                self.message = "EMERGENCY STOP"
                print("[WARN] Emergency stop.")
            elif name == "takeoff" and not self.flying and not self.busy:
                if self.tello.get_battery() < MIN_BATTERY:
                    self.message = f"battery below {MIN_BATTERY}%, not taking off"
                    return
                self.flying = True  # set first: takeoff can time out after the drone is already up
                self._run_blocking(self.tello.takeoff, "taking off...")
            elif name == "land" and self.flying and not self.busy:
                self.tello.send_rc_control(0, 0, 0, 0)
                self.flying = False
                self._run_blocking(self.tello.land, "landing...")
            elif name == "reconnect" and not self.busy:
                if self.flying:
                    self.message = "land before reconnecting"
                    return
                self._run_blocking(self.connect_and_stream, "reconnecting...")

    def connect_and_stream(self):
        """Redo the SDK handshake and restart video, e.g. after the drone rebooted."""
        old = self.tello.background_frame_read
        if old is not None:
            old.stop()
            self.tello.background_frame_read = None
        self.tello.connect()
        self.tello.streamon()
        self.reader = self.tello.get_frame_read()
        print(f"[INFO] Reconnected. Battery {self.tello.get_battery()}%")

    def _run_blocking(self, command, message: str):
        def worker():
            try:
                command()
                result = ""
            except Exception as exc:
                result = f"{command.__name__} failed: {exc}"
                print(f"[ERROR] {result}")
            with self.lock:
                self.busy = False
                self.message = result

        self.busy = True
        self.message = message
        threading.Thread(target=worker, daemon=True).start()

    def rc(self, lr: float, fb: float, ud: float, yaw: float):
        with self.lock:
            if not self.flying or self.busy:
                return
            values = [int(self.speed * max(-1.0, min(1.0, float(v)))) for v in (lr, fb, ud, yaw)]
            self.tello.send_rc_control(*values)
            self.moving = any(values)
            self.last_rc = time.monotonic()

    def watchdog(self):
        grounded_since = None
        while True:
            with self.lock:
                now = time.monotonic()
                if self.moving and now - self.last_rc > HOLD_S:
                    self.tello.send_rc_control(0, 0, 0, 0)
                    self.moving = False
                if self.flying and not self.busy:
                    # Tello auto-lands after ~15 s without commands; a zero rc keeps it hovering.
                    if now - self.last_rc > KEEPALIVE_S:
                        self.tello.send_rc_control(0, 0, 0, 0)
                        self.last_rc = now
                    # Notice when it landed on its own (low battery, overheating, lost link).
                    if self.tello.get_height() <= 0:
                        grounded_since = grounded_since or now
                        if now - grounded_since > GROUNDED_S:
                            self.flying = False
                            self.message = "drone landed on its own"
                    else:
                        grounded_since = None
                else:
                    grounded_since = None
            time.sleep(0.05)

    def status(self) -> dict:
        with self.lock:
            return {
                "flying": self.flying,
                "busy": self.busy,
                "battery": self.tello.get_battery(),
                "height": self.tello.get_height(),
                "temp": self.tello.get_temperature(),
                "message": self.message,
            }


PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Tello Control</title>
<style>
  body { margin: 0; background: #111; color: #eee; font: 15px system-ui, sans-serif; }
  main { max-width: 980px; margin: 0 auto; padding: 16px; }
  #video { width: 100%; aspect-ratio: 4 / 3; background: #000; border-radius: 8px; object-fit: contain; }
  #status { display: flex; gap: 16px; flex-wrap: wrap; margin: 12px 0; font-variant-numeric: tabular-nums; }
  #state { font-weight: 700; }
  #message { color: #fc6; }
  .row { display: flex; gap: 8px; flex-wrap: wrap; margin: 8px 0; }
  button { font: inherit; padding: 12px 16px; border: 0; border-radius: 6px; background: #2a2f38; color: #eee;
           cursor: pointer; user-select: none; touch-action: none; min-width: 64px; }
  button:active, button.held { background: #46607e; }
  #takeoff { background: #1f6b3a; } #land { background: #6b5a1f; }
  #emergency { background: #a11; margin-left: auto; font-weight: 700; }
  .pads { display: flex; gap: 32px; flex-wrap: wrap; margin-top: 8px; }
  .pad { display: grid; grid-template-columns: repeat(3, 64px); gap: 6px; }
  .pad span { text-align: center; color: #888; grid-column: 1 / 4; }
  kbd { background: #2a2f38; padding: 1px 5px; border-radius: 3px; }
  p.help { color: #999; }
</style></head>
<body><main>
  <img id="video" src="/video" alt="Tello camera feed">
  <div id="status"><span id="state">connecting...</span><span id="battery"></span><span id="height"></span>
    <span id="temp"></span><span id="message"></span></div>
  <div class="row">
    <button id="takeoff">Take off <kbd>t</kbd></button>
    <button id="land">Land <kbd>l</kbd></button>
    <button id="reconnect" title="Redo the connection and video, e.g. after a battery swap">Reconnect</button>
    <button id="emergency" title="Cuts all motors: the drone drops">EMERGENCY <kbd>x</kbd></button>
  </div>
  <div class="pads">
    <div class="pad"><span>move</span>
      <i></i><button data-key="w">&#9650; w</button><i></i>
      <button data-key="a">&#9664; a</button><button data-key="s">&#9660; s</button><button data-key="d">&#9654; d</button>
    </div>
    <div class="pad"><span>height / turn</span>
      <i></i><button data-key="r">up r</button><i></i>
      <button data-key="q">&#8634; q</button><button data-key="f">down f</button><button data-key="e">&#8635; e</button>
    </div>
  </div>
  <p class="help">Hold a key or button to move; the drone stops when you let go.</p>
</main>
<script>
const MOVES = { w: [0,1,0,0], s: [0,-1,0,0], a: [-1,0,0,0], d: [1,0,0,0],
                r: [0,0,1,0], f: [0,0,-1,0], q: [0,0,0,-1], e: [0,0,0,1] };
const held = new Set();
let wasMoving = false;

const post = (path, body) => fetch(path, { method: "POST", headers: { "Content-Type": "application/json" },
                                           body: JSON.stringify(body || {}) }).catch(() => {});
const command = name => post("/cmd/" + name);

function sendRc() {
  const v = [0, 0, 0, 0];
  for (const k of held) MOVES[k].forEach((x, i) => v[i] += x);
  if (held.size || wasMoving) post("/rc", { lr: v[0], fb: v[1], ud: v[2], yaw: v[3] });
  wasMoving = held.size > 0;
}
setInterval(sendRc, 100);  // keep sending while held; the server stops the drone if updates stop

function press(k) { held.add(k); document.querySelector(`[data-key="${k}"]`)?.classList.add("held"); }
function release(k) { held.delete(k); document.querySelector(`[data-key="${k}"]`)?.classList.remove("held"); }

document.addEventListener("keydown", ev => {
  const k = ev.key.toLowerCase();
  if (ev.repeat && !(k in MOVES)) return;
  if (k === "x") command("emergency");
  else if (k === "t") command("takeoff");
  else if (k === "l") command("land");
  else if (k in MOVES) { press(k); ev.preventDefault(); }
});
document.addEventListener("keyup", ev => release(ev.key.toLowerCase()));
window.addEventListener("blur", () => [...held].forEach(release));

document.querySelectorAll("[data-key]").forEach(b => {
  const k = b.dataset.key;
  b.addEventListener("pointerdown", ev => { b.setPointerCapture(ev.pointerId); press(k); });
  ["pointerup", "pointercancel", "lostpointercapture"].forEach(t => b.addEventListener(t, () => release(k)));
});
document.getElementById("takeoff").onclick = () => command("takeoff");
document.getElementById("land").onclick = () => command("land");
document.getElementById("reconnect").onclick = () => command("reconnect");
document.getElementById("emergency").onclick = () => command("emergency");

async function poll() {
  try {
    const s = await (await fetch("/status")).json();
    document.getElementById("state").textContent = s.busy ? "busy" : s.flying ? "FLYING" : "landed";
    document.getElementById("battery").textContent = `battery ${s.battery}%`;
    document.getElementById("height").textContent = `height ${s.height} cm`;
    document.getElementById("temp").textContent = `temp ${s.temp}`;
    document.getElementById("message").textContent = s.message;
  } catch (e) {
    document.getElementById("state").textContent = "server not responding";
  }
}
setInterval(poll, 500); poll();
</script></body></html>
"""


def make_handler(drone: Drone):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):  # keep the terminal quiet; rc arrives every 100 ms
            pass

        def _send(self, code: int, body: bytes, content_type: str):
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/":
                self._send(200, PAGE.encode(), "text/html; charset=utf-8")
            elif self.path == "/status":
                self._send(200, json.dumps(drone.status()).encode(), "application/json")
            elif self.path == "/video":
                self._stream_video()
            else:
                self._send(404, b"not found", "text/plain")

        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length) or b"{}")
            if self.path.startswith("/cmd/"):
                drone.command(self.path[len("/cmd/"):])
            elif self.path == "/rc":
                drone.rc(body.get("lr", 0), body.get("fb", 0), body.get("ud", 0), body.get("yaw", 0))
            else:
                self._send(404, b"not found", "text/plain")
                return
            self._send(200, b"{}", "application/json")

        def _stream_video(self):
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            try:
                while True:
                    frame = drone.reader.frame if drone.reader is not None else None
                    if frame is not None:
                        # djitellopy delivers RGB; JPEG encoding expects BGR
                        ok, jpg = cv2.imencode(".jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR),
                                               [cv2.IMWRITE_JPEG_QUALITY, 70])
                        if ok:
                            data = jpg.tobytes()
                            self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n")
                            self.wfile.write(f"Content-Length: {len(data)}\r\n\r\n".encode())
                            self.wfile.write(data + b"\r\n")
                    time.sleep(FRAME_S)
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                pass  # browser tab closed or reloaded

    return Handler


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--speed", type=int, default=30, help="movement speed, 10-100 (default 30)")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    tello = Tello()
    tello.connect()
    print(f"[INFO] Connected. Battery {tello.get_battery()}%")
    tello.streamon()
    reader = tello.get_frame_read()

    drone = Drone(tello, max(10, min(100, args.speed)), reader)
    threading.Thread(target=drone.watchdog, daemon=True).start()

    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(drone))
    server.daemon_threads = True
    url = f"http://127.0.0.1:{args.port}"
    print(f"[INFO] Control page at {url} (Ctrl+C lands and quits)")
    webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("[WARN] Interrupted; landing.")
    finally:
        server.server_close()
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
