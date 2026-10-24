# Copyright (c) 2026, Action Space team.
# SPDX-License-Identifier: Apache-2.0
"""Build a standalone 3D replay page from judge_defend.py outputs (no server, no libraries).

    python scripts/make_replay.py replay_chase.json replay_team.json -o replay.html

One tab per file, with that run's scores; pick a recorded environment, play/pause, scrub,
change speed; drag to orbit, scroll to zoom. Site-centred frame: site cylinder at the origin.
"""

import argparse
import json
from pathlib import Path

PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Defense Replay</title>
<style>
  :root { --bg: #111; --panel: #1b1f26; --text: #eee; --muted: #999; --line: #2a2f38; }
  body { margin: 0; background: var(--bg); color: var(--text); font: 15px system-ui, sans-serif; }
  main { max-width: 1100px; margin: 0 auto; padding: 16px; }
  .tabs, .row { display: flex; gap: 8px; flex-wrap: wrap; align-items: center; margin: 8px 0; }
  button, select { font: inherit; padding: 10px 14px; border: 0; border-radius: 6px; background: #2a2f38; color: var(--text); cursor: pointer; }
  button.on { background: #2f5fd0; }
  #view { width: 100%; background: var(--panel); border-radius: 8px; touch-action: none; cursor: grab; display: block; }
  #scores { display: flex; gap: 18px; flex-wrap: wrap; margin: 8px 0; font-variant-numeric: tabular-nums; }
  #scores b { color: #fff; }
  input[type=range] { flex: 1; min-width: 160px; }
  .muted { color: var(--muted); }
  #event { color: #fc6; min-height: 1.3em; }
</style></head>
<body><main>
  <div class="tabs" id="tabs"></div>
  <div id="scores"></div>
  <canvas id="view" width="1100" height="620"></canvas>
  <div class="row">
    <button id="play">Pause</button>
    <label>Env <select id="env"></select></label>
    <label>Speed <select id="speed"><option>0.5</option><option selected>1</option><option>2</option><option>4</option></select>x</label>
    <input id="scrub" type="range" min="0" value="0"><span id="clock" class="muted"></span>
  </div>
  <div id="event"></div>
  <p class="muted">Blue = defenders (label shows role), red = attackers (grey once captured), green cylinder = protected site
    (an attacker inside it horizontally is a breach), faint rings = capture radius. Drag to orbit, scroll to zoom, double-click to reset.</p>
</main>
<script>
const RUNS = __RUNS__;
const $ = id => document.getElementById(id);
const c = $("view"), g = c.getContext("2d");
let run = 0, env = 0, frame = 0, playing = true, last = performance.now(), acc = 0;
const VIEW0 = { az: -2.2, el: 0.5, r: 15 }, view = { ...VIEW0 };

function cam() {
  const ce = Math.cos(view.el), eye = [view.r * ce * Math.cos(view.az), view.r * ce * Math.sin(view.az), view.r * Math.sin(view.el)];
  const n = v => { const l = Math.hypot(...v); return v.map(x => x / l); };
  const f = n([-eye[0], -eye[1], -eye[2]]), right = n([f[1], -f[0], 0]);
  const up = [right[1] * f[2] - right[2] * f[1], right[2] * f[0] - right[0] * f[2], right[0] * f[1] - right[1] * f[0]];
  return { eye, f, right, up, F: c.height * 1.4 };
}
let K;
function P(p) {
  const d = [p[0] - K.eye[0], p[1] - K.eye[1], p[2] - K.eye[2]];
  const z = d[0] * K.f[0] + d[1] * K.f[1] + d[2] * K.f[2];
  if (z < 0.1) return null;
  return [c.width / 2 + K.F * (d[0] * K.right[0] + d[1] * K.right[1] + d[2] * K.right[2]) / z,
          c.height / 2 - K.F * (d[0] * K.up[0] + d[1] * K.up[1] + d[2] * K.up[2]) / z, z];
}
function seg(a, b, color, w, dash) {
  const pa = P(a), pb = P(b); if (!pa || !pb) return;
  g.setLineDash(dash || []); g.strokeStyle = color; g.lineWidth = w || 1;
  g.beginPath(); g.moveTo(pa[0], pa[1]); g.lineTo(pb[0], pb[1]); g.stroke(); g.setLineDash([]);
}
function circle(cx, cy, cz, r, color, w) {
  const pts = [...Array(40).keys()].map(k => P([cx + r * Math.cos(k * Math.PI / 20), cy + r * Math.sin(k * Math.PI / 20), cz]));
  if (pts.some(p => !p)) return;
  g.strokeStyle = color; g.lineWidth = w || 1; g.beginPath();
  pts.forEach((p, i) => i ? g.lineTo(p[0], p[1]) : g.moveTo(p[0], p[1])); g.closePath(); g.stroke();
}
function ball(p, r, color, label, lc) {
  const s = P(p); if (!s) return;
  g.beginPath(); g.arc(s[0], s[1], Math.max(3, K.F * r / s[2]), 0, 2 * Math.PI); g.fillStyle = color; g.fill();
  if (label) { g.fillStyle = lc; g.font = "13px system-ui"; g.fillText(label, s[0] + 9, s[1] - 9); }
}

function draw() {
  const R = RUNS[run], F = R.replays[env], fr = F[Math.min(frame, F.length - 1)];
  K = cam(); g.clearRect(0, 0, c.width, c.height);
  const zf = -R.site_height / 2;  // ground, in the site-centred frame
  for (let v = -8; v <= 8; v += 1) { seg([-8, v, zf], [8, v, zf], "#2a2f38"); seg([v, -8, zf], [v, 8, zf], "#2a2f38"); }
  circle(0, 0, zf, R.site_radius, "#2c8a4a", 2); circle(0, 0, zf + R.site_height, R.site_radius, "#2c8a4a88", 2);
  for (let k = 0; k < 8; k++) { const a = k * Math.PI / 4; seg([R.site_radius * Math.cos(a), R.site_radius * Math.sin(a), zf],
      [R.site_radius * Math.cos(a), R.site_radius * Math.sin(a), zf + R.site_height], "#2c8a4a55"); }
  circle(0, 0, zf, 6.0, "#a3333355", 1);  // attacker spawn ring
  const items = [];
  fr.att.forEach((p, j) => items.push({ p, draw: () => {
    const live = fr.alive[j];
    if (live) seg(p, [0, 0, 0], "#a33", 1, [4, 4]);
    seg(p, [p[0], p[1], zf], live ? "#a3333388" : "#55555588", 1, [2, 3]);
    ball(p, 0.12, live ? "#e33" : "#555", "A" + (j + 1) + (live ? "" : " down"), live ? "#e88" : "#888");
  } }));
  fr.def.forEach((p, i) => items.push({ p, draw: () => {
    seg(p, [p[0], p[1], zf], "#4c8dff88", 1, [2, 3]);
    circle(p[0], p[1], p[2], R.capture_radius, "#4c8dff55", 1);
    const role = fr.roles ? " " + ({ block: "block", keeper: "GK", intercept: "ATTACK" }[fr.roles[i]] || fr.roles[i]) : "";
    ball(p, 0.12, "#4c8dff", "D" + (i + 1) + role, "#9bf");
  } }));
  items.map(it => ({ ...it, d: (P(it.p) || [0, 0, 1e9])[2] })).sort((a, b) => b.d - a.d).forEach(it => it.draw());
  $("scrub").max = F.length - 1; $("scrub").value = frame;
  $("clock").textContent = `t = ${fr.t.toFixed(1)} s`;
  const recent = F.slice(Math.max(0, frame - 30), frame + 1).reverse().find(x => x.end);
  $("event").textContent = recent ? (recent.end === "breach" ? "BREACH - an attacker reached the site; new episode" : "episode timed out (20 s); new episode") : "";
}

function selectRun(i) {
  run = i; env = 0; frame = 0;
  document.querySelectorAll("#tabs button").forEach((b, k) => b.classList.toggle("on", k === i));
  const R = RUNS[i], pct = v => (100 * v).toFixed(0) + "%";
  $("scores").innerHTML = `<span><b>${R.policy}</b> on ${R.task}, ${R.defenders}v${R.attackers}</span>` +
    `<span>episodes <b>${R.episodes}</b></span><span>interception <b>${pct(R.interception_rate)}</b></span>` +
    `<span>breach <b>${pct(R.breach_rate)}</b></span><span>time to intercept <b>${R.mean_time_to_intercept_s == null ? "-" : R.mean_time_to_intercept_s.toFixed(1) + " s"}</b></span>`;
  $("env").innerHTML = R.replays.map((_, k) => `<option value="${k}">${k}</option>`).join("");
  draw();
}
$("tabs").innerHTML = RUNS.map((R, i) => `<button>${R.policy} ${R.defenders}v${R.attackers}</button>`).join("");
document.querySelectorAll("#tabs button").forEach((b, i) => b.onclick = () => selectRun(i));
$("env").onchange = e => { env = +e.target.value; frame = 0; draw(); };
$("play").onclick = () => { playing = !playing; $("play").textContent = playing ? "Pause" : "Play"; };
$("scrub").oninput = e => { frame = +e.target.value; draw(); };
let drag = null;
c.addEventListener("pointerdown", e => { c.setPointerCapture(e.pointerId); drag = [e.clientX, e.clientY]; });
c.addEventListener("pointermove", e => { if (!drag) return; view.az -= (e.clientX - drag[0]) * 0.008;
  view.el = Math.min(1.5, Math.max(0.05, view.el + (e.clientY - drag[1]) * 0.008)); drag = [e.clientX, e.clientY]; draw(); });
["pointerup", "pointercancel"].forEach(t => c.addEventListener(t, () => drag = null));
c.addEventListener("wheel", e => { e.preventDefault(); view.r = Math.min(40, Math.max(4, view.r * (e.deltaY > 0 ? 1.1 : 1 / 1.1))); draw(); }, { passive: false });
c.addEventListener("dblclick", () => { Object.assign(view, VIEW0); draw(); });
function tick(now) {
  const R = RUNS[run], F = R.replays[env];
  if (playing) { acc += (now - last) / 1000 * +$("speed").value; while (acc >= R.dt) { acc -= R.dt; frame = (frame + 1) % F.length; } }
  last = now; draw(); requestAnimationFrame(tick);
}
selectRun(0); requestAnimationFrame(tick);
</script></body></html>
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("replays", nargs="+", help="judge_defend.py --out files")
    parser.add_argument("-o", "--out", default="replay.html")
    args = parser.parse_args()
    runs = [json.loads(Path(p).read_text()) for p in args.replays]
    Path(args.out).write_text(PAGE.replace("__RUNS__", json.dumps(runs)), encoding="utf-8")
    print(f"wrote {args.out} ({len(runs)} runs)")


if __name__ == "__main__":
    main()
