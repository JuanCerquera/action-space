# Copyright (c) 2025, Action Space hackathon team.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Bundle recorded episodes (media/replay/*.json) into self-contained 2D and 3D HTML replay viewers.

Record first (headless, no rendering needed), then build and open the page:

.. code-block:: bash

    python scripts/eval_controllers.py --headless --num_envs 4 --record_steps 1200 --task Intercept-Swarm-3v2-Bump-Direct-v0 --controllers block
    python scripts/build_replay.py

"""

import base64
import glob
import json
import os
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
REPLAY_DIR = os.path.join(REPO, "media", "replay")

PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>3v2 Intercept Replay</title>
<style>
  :root {
    --bg: #f6f7f9; --panel: #ffffff; --ink: #1d2330; --muted: #6b7385; --line: #d9dde5;
    --room: #eef0f4; --wall: #9aa3b2; --zone: rgba(220, 60, 60, 0.14); --zone-edge: rgba(220, 60, 60, 0.55);
    --def: #2563eb; --att: #dc2626; --down: #9aa3b2; --tag: #111111;
    --win: #15803d; --breach: #b91c1c; --crash: #b45309;
  }
  @media (prefers-color-scheme: dark) {
    :root {
      --bg: #12151b; --panel: #1a1f28; --ink: #e6e9ef; --muted: #98a1b3; --line: #2c3340;
      --room: #20262f; --wall: #5d6677; --zone: rgba(240, 80, 80, 0.16); --zone-edge: rgba(240, 100, 100, 0.6);
      --def: #60a5fa; --att: #f87171; --down: #5d6677; --tag: #e6e9ef;
      --win: #4ade80; --breach: #f87171; --crash: #fbbf24;
    }
  }
  * { box-sizing: border-box; }
  body { margin: 0; background: var(--bg); color: var(--ink); font: 14px/1.45 system-ui, -apple-system, "Segoe UI", sans-serif; }
  main { max-width: 1100px; margin: 0 auto; padding: 20px 16px 32px; }
  h1 { font-size: 20px; margin: 0 0 4px; }
  p.sub { margin: 0 0 16px; color: var(--muted); }
  .controls { display: flex; flex-wrap: wrap; gap: 10px 16px; align-items: center; margin-bottom: 12px; }
  .controls label { color: var(--muted); display: flex; gap: 6px; align-items: center; }
  select, button { font: inherit; color: var(--ink); background: var(--panel); border: 1px solid var(--line); border-radius: 6px; padding: 5px 10px; }
  button { cursor: pointer; min-width: 78px; }
  input[type=range] { width: 100%; }
  .grid { display: grid; grid-template-columns: 1fr; gap: 12px; }
  .card { background: var(--panel); border: 1px solid var(--line); border-radius: 10px; padding: 10px 12px; }
  .card h2 { font-size: 13px; font-weight: 600; color: var(--muted); margin: 0 0 6px; }
  canvas { width: 100%; display: block; }
  .stats { display: flex; flex-wrap: wrap; gap: 8px 20px; font-variant-numeric: tabular-nums; }
  .stats b { font-weight: 600; }
  .win { color: var(--win); } .breach { color: var(--breach); } .crash { color: var(--crash); }
  .banner { font-weight: 600; min-height: 1.45em; }
  .legend { display: flex; flex-wrap: wrap; gap: 6px 16px; color: var(--muted); font-size: 13px; margin-top: 8px; }
  .dot { display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 5px; vertical-align: -1px; }
</style>
</head>
<body>
<main>
  <h1>3v2 Intercept Replay</h1>
  <p class="sub">Recorded from the Isaac Lab sim (headless). Three defenders protect the goal tag on the left wall from two attackers.</p>

  <div class="controls">
    <label>Controller <select id="ctrl"></select></label>
    <label>Arena <select id="env"></select></label>
    <button id="play">Pause</button>
    <label>Speed <select id="speed"><option>0.5</option><option selected>1</option><option>2</option><option>4</option></select></label>
    <span id="clock" class="stats"></span>
  </div>
  <input id="scrub" type="range" min="0" value="0" aria-label="Time">

  <div class="grid">
    <div class="card">
      <h2>Top view (x into the room, y across)</h2>
      <canvas id="top"></canvas>
      <div class="legend">
        <span><span class="dot" style="background:var(--def)"></span>Defender (ring = tag or contact radius)</span>
        <span><span class="dot" style="background:var(--att)"></span>Attacker</span>
        <span><span class="dot" style="background:var(--down)"></span>Attacker stopped</span>
        <span><span class="dot" style="background:var(--zone-edge)"></span>Goal zone</span>
      </div>
    </div>
    <div class="card">
      <h2>Side view (x into the room, z up)</h2>
      <canvas id="side"></canvas>
    </div>
    <div class="card">
      <div id="banner" class="banner"></div>
      <div id="score" class="stats"></div>
    </div>
  </div>
</main>
<script>
const DATA = __DATA__;
const $ = (id) => document.getElementById(id);
const css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
const names = Object.keys(DATA);
names.forEach((n) => $("ctrl").add(new Option(n, n)));
let rec, frames, t = 0, playing = true, last = null;

function load() {
  rec = DATA[$("ctrl").value];
  const envSel = $("env"), keep = envSel.value;
  envSel.innerHTML = "";
  rec.envs.forEach((_, i) => envSel.add(new Option(String(i), String(i))));
  envSel.value = keep && keep < rec.envs.length ? keep : "0";
  pickEnv();
}
function pickEnv() {
  frames = rec.envs[+$("env").value];
  $("scrub").max = frames.length - 1;
  t = 0;
  score();
}
function score() {
  const c = { win: 0, breach: 0, crash: 0, timeout: 0 };
  frames.forEach((f) => { if (f.end) c[f.end]++; });
  const n = c.win + c.breach + c.crash + c.timeout;
  $("score").innerHTML = `<span>Episodes in this clip: <b>${n}</b></span>` +
    `<span class="win">Wins <b>${c.win}</b></span><span class="breach">Breaches <b>${c.breach}</b></span>` +
    `<span class="crash">Crashes <b>${c.crash}</b></span><span>Timeouts <b>${c.timeout}</b></span>`;
}

function setup(canvas, w, h) {
  const dpr = window.devicePixelRatio || 1, cw = canvas.clientWidth, ch = cw * h / w;
  canvas.width = cw * dpr; canvas.height = ch * dpr; canvas.style.height = ch + "px";
  const ctx = canvas.getContext("2d"); ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { ctx, s: cw / w };
}

// axes: top view maps (x, y) with y flipped so +y (right when facing the wall) points down
function drawView(canvas, ax2, frameIdx) {
  const [[x0, x1], [y0, y1], [z0, z1]] = rec.room;
  const lo2 = ax2 === 1 ? y0 : z0, hi2 = ax2 === 1 ? y1 : z1, pad = 0.4;
  const W = x1 - x0 + 2 * pad, H = hi2 - lo2 + 2 * pad;
  const { ctx, s } = setup(canvas, W, H);
  const P = (x, v) => [(x - x0 + pad) * s, (ax2 === 1 ? (v - lo2 + pad) : (hi2 - v + pad)) * s];
  ctx.clearRect(0, 0, W * s, H * s);
  // room
  let [a, b] = P(x0, ax2 === 1 ? y0 : z1);
  ctx.fillStyle = css("--room"); ctx.fillRect(a, b, (x1 - x0) * s, (hi2 - lo2) * s);
  ctx.strokeStyle = css("--wall"); ctx.lineWidth = 1; ctx.strokeRect(a, b, (x1 - x0) * s, (hi2 - lo2) * s);
  // goal zone
  const gz = rec.goal_zone, zlo = gz[ax2][0], zhi = gz[ax2][1];
  [a, b] = P(gz[0][0], ax2 === 1 ? zlo : zhi);
  ctx.fillStyle = css("--zone"); ctx.fillRect(a, b, (gz[0][1] - gz[0][0]) * s, (zhi - zlo) * s);
  ctx.strokeStyle = css("--zone-edge"); ctx.setLineDash([4, 3]); ctx.strokeRect(a, b, (gz[0][1] - gz[0][0]) * s, (zhi - zlo) * s); ctx.setLineDash([]);
  // goal tag on the wall
  const c = rec.goal_tag_center[ax2], h = rec.goal_tag_size / 2;
  ctx.strokeStyle = css("--tag"); ctx.lineWidth = 5;
  ctx.beginPath(); ctx.moveTo(...P(x0, c - h)); ctx.lineTo(...P(x0, c + h)); ctx.stroke(); ctx.lineWidth = 1;
  // trails (last 1 s, within the current episode)
  const trail = Math.round(1 / rec.dt);
  let start = frameIdx;
  while (start > 0 && frameIdx - start < trail && !frames[start - 1].end) start--;
  const f = frames[frameIdx];
  const drawTrail = (key, k, color) => {
    ctx.strokeStyle = color; ctx.globalAlpha = 0.35; ctx.lineWidth = 2; ctx.beginPath();
    for (let i = start; i <= frameIdx; i++) { const p = frames[i][key][k]; const q = P(p[0], p[ax2]); i === start ? ctx.moveTo(...q) : ctx.lineTo(...q); }
    ctx.stroke(); ctx.globalAlpha = 1; ctx.lineWidth = 1;
  };
  f.d.forEach((_, k) => drawTrail("d", k, css("--def")));
  f.a.forEach((_, k) => { if (f.alive[k]) drawTrail("a", k, css("--att")); });
  // drones
  const r = Math.max(4, 0.12 * s);
  f.d.forEach((p, k) => {
    const q = P(p[0], p[ax2]);
    ctx.strokeStyle = css("--def"); ctx.globalAlpha = 0.35; ctx.beginPath(); ctx.arc(...q, rec.tag_radius * s, 0, 2 * Math.PI); ctx.stroke(); ctx.globalAlpha = 1;
    ctx.fillStyle = css("--def"); ctx.beginPath(); ctx.arc(...q, r, 0, 2 * Math.PI); ctx.fill();
    ctx.fillStyle = css("--panel"); ctx.font = `600 ${Math.round(r * 1.2)}px system-ui`; ctx.textAlign = "center"; ctx.textBaseline = "middle"; ctx.fillText(String(k + 1), ...q);
  });
  f.a.forEach((p, k) => {
    const q = P(p[0], p[ax2]);
    if (f.alive[k]) { ctx.fillStyle = css("--att"); ctx.beginPath(); ctx.arc(...q, r, 0, 2 * Math.PI); ctx.fill(); }
    else { ctx.strokeStyle = css("--down"); ctx.lineWidth = 2.5; ctx.beginPath(); ctx.moveTo(q[0] - r, q[1] - r); ctx.lineTo(q[0] + r, q[1] + r); ctx.moveTo(q[0] + r, q[1] - r); ctx.lineTo(q[0] - r, q[1] + r); ctx.stroke(); ctx.lineWidth = 1; }
  });
  // layered defense zones (top view): 1 danger, 2 combat, 3 outer
  if (ax2 === 1 && rec.layers) {
    const gz = rec.goal_zone, g = P((gz[0][0] + gz[0][1]) / 2, (gz[1][0] + gz[1][1]) / 2);
    [[rec.layers[1], "rgba(240,160,32,0.10)", "#f0a020"], [rec.layers[0], "rgba(229,72,77,0.16)", "#e5484d"]].forEach(([rr, fill, line]) => {
      ctx.beginPath(); ctx.arc(...g, rr * s, 0, 2 * Math.PI); ctx.fillStyle = fill; ctx.fill(); ctx.strokeStyle = line; ctx.lineWidth = 1.5; ctx.stroke(); ctx.lineWidth = 1;
    });
  }
  // name tags (top view): who is who and what each one is doing
  if (ax2 === 1) {
    ctx.font = `600 ${Math.max(10, Math.round(r * 1.05))}px system-ui`; ctx.textAlign = "left"; ctx.textBaseline = "bottom";
    const tag = (p, text, color) => {
      const q = P(p[0], p[1]);
      ctx.lineWidth = 3; ctx.strokeStyle = css("--panel"); ctx.strokeText(text, q[0] + r + 3, q[1] - r);
      ctx.fillStyle = color; ctx.fillText(text, q[0] + r + 3, q[1] - r); ctx.lineWidth = 1;
    };
    f.d.forEach((p, k) => tag(p, `${rec.def_names ? rec.def_names[k] : "D" + (k + 1)}${f.dl ? " \u00b7 " + f.dl[k] : ""}`, css("--def")));
    f.a.forEach((p, k) => tag(p, `${rec.att_names ? rec.att_names[k] : "A" + (k + 1)}${f.al ? " \u00b7 " + f.al[k] : ""}`, css("--att")));
  }
}

function draw() {
  const i = Math.min(Math.floor(t), frames.length - 1);
  drawView($("top"), 1, i);
  drawView($("side"), 2, i);
  $("scrub").value = i;
  $("clock").textContent = `t = ${(i * rec.dt).toFixed(1)} s / ${(frames.length * rec.dt).toFixed(0)} s`;
  // banner: most recent episode end within the last 1.5 s
  let msg = "", cls = "";
  for (let j = i; j >= 0 && i - j < 1.5 / rec.dt; j--) if (frames[j].end) {
    msg = { win: "Defense wins: both attackers stopped", breach: "Breach: an attacker reached the goal zone", crash: "Crash: defender hit a wall or teammate", timeout: "Timeout" }[frames[j].end];
    cls = frames[j].end; break;
  }
  $("banner").textContent = msg; $("banner").className = "banner " + cls;
}
function tick(now) {
  if (last !== null && playing) {
    t += (now - last) / 1000 / rec.dt * +$("speed").value;
    if (t >= frames.length) t = 0;
  }
  last = now; draw(); requestAnimationFrame(tick);
}
$("ctrl").onchange = load;
$("env").onchange = pickEnv;
$("play").onclick = () => { playing = !playing; $("play").textContent = playing ? "Pause" : "Play"; };
$("scrub").oninput = (e) => { t = +e.target.value; };
window.onresize = draw;
load();
requestAnimationFrame(tick);
</script>
</body>
</html>
"""


def arena_for_3d() -> dict:
    """Room and tags from config/arena.json, with each tag texture inlined as a PNG data URI."""
    sys.path.insert(0, REPO)
    import cv2

    from tello.localizer import load_arena, make_tag_image

    arena = load_arena()
    tags = []
    for tag in arena["tags"]:
        img, black_frac = make_tag_image(tag["id"], px=240)
        _, png = cv2.imencode(".png", img)
        tags.append({**tag, "black_frac": black_frac, "png": "data:image/png;base64," + base64.b64encode(png).decode()})
    return {"room": [arena["room"][a] for a in "xyz"], "tags": tags, "obstacles": arena.get("obstacles", []),
            "far_wall": bool(arena.get("far_wall", False))}


ARENA_NAME = os.environ.get("ARENA_JSON", "arena").removesuffix(".json")


def main():
    data = {}
    for path in sorted(glob.glob(os.path.join(REPLAY_DIR, "*.json"))):
        with open(path) as f:
            rec = json.load(f)
        if rec.get("arena", "arena") != ARENA_NAME:
            continue  # each arena map gets its own pages (ARENA_JSON=arena_gate -> replay_arena_gate.html)
        data[rec.get("controller", os.path.splitext(os.path.basename(path))[0])] = rec
    if not data:
        raise SystemExit(f"No recordings in {REPLAY_DIR}; run eval_controllers.py --record_steps first.")
    # show the recommended controller first
    order = ["gate_layers", "gate_40k_three_walls", "gate_mesh", "gate_block", "gate_relay", "gate_kamikaze", "gate_kamikaze_demo_wide_strike", "gate_40k_nets_of_purity", "gate_40k_shield_wall", "team_mesh", "team_block", "team_layers", "team_relay", "team_kamikaze", "team_40k_nets_of_purity", "team_40k_shield_wall", "taped_mesh_relay", "relay_blocking", "taped_mesh", "zone_1v1", "gate_taped_mesh_relay", "gate_block", "block_defense", "ppo_defense", "block_bump", "ppo_bump"]
    data = dict(sorted(data.items(), key=lambda kv: order.index(kv[0]) if kv[0] in order else len(order)))
    suffix = "" if ARENA_NAME == "arena" else "_" + ARENA_NAME
    out = os.path.join(REPLAY_DIR, f"replay{suffix}.html")
    with open(out, "w", encoding="utf-8") as f:
        f.write(PAGE.replace("__DATA__", json.dumps(data, separators=(",", ":"))))
    print(f"Wrote {out} ({', '.join(data)})")

    with open(os.path.join(os.path.dirname(__file__), "replay3d_template.html"), encoding="utf-8") as f:
        page_3d = f.read()
    page_3d = page_3d.replace("__DATA__", json.dumps(data, separators=(",", ":")))
    page_3d = page_3d.replace("__ARENA__", json.dumps(arena_for_3d(), separators=(",", ":")))
    out_3d = os.path.join(REPLAY_DIR, f"replay3d{suffix}.html")
    with open(out_3d, "w", encoding="utf-8") as f:
        f.write(page_3d)
    print(f"Wrote {out_3d}")


if __name__ == "__main__":
    main()
