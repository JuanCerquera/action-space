# Copyright (c) 2025, Action Space hackathon team.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Build a light, shareable web version of the 3D replay (docs/web/action_space_3d.html, under 16 MB).

Same viewer as media/replay/replay3d_arena_gate.html (three.js, orbit, name tags, zones, mesh), but
with the main strategies only, two rounds each and every second frame, so it can be hosted as a web page.

    python scripts/make_readme_media.py && python scripts/build_web_replay.py
"""

import json
import os
import re
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("ARENA_JSON", "arena_gate")
from build_replay import arena_for_3d  # noqa: E402

CLIPS = [  # (clip, name shown in the picker)
    ("gate_mesh", "Taped mesh + relay (99.0% wins)"),
    ("gate_block", "Tuned blocking (95.5%)"),
    ("gate_layers", "3 zones (94.9%)"),
    ("gate_kamikaze", "Kamikaze keeper (94.6%)"),
    ("gate_relay", "Relay blocking (94.0%)"),
    ("gate_kamikaze_demo_wide_strike", "Kamikaze demo: wider strike range"),
    ("gate_layers_demo_big_danger", "3 zones demo: reinforcement (40k labels)"),
    ("gate_40k_nets_of_purity", "Nets of Purity (40k labels)"),
]
ENVS, STRIDE = 2, 2


def slim(rec):
    envs = []
    for frames in rec["envs"][:ENVS]:
        out = []
        for i, f in enumerate(frames):
            if i % STRIDE == 0 or f.get("end"):
                g = dict(f)
                g["d"] = [[round(v, 2) for v in p] for p in f["d"]]
                g["a"] = [[round(v, 2) for v in p] for p in f["a"]]
                out.append(g)
        envs.append(out)
    return {**rec, "envs": envs, "dt": rec["dt"] * STRIDE}


THEME = """
  :root { --bg: #f4f6f9; --panel: #ffffff; --ink: #1b2230; --muted: #5f687a; --line: #d6dbe4;
          --win: #15803d; --breach: #b91c1c; --crash: #b45309; --stage: #c9d2de; }
  @media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {
    --bg: #11151c; --panel: #1a1f29; --ink: #e5e8ee; --muted: #9aa3b5; --line: #2b3340;
    --win: #4ade80; --breach: #f87171; --crash: #fbbf24; --stage: #2a3240; color-scheme: dark } }
  :root[data-theme="dark"] { --bg: #11151c; --panel: #1a1f29; --ink: #e5e8ee; --muted: #9aa3b5; --line: #2b3340;
    --win: #4ade80; --breach: #f87171; --crash: #fbbf24; --stage: #2a3240; color-scheme: dark }"""


def main():
    data = {}
    for clip, label in CLIPS:
        path = os.path.join(REPO, "media", "replay", f"{clip}.json")
        with open(path) as f:
            data[label] = slim(json.load(f))
    with open(os.path.join(REPO, "scripts", "replay3d_template.html"), encoding="utf-8") as f:
        page = f.read()
    # the host wraps the page in its own document: keep only the title, style, content and scripts
    page = re.sub(r"<!doctype html>\s*<html[^>]*>\s*<head>\s*", "", page, flags=re.I)
    page = re.sub(r'<meta charset="utf-8">\s*<meta name="viewport"[^>]*>\s*', "", page)
    page = page.replace("<title>3v2 Arena 3D</title>", "<title>Action Space 3D Replay</title>")
    page = re.sub(r"</head>\s*<body>\s*", "", page)
    page = re.sub(r"</body>\s*</html>\s*$", "", page.strip())
    page = re.sub(r"  :root \{ --bg.*?\n  \}\n", THEME.lstrip("\n") + "\n", page, count=1, flags=re.S)
    page = page.replace("background: #c9d2de; }", "background: var(--stage); }")
    page = page.replace("<h1>3v2 Arena 3D</h1>", "<h1>Action Space: 3v2 counter-drone defense</h1>")
    page = page.replace(
        '<p class="sub">Recorded from the Isaac Lab sim. Blue defenders block and push red attackers away from the goal tag on the wall.</p>',
        '<p class="sub">Simulated rounds in the photographed hackathon room at real Tello speed. Three blue defenders keep two red '
        "attackers out of the gap between the panels. Pick a strategy; every drone's tag says what it is doing.</p>")
    page = page.replace("<label>Controller <select", "<label>Strategy <select").replace("<label>Arena <select", "<label>Round set <select")
    page = page.replace("__DATA__", json.dumps(data, separators=(",", ":")))
    page = page.replace("__ARENA__", json.dumps(arena_for_3d(), separators=(",", ":")))
    out_dir = os.path.join(REPO, "docs", "web")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, "action_space_3d.html")
    with open(out, "w", encoding="utf-8") as f:
        f.write(page)
    print(f"{out}: {os.path.getsize(out) / 1e6:.1f} MB, {len(data)} strategies")


if __name__ == "__main__":
    main()
