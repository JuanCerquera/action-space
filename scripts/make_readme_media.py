# Copyright (c) 2025, Action Space hackathon team.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Render README media from recorded clips: top-down GIFs of each strategy, a zone diagram, a results chart.

Reads media/replay/<clip>.json (scripts/fast_sim.py record) and the room from config/<arena>.json.
Writes docs/img/*.gif and *.png (small enough for GitHub).

    python scripts/make_readme_media.py
"""

import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import animation, patches  # noqa: E402

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.join(REPO, "docs", "img")
DEF, ATT, DOWN, MESH = "#2f6fe8", "#e0362f", "#8b9099", "#f2c418"


def load(path):
    with open(path) as f:
        return json.load(f)


def short(job: str) -> str:
    """Compact job text for the small GIF labels."""
    for a, b in (("blocking ", "block "), ("pushing ", "push "), (" away", ""), ("goalkeeper", "goalie"),
                 ("beaten by ", "beaten by "), (", looping back", " -> loop"), ("beaten, back to goal", "retreat"),
                 ("dropping mesh on ", "MESH DROP "), (" from above", " (above)"), ("KAMIKAZE on ", "KAMIKAZE "),
                 ("reinforcing vs ", "REINFORCE "), (" (danger zone)", ""), ("holding layer 2 vs ", "hold L2 "),
                 ("last line (outnumbered)", "last line"), ("attacking the goal", "attack"),
                 ("maneuvering", "maneuver"), ("hovering (bait)", "bait"), ("bumped back", "BUMPED"),
                 ("caught in mesh", "CAUGHT"), ("knocked down", "DOWN"),
                 # Warhammer 40k labels
                 ("FOR THE EMPEROR! martyr strike on ", "FOR THE EMPEROR! "), ("reinforcements! purging ", "REINFORCE "),
                 (" at the inner sanctum", ""), ("manning the second wall against ", "2nd wall vs "),
                 ("holding the line against ", "line vs "), ("bayonet charge, driving back ", "BAYONET "),
                 ("casting the Net of Purity on ", "NET "), ("falling back to the Aegis line", "fall back"),
                 ("outflanked by ", "outflanked by "), (", regrouping", ""), ("Castellan of the Shrine", "Castellan"),
                 ("last stand before the Shrine", "last stand"), ("fallen, martyred for the Emperor", "martyred"),
                 ("assaulting the Shrine", "assault"), ("flanking through the ruins", "flank"),
                 ("luring the faithful (bait)", "bait"), ("repelled by the Guard", "REPELLED"),
                 ("ensnared by the Net of Purity", "ENSNARED"), ("Heretic ", "H"), ("Guardsman ", "G")):
        job = job.replace(a, b)
    return job


def pick_round(rec, want=None):
    """(env index, first frame, last frame) of a round; prefer one whose labels contain ``want``."""
    best = None
    for e, frames in enumerate(rec["envs"]):
        start = 0
        for i, f in enumerate(frames):
            if f.get("end"):
                seg = (e, start, i)
                hit = want is None or any(want in x for g in frames[start:i + 1] for x in g.get("dl", []) + g.get("al", []))
                if hit and (best is None or (seg[2] - seg[1]) > (best[2] - best[1]) * 0.6):
                    best = seg
                    if want is not None:
                        return best
                start = i + 1
    return best or (0, 0, min(len(rec["envs"][0]) - 1, 600))


def draw_room(ax, arena, rec):
    room = arena["room"]
    (x0, x1), (y0, y1) = room["x"], room["y"]
    floor = patches.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=True, fc="#eef0f3", ec="#9aa1ad", lw=1.2)
    ax.add_patch(floor)
    for gx in np.arange(np.ceil(x0), x1, 1.0):
        ax.plot([gx, gx], [y0, y1], color="#dde0e5", lw=0.6, zorder=0)
    for gy in np.arange(np.ceil(y0), y1, 1.0):
        ax.plot([x0, x1], [gy, gy], color="#dde0e5", lw=0.6, zorder=0)
    gz = arena["goal"]["zone"]
    gc = np.array([(gz["x"][0] + gz["x"][1]) / 2, (gz["y"][0] + gz["y"][1]) / 2])
    if rec.get("layers"):
        r1, r2 = rec["layers"]
        for r, col, a in ((r2, "#f0a020", 0.16), (r1, "#e5484d", 0.22)):
            w = ax.add_patch(patches.Wedge(gc, r, -90, 90, fc=col, alpha=a, ec=col, lw=1.0))
            w.set_clip_path(floor)  # zones stop at the walls
        ax.text(gc[0] + r1 * 0.55, gc[1] + 0.1, "1 DANGER", color="#b3262b", fontsize=7, ha="center", weight="bold")
        ax.text(gc[0] + (r1 + r2) / 2, gc[1] + 0.1, "2 COMBAT", color="#a86a00", fontsize=7, ha="center", weight="bold")
        ax.text(min(gc[0] + r2 + 1.0, x1 - 0.6), gc[1] + 0.1, "3 OUTER", color="#2b7a3d", fontsize=7, ha="center", weight="bold")
    ax.add_patch(patches.Rectangle((gz["x"][0], gz["y"][0]), gz["x"][1] - gz["x"][0], gz["y"][1] - gz["y"][0],
                                   fc="#e04545", alpha=0.35, ec="#c02020", lw=1.2))
    ax.text(gz["x"][1] + 0.08, gz["y"][0] - 0.25, "GOAL", color="#c02020", fontsize=7, weight="bold")
    for o in arena.get("obstacles", []):
        c, s = o["center"], o["size"]
        ax.add_patch(patches.Rectangle((c[0] - s[0] / 2, c[1] - s[1] / 2), max(s[0], 0.08), s[1], fc="#c9c2b2", ec="#7a725f"))
    for t in arena["tags"]:
        ax.plot(t["center"][0], t["center"][1], marker="s", ms=3, color="#222")
    ax.set_xlim(x0 - 0.2, x1 + 0.2)
    ax.set_ylim(y0 - 0.2, y1 + 0.2)
    ax.set_aspect("equal")
    ax.axis("off")


def make_gif(clip, title, out_name, arena_path, want=None, seconds=18.0, stride=4, fps=10):
    rec = load(os.path.join(REPO, "media", "replay", f"{clip}.json"))
    arena = load(arena_path)
    e, a, b = pick_round(rec, want)
    frames = rec["envs"][e][a:b + 1]
    n_show = int(seconds / (rec["dt"] * stride))
    if want is not None:  # center the window on the first labelled moment
        first = next((i for i, f in enumerate(frames) if any(want in x for x in f.get("dl", []) + f.get("al", []))), 0)
        lo = max(0, first - n_show * stride // 2)
        frames = frames[lo:lo + n_show * stride]
    else:
        frames = frames[: n_show * stride]
    idx = list(range(0, len(frames), stride))
    names_d = [short(n) for n in rec.get("def_names") or [f"D{k + 1}" for k in range(len(frames[0]["d"]))]]
    names_a = [short(n) for n in rec.get("att_names") or [f"A{k + 1}" for k in range(len(frames[0]["a"]))]]

    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=80)
    fig.subplots_adjust(0.01, 0.01, 0.99, 0.9)

    def frame(k):
        ax.clear()
        draw_room(ax, arena, rec)
        i = idx[k]
        f = frames[i]
        lo = max(0, i - int(1.5 / rec["dt"]))
        for key, color, alive_key in (("d", DEF, "dalive"), ("a", ATT, "alive")):
            for j in range(len(f[key])):
                tr = np.array([g[key][j] for g in frames[lo:i + 1]])
                ax.plot(tr[:, 0], tr[:, 1], color=color, alpha=0.35, lw=1.5)
        for j, p in enumerate(f["d"]):
            up = f.get("dalive", [True] * 3)[j]
            ax.plot(p[0], p[1], "o", ms=9, color=DEF if up else DOWN, mec="white", mew=1.2)
            if f.get("net") and f["net"][j] and up:
                ax.plot(p[0], p[1], "o", ms=14, mfc="none", mec=MESH, mew=2)
            job = short(f["dl"][j]) if f.get("dl") else ""
            ax.text(p[0] + 0.12, p[1] + 0.12, f"{names_d[j]} {job}", fontsize=6.5, color=DEF if up else DOWN, weight="bold")
        for j, p in enumerate(f["a"]):
            up = f["alive"][j]
            ax.plot(p[0], p[1], "o" if up else "x", ms=9 if up else 8, color=ATT if up else DOWN, mec="white" if up else None, mew=1.2)
            if f.get("netted") and f["netted"][j]:
                ax.plot(p[0], p[1], "s", ms=12, mfc="none", mec=MESH, mew=2)
            st = short(f["al"][j]) if f.get("al") else ""
            ax.text(p[0] + 0.12, p[1] - 0.3, f"{names_a[j]} {st}", fontsize=6.5, color=ATT if up else DOWN)
        t = i * rec["dt"]
        end = next((g["end"] for g in frames[max(0, i - 20):i + 1] if g.get("end")), None)
        ax.set_title(f"{title}   t = {t:4.1f} s" + (f"   {end.upper()}" if end else ""), fontsize=10, loc="left")

    anim = animation.FuncAnimation(fig, frame, frames=len(idx), interval=1000 / fps)
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, out_name)
    anim.save(path, writer=animation.PillowWriter(fps=fps))
    plt.close(fig)
    print(f"{path}  {os.path.getsize(path) / 1e6:.1f} MB  ({len(idx)} frames, env {e})")


def zones_png(arena_path, layers):
    arena = load(arena_path)
    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=90)
    fig.subplots_adjust(0.01, 0.01, 0.99, 0.92)
    draw_room(ax, arena, {"layers": layers})
    st = np.array(arena["defender_starts"])
    ax.plot(st[:, 0], st[:, 1], "x", color="#e8590c", ms=9, mew=2.5, label="defender takeoff X marks")
    sp = arena["attacker_spawn"]
    ax.add_patch(patches.Rectangle((sp["x"][0], sp["y"][0]), sp["x"][1] - sp["x"][0], sp["y"][1] - sp["y"][0],
                                   fc=ATT, alpha=0.12, ec=ATT, ls="--", label="attacker start area"))
    ax.legend(loc="lower right", fontsize=7, frameon=True)
    ax.set_title(f"Photographed room, top view: goal = gap between the panels; zones {layers[0]:.2f} m / {layers[1]:.2f} m",
                 fontsize=9, loc="left")
    path = os.path.join(OUT, "zones.png")
    fig.savefig(path)
    plt.close(fig)
    print(path)


def results_png(rows):
    fig, ax = plt.subplots(figsize=(7.2, 3.0), dpi=90)
    fig.subplots_adjust(0.3, 0.17, 0.97, 0.88)
    names = [r[0] for r in rows][::-1]
    wins = [r[1] for r in rows][::-1]
    colors = ["#2f9e44" if w == max(wins) else "#2f6fe8" for w in wins]
    ax.barh(names, wins, color=colors)
    for y, w in enumerate(wins):
        ax.text(w + 0.4, y, f"{w:.1f}%", va="center", fontsize=8)
    ax.set_xlim(80, 100)
    ax.set_xlabel("win rate (%), 2,000 identical rounds, real Tello speed")
    ax.set_title("Strategies in the photographed room", fontsize=10, loc="left")
    path = os.path.join(OUT, "results.png")
    fig.savefig(path)
    plt.close(fig)
    print(path)


def main():
    gate = os.path.join(REPO, "config", "arena_gate.json")
    results_png([("Taped mesh + relay", 99.0), ("Tuned blocking", 95.5), ("3 zones", 94.9), ("Kamikaze keeper", 94.6),
                 ("Relay blocking", 94.0), ("Kamikaze + relay", 91.5), ("Zone 1v1 (defense box)", 90.7)])
    zones_png(gate, load(os.path.join(REPO, "config", "gate_layers.json"))["settings"]["layers"])
    make_gif("gate_block", "Tuned blocking", "block.gif", gate, want="push")
    make_gif("gate_mesh", "Taped mesh + relay", "mesh.gif", gate, want="CAUGHT".lower().replace("caught", "caught in mesh"))
    make_gif("gate_layers", "3 zones", "layers.gif", gate, want="holding layer 2")
    make_gif("gate_layers_demo_big_danger", "3 zones: reinforcement (demo, 2.5 m danger zone)", "layers_reinforce.gif", gate,
             want="reinforcements", seconds=12)
    make_gif("gate_relay", "Relay blocking", "relay.gif", gate, want="beaten")
    make_gif("gate_kamikaze", "Kamikaze keeper", "kamikaze.gif", gate, want="KAMIKAZE")
    make_gif("gate_40k_shield_wall", "For the Emperor: Shield Wall (40k labels)", "40k.gif", gate, want="bayonet", seconds=12)


if __name__ == "__main__":
    sys.exit(main())
