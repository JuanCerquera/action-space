# Copyright (c) 2025, Action Space hackathon team.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""One overview video of every defense strategy, each with its signature move highlighted.

For each strategy: a title card (name, win rate, one line on how it works), then a recorded round in top
view and 3D side by side. Whenever a drone performs the strategy's signature move (a mesh catch, a push,
a ram, a reinforcement, ...), it gets an orange ring in both views and the move is named on screen.

    python scripts/make_strategy_video.py            # -> media/videos/strategies_overview.mp4 (every strategy)
    python scripts/make_strategy_video.py --minute   # -> media/videos/strategies_1min.mp4 (submission cut)
"""

import json
import os

import matplotlib

matplotlib.use("Agg")
import imageio_ffmpeg  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import patches  # noqa: E402

from make_readme_3d import draw_static  # noqa: E402
from make_readme_media import ATT, DEF, DOWN, MESH, REPO, draw_room, load, short  # noqa: E402

FPS, STRIDE, SECONDS = 10, 3, 9.0
W, H = 1280, 720
HL = "#ff7a00"  # highlight orange
ZONE_BOX = [[0.6, 3.0], [-2.0, 2.0], [0.6, 2.4]]

# clip, name, win rate, how it works, signature-move label fragments, banner, extra drawing
STRATEGIES = [
    ("gate_mesh", "Taped mesh + relay", "99.0%",
     "A taped mesh strip hangs under each defender. Carriers fly above the attacker's line; an attacker that touches the tape sticks and drops.",
     ("caught in mesh", "dropping mesh"), "MESH CATCH", None),
    ("gate_block", "Tuned blocking", "95.5%",
     "Two markers hold the line between their attacker and the goal and push it away; the third keeps goal.",
     ("pushing",), "PUSH AWAY", None),
    ("gate_layers", "3 zones", "94.9%",
     "Danger / combat / outer zones. Guards wait at the combat-zone edge while attackers are far out.",
     ("holding layer 2",), "HOLD AT ZONE 2", None),
    ("gate_layers_demo_big_danger", "3 zones: reinforcement", "demo",
     "An attacker in the danger zone brings the reserve out: two defenders push it back to zone 2.",
     ("reinforcements",), "REINFORCE", None),
    ("gate_relay", "Relay blocking", "94.0%",
     "Markers meet the attacker's predicted path. A beaten marker hands its attacker over and retreats to goal.",
     ("beaten",), "HANDOVER + RETREAT", None),
    ("gate_kamikaze", "Kamikaze keeper", "94.6%",
     "The defender nearest the goal is the keeper; when an attacker comes close, it rams it at full speed.",
     ("KAMIKAZE",), "KAMIKAZE RAM", None),
    ("gate_kamikaze_relay", "Kamikaze + relay", "91.5%",
     "Relay marking with the kamikaze keeper as the last line.",
     ("KAMIKAZE", "beaten"), "RAM / HANDOVER", None),
    ("gate_zone", "Zone 1v1 (defense box)", "90.7%",
     "Defenders never leave the box in front of the goal; each marker holds its attacker's line at the box edge.",
     ("blocking",), "HOLD THE BOX EDGE", "box"),
    ("gate_40k_nets_of_purity", "Warhammer 40k: Nets of Purity", "99.0%",
     "Same mesh strategy with themed labels: Guardsmen (G) against Heretics (H).",
     ("NET", "ENSNARED", "casting", "ensnared"), "FOR THE EMPEROR", None),
]


def segment(rec, keys, seconds):
    """Frames of the env/round with the most signature moves, centered on the first one."""
    best = None
    for e, frames in enumerate(rec["envs"]):
        hits = [i for i, f in enumerate(frames) if any(k in x for k in keys for x in f.get("dl", []) + f.get("al", []))]
        if hits and (best is None or len(hits) > best[2]):
            best = (e, hits[0], len(hits))
    e, first = (best[0], best[1]) if best else (0, 0)
    frames = rec["envs"][e]
    n = int(seconds / (rec["dt"] * STRIDE)) * STRIDE
    lo = max(0, min(first - n // 3, len(frames) - n))
    return frames[lo:lo + n]


def active(f, keys):
    """(defender indices, attacker indices) doing the signature move in this frame."""
    d = [j for j, x in enumerate(f.get("dl", [])) if any(k in x for k in keys)]
    a = [j for j, x in enumerate(f.get("al", [])) if any(k in x for k in keys)]
    return d, a


def draw_box_top(ax):
    (x0, x1), (y0, y1), _ = ZONE_BOX
    ax.add_patch(patches.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec=HL, lw=1.8, ls="--"))
    ax.text(x1 - 0.1, y1 - 0.25, "DEFENSE BOX", color=HL, fontsize=8, ha="right", weight="bold")


def draw_box_3d(ax):
    (x0, x1), (y0, y1), (z0, z1) = ZONE_BOX
    for z in (z0, z1):
        ax.plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0], [z] * 5, color=HL, lw=1.0, ls="--", zorder=3)
    for x, y in ((x0, y0), (x1, y0), (x1, y1), (x0, y1)):
        ax.plot([x, x], [y, y], [z0, z1], color=HL, lw=1.0, ls="--", zorder=3)


def strategy_frames(fig, writer, arena, spec):
    clip, name, win, how, keys, banner, extra = spec
    rec = load(os.path.join(REPO, "media", "replay", f"{clip}.json"))
    frames = segment(rec, keys, SECONDS)
    names_d = [short(n) for n in rec.get("def_names") or ["D1", "D2", "D3"]]
    names_a = [short(n) for n in rec.get("att_names") or ["A1", "A2"]]
    idx = list(range(0, len(frames), STRIDE))
    count, seen = 0, False
    for k, i in enumerate(idx):
        f = frames[i]
        act_d, act_a = active(f, keys)
        if (act_d or act_a) and not seen:
            count += 1
        seen = bool(act_d or act_a)
        fig.clf()
        fig.patch.set_facecolor("#f4f6f9")
        fig.text(0.02, 0.945, name, fontsize=22, weight="bold", color="#1b2230")
        fig.text(0.02, 0.905, how, fontsize=10.5, color="#4a5263")
        fig.text(0.98, 0.945, f"{win} wins" if win != "demo" else "demo", fontsize=20, weight="bold",
                 color="#2f9e44", ha="right")
        ax2 = fig.add_axes([0.01, 0.06, 0.45, 0.8])
        ax3 = fig.add_axes([0.45, 0.02, 0.56, 0.88], projection="3d", computed_zorder=False)
        draw_room(ax2, arena, rec)
        draw_static(ax3, arena, rec)
        ax3.view_init(elev=30, azim=-50 + 22 * k / max(len(idx) - 1, 1))
        if extra == "box":
            draw_box_top(ax2)
            draw_box_3d(ax3)
        lo = max(0, i - int(1.5 / rec["dt"]))
        for key, color in (("d", DEF), ("a", ATT)):
            for j in range(len(f[key])):
                tr = np.array([g[key][j] for g in frames[lo:i + 1]])
                ax2.plot(tr[:, 0], tr[:, 1], color=color, alpha=0.35, lw=1.5)
                ax3.plot(tr[:, 0], tr[:, 1], tr[:, 2], color=color, alpha=0.45, lw=1.4, zorder=6)
        pulse = 1.0 + 0.35 * np.sin(k * 1.3)
        for key, color, alive, names, labels, act in (("d", DEF, f.get("dalive", [True] * 3), names_d, f.get("dl"), act_d),
                                                     ("a", ATT, f["alive"], names_a, f.get("al"), act_a)):
            for j, p in enumerate(f[key]):
                up = alive[j]
                z = p[2] if up else 0.03
                c = color if up else DOWN
                hot = j in act
                ax2.plot(p[0], p[1], "o" if up else "x", ms=10, color=c, mec="white", mew=1.2, zorder=5)
                ax3.plot([p[0], p[0]], [p[1], p[1]], [0.0, z], color="#9aa1ad", lw=0.6, alpha=0.6, zorder=5)
                ax3.scatter([p[0]], [p[1]], [0.005], s=18, color="#7d838d", alpha=0.35, depthshade=False, zorder=4)
                ax3.scatter([p[0]], [p[1]], [z], s=95, color=c, edgecolors="white", linewidths=0.8, depthshade=False,
                            marker="o" if up else "X", zorder=8)
                if hot:  # the signature move: a pulsing orange ring in both views
                    ax2.plot(p[0], p[1], "o", ms=24 * pulse, mfc="none", mec=HL, mew=2.5, zorder=6)
                    ax3.scatter([p[0]], [p[1]], [z], s=520 * pulse, facecolors="none", edgecolors=HL, linewidths=2.5,
                                depthshade=False, zorder=9)
                if key == "d" and f.get("net") and f["net"][j] and up:
                    ax2.plot(p[0], p[1], "o", ms=15, mfc="none", mec=MESH, mew=2, zorder=5)
                    ax3.plot([p[0], p[0]], [p[1], p[1]], [z - 0.08, max(z - rec.get("net_length", 0.8), 0.02)], color=MESH,
                             lw=5, alpha=0.9, zorder=7)
                if key == "a" and f.get("netted") and f["netted"][j]:
                    ax2.plot(p[0], p[1], "s", ms=14, mfc="none", mec=MESH, mew=2.2, zorder=6)
                txt = f"{names[j]} {short(labels[j]) if labels else ''}"
                ax2.text(p[0] + 0.12, p[1] + (0.12 if key == "d" else -0.32), txt, fontsize=7.5, color=HL if hot else c,
                         weight="bold" if (hot or key == "d") else None, zorder=7)
                ax3.text(p[0], p[1], z + 0.25, txt, fontsize=7.5, color=HL if hot else c, weight="bold" if hot else None, zorder=10)
        t = i * rec["dt"]
        fig.text(0.02, 0.025, f"top view  |  t = {t:4.1f} s  |  blue = our defenders, red = attackers, gray = down, "
                              "yellow = mesh", fontsize=9, color="#4a5263")
        if seen:
            fig.text(0.72, 0.8, f"{banner}!", fontsize=22, weight="bold", color="white", ha="center",
                     bbox=dict(boxstyle="round,pad=0.35", fc=HL, ec="none"))
        if count:
            fig.text(0.98, 0.025, f"{banner.lower()} x{count}", fontsize=10, color=HL, ha="right", weight="bold")
        writer.send(canvas_rgb(fig))


def canvas_rgb(fig):
    fig.canvas.draw()
    return np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()


def card(fig, writer, title, lines, seconds, image=None):
    for _ in range(int(seconds * FPS)):
        fig.clf()
        fig.patch.set_facecolor("#11151c")
        fig.text(0.5, 0.78 if image is None else 0.9, title, fontsize=30, weight="bold", color="white", ha="center")
        for n, line in enumerate(lines):
            fig.text(0.5, (0.64 if image is None else 0.83) - n * 0.065, line, fontsize=15, color="#c9d2de", ha="center")
        if image is not None:
            ax = fig.add_axes([0.12, 0.04, 0.76, 0.66])
            ax.imshow(image)
            ax.axis("off")
        writer.send(canvas_rgb(fig))


def main():
    import sys

    global SECONDS
    short_cut = "--minute" in sys.argv  # the 1-minute submission cut: main strategies, shorter cards
    arena = load(os.path.join(REPO, "config", "arena_gate.json"))
    out = os.path.join(REPO, "media", "videos", "strategies_1min.mp4" if short_cut else "strategies_overview.mp4")
    specs = [s for s in STRATEGIES if not short_cut or s[0] not in ("gate_kamikaze_relay", "gate_40k_nets_of_purity")]
    intro_s, card_s = (3.0, 1.0) if short_cut else (4.0, 1.6)
    if short_cut:
        SECONDS = 7.0
    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
    writer = imageio_ffmpeg.write_frames(out, (W, H), fps=FPS, codec="libx264", pix_fmt_out="yuv420p",
                                         output_params=["-crf", "23"], macro_block_size=16)
    writer.send(None)
    card(fig, writer, "Action Space: 3v2 counter-drone defense",
         ["3 Tello defenders keep 2 attacker Tellos out of the gap between the panels.",
          "Simulated in the photographed hackathon room at real Tello speed.",
          "Each strategy's signature move is highlighted in orange."], intro_s)
    chart = plt.imread(os.path.join(REPO, "docs", "img", "results.png"))
    card(fig, writer, "Win rate, 2,000 identical rounds each", [], intro_s, image=chart)
    for spec in specs:
        card(fig, writer, spec[1], [spec[3][:95] + ("..." if len(spec[3]) > 95 else ""), f"Signature move: {spec[5]}"], card_s)
        strategy_frames(fig, writer, arena, spec)
        print(f"{spec[1]} done", flush=True)
    card(fig, writer, "Fly it", ["Best: taped mesh + relay (99.0%).  Default without a mesh: tuned blocking (95.5%).",
                                 "github.com/JuanCerquera/action-space"], intro_s)
    writer.close()
    plt.close(fig)
    print(f"{out}  {os.path.getsize(out) / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
