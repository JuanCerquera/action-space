# Copyright (c) 2025, Action Space hackathon team.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Render 3D animated GIFs of recorded rounds for the README (GitHub cannot embed the interactive replay).

Perspective view of the photographed room with a slowly orbiting camera: panels as blocks, the goal
zone, AprilTags, the defense zones on the floor, drones with floor shadows (so heights read), 1.5 s
trails, hanging mesh strips and each drone's job label.

    python scripts/make_readme_3d.py          # -> docs/img/*_3d.gif
"""

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import animation  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: E402

from make_readme_media import DEF, ATT, DOWN, MESH, OUT, REPO, load, pick_round, short  # noqa: E402


def box_faces(lo, hi):
    x0, y0, z0 = lo
    x1, y1, z1 = hi
    v = np.array([[x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0], [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]])
    return [v[[0, 1, 2, 3]], v[[4, 5, 6, 7]], v[[0, 1, 5, 4]], v[[2, 3, 7, 6]], v[[1, 2, 6, 5]], v[[0, 3, 7, 4]]]


def draw_static(ax, arena, rec):
    (x0, x1), (y0, y1), (z0, z1) = arena["room"]["x"], arena["room"]["y"], arena["room"]["z"]
    x1 = min(x1, x0 + 6.0)
    floor = [[(x0, y0, 0), (x1, y0, 0), (x1, y1, 0), (x0, y1, 0)]]
    ax.add_collection3d(Poly3DCollection(floor, fc="#e9ecf1", ec="#b8bec9", lw=0.6, zorder=0))
    for gx in np.arange(np.ceil(x0), x1, 1.0):
        ax.plot([gx, gx], [y0, y1], [0, 0], color="#d3d7de", lw=0.5)
    for gy in np.arange(np.ceil(y0), y1, 1.0):
        ax.plot([x0, x1], [gy, gy], [0, 0], color="#d3d7de", lw=0.5)
    # the wall behind the goal (the camera looks at it from the attack side; no near wall to block the view)
    ax.add_collection3d(Poly3DCollection([[(x0, y0, 0), (x0, y1, 0), (x0, y1, z1), (x0, y0, z1)]], fc="#dfe3ea", alpha=0.35, ec="#c5cad3", zorder=1))
    gz = arena["goal"]["zone"]
    glo = [gz["x"][0], gz["y"][0], gz["z"][0]]
    ghi = [gz["x"][1], gz["y"][1], gz["z"][1]]
    ax.add_collection3d(Poly3DCollection(box_faces(glo, ghi), fc="#e04545", alpha=0.18, ec="#c02020", lw=0.6, zorder=2))
    ax.text(ghi[0] + 0.1, ghi[1] + 0.05, ghi[2] + 0.1, "GOAL", color="#c02020", fontsize=7, weight="bold")
    for o in arena.get("obstacles", []):
        c, s = np.array(o["center"]), np.array(o["size"])
        ax.add_collection3d(Poly3DCollection(box_faces(c - s / 2, c + s / 2), fc="#e6dfcc", ec="#8a816b", lw=0.6, alpha=0.95, zorder=2))
    for t in arena["tags"]:
        p = t["center"]
        ax.scatter([p[0]], [p[1]], [p[2]], marker="s", s=10, color="#111", depthshade=False)
    if rec.get("layers"):
        gc = np.array([(gz["x"][0] + gz["x"][1]) / 2, (gz["y"][0] + gz["y"][1]) / 2])
        th = np.linspace(-np.pi / 2, np.pi / 2, 60)
        for r, col in ((rec["layers"][0], "#e5484d"), (rec["layers"][1], "#f0a020")):
            xs, ys = gc[0] + r * np.cos(th), gc[1] + r * np.sin(th)
            keep = (ys >= y0) & (ys <= y1) & (xs <= x1)
            ax.plot(xs[keep], ys[keep], np.zeros(keep.sum()) + 0.01, color=col, lw=2.0, zorder=1)
    xv = min(x1, x0 + 6.0)  # frame the defended half, where the action is
    ax.set_xlim(x0, xv)
    ax.set_ylim(y0, y1)
    ax.set_zlim(0, z1)
    ax.set_box_aspect((xv - x0, y1 - y0, z1 - z0), zoom=1.15)
    ax.set_axis_off()


def make_gif3d(clip, title, out_name, arena_path, want=None, seconds=14.0, stride=4, fps=10, azim=(-50, -25)):
    rec = load(os.path.join(REPO, "media", "replay", f"{clip}.json"))
    arena = load(arena_path)
    e, a, b = pick_round(rec, want)
    frames = rec["envs"][e][a:b + 1]
    n_show = int(seconds / (rec["dt"] * stride))
    if want is not None:
        first = next((i for i, f in enumerate(frames) if any(want in x for x in f.get("dl", []) + f.get("al", []))), 0)
        lo = max(0, first - n_show * stride // 2)
        frames = frames[lo:lo + n_show * stride]
    else:
        frames = frames[: n_show * stride]
    idx = list(range(0, len(frames), stride))
    names_d = [short(n) for n in rec.get("def_names") or [f"D{k + 1}" for k in range(len(frames[0]["d"]))]]
    names_a = [short(n) for n in rec.get("att_names") or [f"A{k + 1}" for k in range(len(frames[0]["a"]))]]
    net_len = rec.get("net_length", 0.8)

    fig = plt.figure(figsize=(8.4, 5.0), dpi=85)
    ax = fig.add_subplot(111, projection="3d", computed_zorder=False)  # we set the layering: floor, walls, drones
    fig.subplots_adjust(-0.05, -0.08, 1.05, 1.0)

    def frame(k):
        ax.cla()
        draw_static(ax, arena, rec)
        ax.view_init(elev=30, azim=azim[0] + (azim[1] - azim[0]) * k / max(len(idx) - 1, 1))
        i = idx[k]
        f = frames[i]
        lo = max(0, i - int(1.5 / rec["dt"]))
        for key, color in (("d", DEF), ("a", ATT)):
            for j in range(len(f[key])):
                tr = np.array([g[key][j] for g in frames[lo:i + 1]])
                ax.plot(tr[:, 0], tr[:, 1], tr[:, 2], color=color, alpha=0.45, lw=1.4, zorder=6)
        for key, color, alive in (("d", DEF, f.get("dalive", [True] * 3)), ("a", ATT, f["alive"])):
            for j, p in enumerate(f[key]):
                up = alive[j]
                z = p[2] if up else 0.03
                ax.plot([p[0], p[0]], [p[1], p[1]], [0.0, z], color="#9aa1ad", lw=0.6, alpha=0.6, zorder=5)  # height stalk
                ax.scatter([p[0]], [p[1]], [0.005], s=18, color="#7d838d", alpha=0.35, depthshade=False, zorder=4)  # shadow
                ax.scatter([p[0]], [p[1]], [z], s=90, color=(color if up else DOWN), edgecolors="white", linewidths=0.8,
                           depthshade=False, marker="o" if up else "X", zorder=8)
                if key == "d" and f.get("net") and f["net"][j] and up:
                    ax.plot([p[0], p[0]], [p[1], p[1]], [z - 0.08, max(z - net_len, 0.02)], color=MESH, lw=5.0, alpha=0.9, zorder=7)
                if key == "a" and f.get("netted") and f["netted"][j]:
                    ax.scatter([p[0]], [p[1]], [0.04], s=90, marker="s", facecolors="none", edgecolors=MESH, linewidths=2, zorder=7)
                label = f"{(names_d if key == 'd' else names_a)[j]} {short((f.get('dl') if key == 'd' else f.get('al'))[j]) if f.get('dl') else ''}"
                ax.text(p[0], p[1], z + 0.25, label, fontsize=7.5, color=(color if up else DOWN), weight="bold" if key == "d" else None, zorder=9)
        t = i * rec["dt"]
        end = next((g["end"] for g in frames[max(0, i - 20):i + 1] if g.get("end")), None)
        ax.text2D(0.03, 0.93, f"{title}   t = {t:4.1f} s" + (f"   {end.upper()}" if end else ""), transform=ax.transAxes, fontsize=11)

    anim = animation.FuncAnimation(fig, frame, frames=len(idx), interval=1000 / fps)
    path = os.path.join(OUT, out_name)
    anim.save(path, writer=animation.PillowWriter(fps=fps))
    plt.close(fig)
    print(f"{path}  {os.path.getsize(path) / 1e6:.1f} MB  ({len(idx)} frames)")


def main():
    gate = os.path.join(REPO, "config", "arena_gate.json")
    os.makedirs(OUT, exist_ok=True)
    make_gif3d("gate_mesh", "3D: taped mesh + relay", "mesh_3d.gif", gate, want="caught in mesh")
    make_gif3d("gate_block", "3D: tuned blocking", "block_3d.gif", gate, want="push")
    make_gif3d("gate_kamikaze", "3D: kamikaze keeper", "kamikaze_3d.gif", gate, want="KAMIKAZE")
    make_gif3d("gate_layers", "3D: three zones", "layers_3d.gif", gate, want="holding layer 2")


if __name__ == "__main__":
    main()
