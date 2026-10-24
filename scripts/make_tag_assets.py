#!/usr/bin/env python3
"""Generate render assets for render_blender.py: tag36h11 PNGs (with 1-cell white margin) and
tag poses converted from the MAP frame to the WORLD frame (world = P @ map + [0,0,1.22]).

    uv run --no-project --with opencv-python-headless --with pyyaml --with numpy \
        python scripts/make_tag_assets.py
"""
import json
import os

import cv2
import numpy as np
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "assets", "render")
P = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], float)
TAG10_H = 1.22


def main():
    m = yaml.safe_load(open(os.path.join(ROOT, "config", "judges_map.yaml")))
    sizes = yaml.safe_load(open(os.path.join(ROOT, "config", "judges_tags.yaml")))["sizes_mm"]
    os.makedirs(os.path.join(OUT, "tags"), exist_ok=True)
    d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    out = {}
    for k, v in m["tags"].items():
        k = int(k)
        img = cv2.aruco.generateImageMarker(d, k, 8)  # 8x8 cells: 6x6 data + black border, 1 px/cell
        img = cv2.copyMakeBorder(img, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=255)  # white margin
        img = cv2.resize(img, (400, 400), interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(os.path.join(OUT, "tags", f"tag36h11_{k}.png"), img)
        t = P @ np.array(v["t"], float) + [0, 0, TAG10_H]
        R = P @ np.array(v["R"], float)
        out[k] = {"t": t.tolist(), "R": R.tolist(), "size_black_m": float(sizes.get(k, 132.08)) / 1000.0}
    json.dump(out, open(os.path.join(OUT, "tags_world.json"), "w"), indent=1)
    print(f"wrote {len(out)} tags to {OUT}")


if __name__ == "__main__":
    main()
