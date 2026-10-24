#!/usr/bin/env python3
"""
Convert the organizers' tag map (Action_Space drone_deployment_infra/config/map.yaml)
into tag_map.json for localization.py.

  python convert_map.py --tag10-height 1.20            # MEASURE tag 10's centre height first

map.yaml frame : origin = centre of tag 10, x right (looking at the tag), y up, z out of the face
tag_map.json   : origin = floor point under tag 10's centre, x out of the wall into the room,
                 y left (back to the wall), z up        ->  world = (z, x, y + tag10_height)
Tag sizes (black square edge) from the organizers' config/tags.yaml.
"""
import argparse
import json

import numpy as np
import yaml

# organizers' tags.yaml: tags 1-9 and 11 are 132.08 mm, tag 10 is 571.5 mm
SIZES_M = {10: 0.5715}
DEFAULT_SIZE_M = 0.13208

# columns: where the map.yaml axes (x right, y up, z out) point in our world frame
M = np.array([[0, 0, 1],
              [1, 0, 0],
              [0, 1, 0]], float)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--map", default="organizers_map.yaml")
    ap.add_argument("--tag10-height", type=float, required=True, help="tag 10 centre above the floor, m")
    ap.add_argument("--target", type=int, default=6, help="tag on the site we defend")
    ap.add_argument("--out", default="tag_map.json")
    a = ap.parse_args()

    with open(a.map) as f:
        m = yaml.safe_load(f)
    tags = []
    for tid, d in sorted(m["tags"].items(), key=lambda kv: int(kv[0])):
        R, t = np.array(d["R"], float), np.array(d["t"], float)
        c = M @ t + np.array([0, 0, a.tag10_height])
        tags.append({"id": int(tid), "size": SIZES_M.get(int(tid), DEFAULT_SIZE_M),
                     "center": np.round(c, 4).tolist(),
                     "normal": np.round(M @ R[:, 2], 4).tolist(),
                     "up": np.round(M @ R[:, 1], 4).tolist()})
    out = {"family": m.get("family", "tag36h11"),
           "_source": f"converted from {a.map} with tag 10 centre at {a.tag10_height} m above the floor",
           "target": {"id": a.target},
           "tags": tags}
    with open(a.out, "w") as f:
        json.dump(out, f, indent=2)
    for t in tags:
        print(f"tag {t['id']:>2}: centre {t['center']}  faces {t['normal']}  size {t['size'] * 1000:.1f} mm")
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
