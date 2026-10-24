# Copyright (c) 2025, Action Space hackathon team.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Train a YOLOv8 drone detector on the synthetic, auto-labeled RealSense-twin frames.

Frames come from ``scripts/vision_loop.py --export_yolo N`` (media/yolo_dataset/images + labels).
Every 7th frame is held out for validation. The best weights land in media/yolo/drone/weights/best.pt.

.. code-block:: bash

    python scripts/train_yolo.py --epochs 20
    python scripts/vision_loop.py --headless --rounds 5 --yolo media/yolo/drone/weights/best.pt

For the real arena, add labeled RealSense color frames of the real Tellos to the same folders and retrain:
synthetic frames alone teach the shape, real frames teach the real lighting and background.
"""

import argparse
import glob
import os
import shutil

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA = os.path.join(REPO, "media", "yolo_dataset")


def split(val_every: int = 7) -> str:
    """Move every n-th frame into val/ and write data.yaml; returns its path."""
    for sub in ("train", "val"):
        for kind in ("images", "labels"):
            os.makedirs(os.path.join(DATA, sub, kind), exist_ok=True)
    images = sorted(glob.glob(os.path.join(DATA, "images", "*.jpg")))
    for i, img in enumerate(images):
        sub = "val" if i % val_every == 0 else "train"
        name = os.path.splitext(os.path.basename(img))[0]
        shutil.move(img, os.path.join(DATA, sub, "images", name + ".jpg"))
        lab = os.path.join(DATA, "labels", name + ".txt")
        if os.path.exists(lab):
            shutil.move(lab, os.path.join(DATA, sub, "labels", name + ".txt"))
    n_train = len(glob.glob(os.path.join(DATA, "train", "images", "*.jpg")))
    n_val = len(glob.glob(os.path.join(DATA, "val", "images", "*.jpg")))
    print(f"[yolo] {n_train} train / {n_val} val frames")
    path = os.path.join(DATA, "data.yaml")
    with open(path, "w") as f:
        f.write(f"path: {DATA}\ntrain: train/images\nval: val/images\nnames:\n  0: drone\n")
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--imgsz", type=int, default=864, help="Keep near the 848 px frame width: drones are small.")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--model", default="yolov8n.pt", help="Pretrained start point (downloaded on first use).")
    args = ap.parse_args()

    from ultralytics import YOLO

    data = split()
    model = YOLO(args.model)
    model.train(data=data, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch, workers=2,
                project=os.path.join(REPO, "media", "yolo"), name="drone", exist_ok=True, plots=False, verbose=False)
    metrics = model.val(data=data, imgsz=args.imgsz, batch=args.batch, plots=False, verbose=False)
    print(f"[yolo] val mAP50 {metrics.box.map50:.3f}  mAP50-95 {metrics.box.map:.3f}  "
          f"precision {metrics.box.mp:.3f}  recall {metrics.box.mr:.3f}")
    print(f"[yolo] weights: {os.path.join(REPO, 'media', 'yolo', 'drone', 'weights', 'best.pt')}")


if __name__ == "__main__":
    main()
