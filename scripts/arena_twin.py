# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Digital twin of the hackathon arena for testing AprilTag localization in Isaac Sim.

Builds the room, goal zone and every AprilTag from config/arena.json, flies a Tello-like camera
(960x720, Tello intrinsics) through a set of test poses, renders each frame, runs the same
localizer the real drones use (tello/localizer.py) and prints ground truth vs estimate.

.. code-block:: bash

    python scripts/arena_twin.py --enable_cameras --headless

"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="AprilTag localization test in a digital twin of the arena.")
parser.add_argument("--frames_dir", type=str, default="media/twin", help="Where to save rendered frames.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import math
import os
import sys

import cv2
import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.sensors import Camera, CameraCfg
from isaaclab.utils.math import quat_from_matrix

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO)
from arena_scene import design_scene  # noqa: E402
from tello.localizer import TELLO_IMAGE_SIZE, TELLO_K, AprilTagLocalizer, load_arena, look_at_R_wc  # noqa: E402


def test_poses(arena: dict) -> list[tuple[np.ndarray, np.ndarray]]:
    """Camera positions and look-at targets spread around the room."""
    goal = np.asarray(next(t for t in arena["tags"] if t["id"] == arena["goal"]["tag_id"])["center"])
    poses = [
        ([2.0, 0.0, 1.2], goal),
        ([3.0, -1.5, 1.0], goal),
        ([3.0, 1.5, 1.4], goal),
        ([5.0, 0.5, 1.5], goal),
        ([6.5, -1.0, 1.3], goal + [0.0, 0.8, 0.0]),
        ([4.2, 0.3, 1.3], [2.0, -0.8, 0.0]),  # tilted down at floor tag 4
        ([1.3, 2.0, 1.5], [0.6, 2.2, 1.5]),  # close to wall tag 5
    ]
    return [(np.asarray(p, dtype=np.float64), np.asarray(t, dtype=np.float64)) for p, t in poses]


def main():
    arena = load_arena()
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.01, device=args_cli.device))
    sim.set_camera_view([7.5, -2.5, 3.0], [2.0, 0.0, 1.0])
    design_scene(arena)

    width, height = TELLO_IMAGE_SIZE
    camera = Camera(
        CameraCfg(
            prim_path="/World/TelloCam",
            update_period=0,
            width=width,
            height=height,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg.from_intrinsic_matrix(
                intrinsic_matrix=TELLO_K.flatten().tolist(), width=width, height=height, clipping_range=(0.05, 50.0)
            ),
        )
    )
    sim.reset()

    # the rendered image is undistorted, so the localizer runs without distortion here
    localizer = AprilTagLocalizer(arena, TELLO_K, dist=None)
    frames_dir = os.path.join(REPO, args_cli.frames_dir)
    os.makedirs(frames_dir, exist_ok=True)

    print(f"{'#':>2} {'gt pos':>22} {'est pos':>22} {'dpos[cm]':>9} {'dyaw[deg]':>9} {'err[px]':>7}  tags")
    for i, (pos, target) in enumerate(test_poses(arena)):
        R_wc = look_at_R_wc(pos, target)  # OpenCV optical frame == Isaac "ros" convention
        quat = quat_from_matrix(torch.tensor(R_wc, dtype=torch.float32)).unsqueeze(0)
        camera.set_world_poses(
            torch.tensor(pos, dtype=torch.float32, device=sim.device).unsqueeze(0),
            quat.to(sim.device),
            convention="ros",
        )
        # render a few frames so textures and lighting settle
        for _ in range(30 if i == 0 else 8):
            sim.step()
        camera.update(dt=sim.get_physics_dt())

        rgb = camera.data.output["rgb"][0, ..., :3].cpu().numpy().astype(np.uint8)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(frames_dir, f"frame_{i:02d}.png"), bgr)

        est = localizer.estimate(bgr)
        gt_yaw = math.atan2(R_wc[1, 2], R_wc[0, 2])
        if est is None:
            print(f"{i:>2} {np.array2string(pos, precision=2):>22} {'no tags detected':>22}")
            continue
        dpos = 100 * np.linalg.norm(est["position"] - pos)
        dyaw = math.degrees(abs((est["yaw"] - gt_yaw + math.pi) % (2 * math.pi) - math.pi))
        print(
            f"{i:>2} {np.array2string(pos, precision=2):>22} {np.array2string(est['position'], precision=2):>22}"
            f" {dpos:>9.2f} {dyaw:>9.2f} {est['reproj_err']:>7.2f}  {est['tag_ids']}"
        )

    # keep the GUI open for inspection
    while simulation_app.is_running() and not args_cli.headless:
        sim.step()


if __name__ == "__main__":
    main()
    simulation_app.close()
