# Copyright (c) 2025, Action Space hackathon team.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The whole ground-station pipeline in Isaac Sim: rendered RealSense -> computer vision -> classical control.

A RealSense twin (848x480 RGB-D, D435-like field of view) is rendered in the 3D arena. The defenders see
the attackers only through it, with the same code the real ground station runs::

    rendered depth --DepthDroneDetector (background subtraction)--> blobs      [CV, tello/realsense_tracker.py]
       (or rendered color --YOLOv8--> boxes --depth--> 3D, with --yolo weights.pt)
    blobs near a defender --> defender fix; the rest --> Kalman + Hungarian tracker --> attacker tracks
    defender positions + attacker tracks --> BlockController (+ hanging mesh) --> velocity commands
                                                                         [classical, tello/controllers.py]

The truth stays inside the sim, so the vision error (track vs true attacker position) is printed each
round. ``--export_yolo N`` saves every N-th rendered frame with auto-generated YOLO labels (the sim knows
where every drone is) into ``media/yolo_dataset/`` to train a YOLOv8 drone detector.

.. code-block:: bash

    python scripts/vision_loop.py --tuned                       # sticky-mesh game, camera window on
    python scripts/vision_loop.py --task Intercept-Defense-3v2-v0 --tuned
    python scripts/vision_loop.py --headless --export_yolo 5 --rounds 40   # synthetic YOLO dataset
    python scripts/vision_loop.py --headless --rounds 6 --video media/videos/vision_depth.mp4   # no window, recorded

"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Vision-in-the-loop 3v2 defense in Isaac Sim.")
parser.add_argument("--task", type=str, default="Intercept-Defense-3v2-StickyNet-v0")
parser.add_argument("--tuned", action="store_true", help="Use config/block_tuned.json.")
parser.add_argument("--cam_pos", type=float, nargs=3, default=[7.6, -2.6, 2.4], help="RealSense position (arena frame).")
parser.add_argument("--cam_target", type=float, nargs=3, default=[2.0, 0.3, 1.2], help="Point the RealSense looks at.")
parser.add_argument("--no_window", action="store_true", help="Do not show the camera/vision window.")
parser.add_argument("--rounds", type=int, default=0, help="Stop after this many rounds (0 = until closed).")
parser.add_argument("--export_yolo", type=int, default=0, help="Save every N-th frame with YOLO labels (0 = off).")
parser.add_argument("--yolo", type=str, default=None, help="YOLOv8 weights: detect on color instead of depth blobs.")
parser.add_argument("--relay", action="store_true", help="Relay blocking (predicted intercepts, beaten marker retreats).")
parser.add_argument("--video", type=str, default=None, help="Write the camera view with detections to this .mp4.")
parser.add_argument("--summary", type=str, default=None, help="Write the run's win/tracking stats to this .json.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import json
import os
import sys
import time

import cv2
import gymnasium as gym
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

import action_space.tasks  # noqa: F401
import isaaclab.sim as sim_utils
from isaaclab.envs import ViewerCfg
from isaaclab.sensors import CameraCfg
from isaaclab.utils.math import quat_from_matrix
from isaaclab_tasks.utils import load_cfg_from_registry

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO)
from arena_scene import design_scene, make_tello_usd  # noqa: E402
from tello.controllers import BlockController  # noqa: E402
from tello.localizer import load_arena, look_at_R_wc  # noqa: E402
from tello.ppo_policy import split_obs  # noqa: E402
from tello.realsense_tracker import DepthDroneDetector, MultiTargetTracker, make_T, transform_points  # noqa: E402

W, H = 848, 480
FX = FY = 424.0 / np.tan(np.radians(69.0 / 2))  # D435 color: ~69 deg horizontal field of view
K = np.array([[FX, 0.0, W / 2], [0.0, FY, H / 2], [0.0, 0.0, 1.0]])
DRONE_HALF = np.array([0.06, 0.06, 0.025])  # Tello half-size with props [m], for YOLO boxes


def project(T_cw: np.ndarray, pts: np.ndarray):
    """World points (N,3) -> pixels (N,2) and depth (N,)."""
    pc = transform_points(T_cw, pts)
    z = pc[:, 2]
    uv = (pc[:, :2] / np.maximum(z, 1e-6)[:, None]) * [FX, FY] + [W / 2, H / 2]
    return uv, z


def project_one(T_cw: np.ndarray, p: np.ndarray):
    """One world point (1,3) -> ([u], [v]) pixel, and its depth."""
    uv, z = project(T_cw, p)
    return (uv[:, 0], uv[:, 1]), z


def yolo_boxes(T_cw: np.ndarray, centers: np.ndarray) -> list[tuple[float, float, float, float]]:
    """Normalized YOLO boxes (cx, cy, w, h) of the drones at ``centers`` that are in view."""
    corners = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]) * DRONE_HALF
    out = []
    for c in centers:
        uv, z = project(T_cw, c + corners)
        if np.any(z < 0.2):
            continue
        x0, y0 = np.clip(uv.min(axis=0), 0, [W, H])
        x1, y1 = np.clip(uv.max(axis=0), 0, [W, H])
        if x1 - x0 < 2 or y1 - y0 < 2:
            continue
        out.append(((x0 + x1) / 2 / W, (y0 + y1) / 2 / H, (x1 - x0) / W, (y1 - y0) / H))
    return out


def splat_depth(depth: np.ndarray, T_cw: np.ndarray, blobs, rng) -> np.ndarray:
    """Add objects to the rendered depth image as filled ellipses at their range, plus RealSense-like noise.

    Isaac Lab draws the drones as visualization markers, which the color image shows but the depth
    annotator skips, so their depth is added here. ``blobs`` = list of (center (3,), half width, half height).
    """
    layer = np.full_like(depth, np.inf)
    items = []
    for c, hw, hh in blobs:
        (uv,), (z,) = project(T_cw, np.asarray(c)[None])
        if z > 0.2:
            items.append((z, uv, hw, hh))
    for z, uv, hw, hh in sorted(items, key=lambda it: -it[0]):  # far first, near drawn over it
        axes = (max(1, int(FX * hw / z)), max(1, int(FY * hh / z)))
        cv2.ellipse(layer, (int(uv[0]), int(uv[1])), axes, 0, 0, 360, float(z), -1)
    out = np.minimum(np.where(np.isfinite(depth), depth, np.inf), layer)
    noisy = out + rng.normal(0.0, 1.0, out.shape).astype(np.float32) * (0.0015 * out**2)  # D435-like: ~2.5 cm at 4 m, ~10 cm at 8 m
    return np.where(np.isfinite(out), noisy, np.inf).astype(np.float32)


def mesh_points(top: np.ndarray, vel: np.ndarray, cfg) -> np.ndarray:
    """Points along a hanging mesh strip, same geometry as the env's capture test."""
    start = top - np.array([0.0, 0.0, 0.08])
    bottom = top - vel * cfg.net_trail * np.array([1.0, 1.0, 0.0])
    bottom[2] = top[2] - cfg.net_length
    return np.linspace(start, bottom, 20)


def seg_dist(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Distance from points p (M,3) to the segment a-b."""
    ab = b - a
    t = np.clip(((p - a) @ ab) / max(ab @ ab, 1e-9), 0.0, 1.0)
    return np.linalg.norm(p - (a + t[:, None] * ab), axis=-1)


class YoloDetector:
    """YOLOv8 boxes on the color image, lifted to 3D with the median depth inside each box."""

    def __init__(self, weights: str, T_wc: np.ndarray):
        from ultralytics import YOLO  # lazy: only with --yolo

        self.model = YOLO(weights)
        self.T_wc = T_wc

    def detect(self, rgb: np.ndarray, depth: np.ndarray):
        res = self.model.predict(rgb[..., ::-1], verbose=False, conf=0.3)[0]
        out = []
        for x0, y0, x1, y1 in res.boxes.xyxy.cpu().numpy():
            patch = depth[int(y0):int(y1) + 1, int(x0):int(x1) + 1]
            patch = patch[np.isfinite(patch) & (patch > 0.2)]
            if patch.size == 0:
                continue
            d = float(np.median(patch)) + 0.05
            u, v = (x0 + x1) / 2, (y0 + y1) / 2
            ray = np.array([(u - W / 2) / FX, (v - H / 2) / FY, 1.0])
            out.append((transform_points(self.T_wc, ray * d)[0], int((x1 - x0) * (y1 - y0))))
        return out


def main():
    arena = load_arena()
    cfg = load_cfg_from_registry(args_cli.task, "env_cfg_entry_point")
    cfg.scene.num_envs = 1
    models = os.path.join(REPO, "media", "models")
    # true-size Tellos (the depth camera must see realistic blobs)
    cfg.defender_usd_path = make_tello_usd(os.path.join(models, "tello_blue_1x.usda"), (0.15, 0.4, 0.95), scale=1.0)
    cfg.attacker_usd_path = make_tello_usd(os.path.join(models, "tello_red_1x.usda"), (0.9, 0.15, 0.12), scale=1.0)
    cfg.downed_usd_path = make_tello_usd(os.path.join(models, "tello_gray_1x.usda"), (0.45, 0.47, 0.5), scale=1.0)
    cfg.show_arena_markers = False
    cfg.viewer = ViewerCfg(eye=(8.5, -5.5, 4.0), lookat=(2.5, 0.0, 1.0))
    cfg.realsense = CameraCfg(
        prim_path="/World/RealSense",
        update_period=0,
        width=W,
        height=H,
        data_types=["rgb", "distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg.from_intrinsic_matrix(
            intrinsic_matrix=K.flatten().tolist(), width=W, height=H, clipping_range=(0.1, 20.0)
        ),
    )
    env = gym.make(args_cli.task, cfg=cfg)
    design_scene(arena, ground=False, back_wall=False, wall_opacity=0.35)
    u = env.unwrapped
    dev = u.device
    nd, na = cfg.num_defenders, cfg.num_attackers

    # RealSense pose (OpenCV optical frame == Isaac "ros" convention)
    cam_pos = np.array(args_cli.cam_pos, float)
    R_wc = look_at_R_wc(cam_pos, np.array(args_cli.cam_target, float))
    T_wc = make_T(R_wc, cam_pos)
    T_cw = np.linalg.inv(T_wc)
    quat = quat_from_matrix(torch.tensor(R_wc, dtype=torch.float32)).unsqueeze(0).to(dev)
    u.realsense.set_world_poses(torch.tensor(cam_pos, dtype=torch.float32, device=dev).unsqueeze(0), quat, convention="ros")

    def grab():
        out = u.realsense.data.output
        rgb = out["rgb"][0, ..., :3].cpu().numpy().astype(np.uint8)
        depth = out["distance_to_image_plane"][0, ..., 0].cpu().numpy().astype(np.float32)
        return rgb, depth

    def render(n: int):
        for _ in range(n):
            u.sim.render()
            u.realsense.update(dt=0.0, force_recompute=True)

    # empty-arena background: hide every drone (and mesh) marker, as on the real field before the game
    obs, _ = env.reset()
    u.episode_length_buf[:] = 0
    u.markers.set_visibility(False)
    render(20)
    detector = DepthDroneDetector(K, T_wc, arena)
    frames = []
    noise_rng = np.random.default_rng(1)
    for _ in range(10):
        render(1)
        frames.append(splat_depth(grab()[1], T_cw, [], noise_rng))
    detector.build_background(frames)
    u.markers.set_visibility(True)
    yolo = YoloDetector(args_cli.yolo, T_wc) if args_cli.yolo else None
    print(f"[vision] RealSense at {cam_pos.tolist()} -> {args_cli.cam_target}, background built, "
          f"detector: {'YOLOv8 ' + args_cli.yolo if yolo else 'depth background subtraction'}", flush=True)

    settings = {}
    tuned_path = os.path.join(REPO, "config", "block_tuned.json")
    if args_cli.tuned and os.path.exists(tuned_path):
        with open(tuned_path) as f:
            settings = json.load(f)["settings"]
    ctrl = BlockController(arena, nd, na, max_speed_xy=cfg.def_max_speed_xy, max_speed_z=cfg.def_max_speed_z,
                           net_carriers=list(cfg.net_carriers[:nd]), relay=args_cli.relay, **settings)
    max_vel = np.array([cfg.def_max_speed_xy, cfg.def_max_speed_xy, cfg.def_max_speed_z])
    tracker = MultiTargetTracker(arena, max_attackers=na)
    slot_of_track: dict[int, int] = {}

    yolo_dir = os.path.join(REPO, "media", "yolo_dataset")
    if args_cli.export_yolo:
        for split in ("images", "labels"):
            os.makedirs(os.path.join(yolo_dir, split), exist_ok=True)
        with open(os.path.join(yolo_dir, "data.yaml"), "w") as f:
            f.write(f"path: {yolo_dir}\ntrain: images\nval: images\nnames:\n  0: drone\n")

    show = not args_cli.no_window and not args_cli.headless
    run_stamp = time.strftime("%H%M%S")
    writer = None
    if args_cli.video:
        os.makedirs(os.path.dirname(os.path.abspath(args_cli.video)), exist_ok=True)
        writer = cv2.VideoWriter(args_cli.video, cv2.VideoWriter_fourcc(*"mp4v"), 1.0 / cfg.sim.dt / cfg.decimation, (W, H))
    totals = {"rounds": 0, "wins": 0, "breaches": 0, "catches": 0.0, "hits": 0, "alive_steps": 0, "err_sum": 0.0}
    rng = np.random.default_rng(0)
    step, rounds, wins, saved = 0, 0, 0, 0
    err_sum, err_n, seen_steps = 0.0, 0, 0
    with torch.inference_mode():
        while simulation_app.is_running():
            t0 = time.time()
            rgb, depth = grab()
            t_sim = step * u.step_dt
            # what a real depth camera sees: every drone (flying or down) and every hanging mesh strip
            dp, dv = u.def_pos[0].cpu().numpy(), u.def_vel[0].cpu().numpy()
            carriers = (u.net[0] & u.def_alive[0]).cpu().numpy()
            blobs = [(c, 0.06, 0.03) for c in np.concatenate([dp, u.att_pos[0].cpu().numpy()])]
            for i in np.where(carriers)[0]:
                blobs += [(q, 0.06, 0.03) for q in mesh_points(dp[i], dv[i], cfg)]
            depth = splat_depth(depth, T_cw, blobs, rng)
            dets = yolo.detect(rgb, depth) if yolo else detector.detect(depth)

            # own drones' self-localized positions (noisy, from the obs) take the blobs next to them
            def_pos, def_vel, _, _, _, def_alive = split_obs(obs["policy"].cpu().numpy(), nd, na)
            live = np.where(def_alive[0])[0]
            Z = np.array([p for p, _ in dets]).reshape(-1, 3)
            att_dets = list(dets)
            if len(Z):
                lag = 0.0 if tracker._t is None else max(0.0, t_sim - tracker._t)
                owners = [def_pos[0, i] for i in live] + [tr.x[:3] + tr.x[3:] * lag for tr in tracker.tracks]
                if owners:
                    cost = np.linalg.norm(np.array(owners)[:, None] - Z[None], axis=-1)
                    # a blob anywhere along a carrier's mesh strip is that carrier, not an attacker
                    net_now = u.net[0].cpu().numpy()
                    for r, i in enumerate(live):
                        if net_now[i]:
                            pts = mesh_points(def_pos[0, i], def_vel[0, i], cfg)
                            cost[r] = np.minimum(cost[r], seg_dist(Z, pts[0], pts[-1]))
                    taken = {c for r, c in zip(*linear_sum_assignment(cost)) if r < len(live) and cost[r, c] < 0.35}
                    if len(live):
                        taken |= {c for c in range(len(Z)) if cost[: len(live), c].min() < 0.2}
                    att_dets = [d for c, d in enumerate(dets) if c not in taken]
            tracks = tracker.update(att_dets, t_sim)

            ids = {tr["id"] for tr in tracks}
            slot_of_track = {i: s for i, s in slot_of_track.items() if i in ids}
            for tr in tracks:
                if tr["id"] not in slot_of_track:
                    free = [s for s in range(na) if s not in slot_of_track.values()]
                    if free:
                        slot_of_track[tr["id"]] = free[0]
            att_pos = np.zeros((1, na, 3))
            att_vel = np.zeros((1, na, 3))
            att_alive = np.zeros((1, na), bool)
            for tr in tracks:
                s = slot_of_track.get(tr["id"])
                if s is not None:
                    att_pos[0, s], att_vel[0, s], att_alive[0, s] = tr["pos"], tr["vel"], True

            # vision vs the sim's truth: is each flying attacker tracked (a track within 0.5 m), and how well
            true_att = u.att_pos[0].cpu().numpy()
            true_alive = u.att_alive[0].cpu().numpy()
            for j in np.where(true_alive)[0]:
                seen_steps += 1
                if tracks:
                    d = min(float(np.linalg.norm(tr["pos"] - true_att[j])) for tr in tracks)
                    if d < 0.5:
                        err_n += 1
                        err_sum += d

            if args_cli.export_yolo and step % args_cli.export_yolo == 0:
                centers = np.concatenate([u.def_pos[0].cpu().numpy(), true_att])
                boxes = yolo_boxes(T_cw, centers)
                name = f"sim_{run_stamp}_{saved:05d}"
                cv2.imwrite(os.path.join(yolo_dir, "images", name + ".jpg"), rgb[..., ::-1])
                with open(os.path.join(yolo_dir, "labels", name + ".txt"), "w") as f:
                    f.writelines(f"0 {cx:.5f} {cy:.5f} {w:.5f} {h:.5f}\n" for cx, cy, w, h in boxes)
                saved += 1

            if show or writer is not None:
                view = rgb[..., ::-1].copy()
                mask = detector.last_mask
                if mask is not None and not yolo:
                    view[mask > 0] = (0.5 * view[mask > 0] + [0, 0, 127]).astype(np.uint8)
                for p, _ in dets:
                    (px, py), _ = project_one(T_cw, p[None])
                    cv2.circle(view, (int(px[0]), int(py[0])), 10, (0, 255, 255), 1)
                for tr in tracks:
                    (px, py), _ = project_one(T_cw, tr["pos"][None])
                    cv2.circle(view, (int(px[0]), int(py[0])), 14, (0, 0, 255), 2)
                    cv2.putText(view, f"attacker {tr['id']}", (int(px[0]) + 16, int(py[0])), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (0, 0, 255), 1)
                for i in live:
                    (px, py), _ = project_one(T_cw, def_pos[0, i][None])
                    cv2.circle(view, (int(px[0]), int(py[0])), 14, (255, 128, 0), 2)
                    mesh = " +mesh" if bool(u.net[0, i]) else ""
                    cv2.putText(view, f"defender {i}{mesh}", (int(px[0]) + 16, int(py[0])), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (255, 128, 0), 1)
                cv2.putText(view, f"RealSense twin | blobs {len(dets)} | attacker tracks {len(tracks)} | record {wins}/{rounds}",
                            (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
                if writer is not None:
                    writer.write(view)
                if show:
                    cv2.imshow("RealSense twin: CV -> tracker -> block controller", view)
                    cv2.waitKey(1)

            ctrl.net = u.net.cpu().numpy()
            vel = ctrl(def_pos, def_vel, att_pos, att_vel, att_alive, def_alive)
            act = torch.as_tensor((vel / max_vel).reshape(1, -1), dtype=torch.float32, device=dev)
            obs, _, terminated, truncated, extras = env.step(act)
            step += 1
            if bool(terminated[0] | truncated[0]):
                log = extras["log"]
                rounds += 1
                win = log["Metrics/win_rate"] > 0.5
                wins += int(win)
                result = "WIN" if win else ("BREACH" if log["Metrics/breach_rate"] > 0.5 else "LOST")
                err = err_sum / max(err_n, 1)
                totals["rounds"] += 1
                totals["wins"] += int(win)
                totals["breaches"] += int(log["Metrics/breach_rate"] > 0.5)
                totals["catches"] += log.get("Metrics/net_catches", 0.0)
                totals["hits"] += err_n
                totals["alive_steps"] += seen_steps
                totals["err_sum"] += err_sum
                print(f"[round {rounds}] {result:<6} mesh catches {log.get('Metrics/net_catches', 0):.0f}"
                      f"  tracking error {100 * err:.0f} cm  attackers tracked {err_n / max(seen_steps, 1):.0%} of the time"
                      f"  record {wins}/{rounds}" + (f"  yolo frames {saved}" if saved else ""), flush=True)
                err_sum, err_n, seen_steps = 0.0, 0, 0
                tracker.reset()
                slot_of_track = {}
                if args_cli.rounds and rounds >= args_cli.rounds:
                    break
            if not args_cli.headless:
                time.sleep(max(0.0, u.step_dt - (time.time() - t0)))
    if writer is not None:
        writer.release()
        print(f"[vision] video: {args_cli.video}")
    if args_cli.summary:
        t = totals
        summary = {
            "task": args_cli.task, "detector": "yolov8" if yolo else "depth", "relay": args_cli.relay, "rounds": t["rounds"],
            "win_rate": t["wins"] / max(t["rounds"], 1), "breach_rate": t["breaches"] / max(t["rounds"], 1),
            "mesh_catches_per_round": t["catches"] / max(t["rounds"], 1),
            "tracked_fraction": t["hits"] / max(t["alive_steps"], 1), "tracking_error_m": t["err_sum"] / max(t["hits"], 1),
        }
        with open(args_cli.summary, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"[vision] summary: {json.dumps(summary)}", flush=True)
    if show:
        cv2.destroyAllWindows()
    u.set_debug_vis(False)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
