# Copyright (c) 2025, Action Space hackathon team.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Run a trained rsl_rl PPO defender policy with plain torch (no Isaac Lab needed).

The observation must match the envs' ``_get_observations``:
defender pos (ND*3) * pos_scale, defender vel (ND*3) * vel_scale,
attacker pos (NA*3) * pos_scale (zeroed if down), attacker vel (NA*3) * vel_scale, attacker alive (NA),
and for the Defense task also defender alive (ND).
Actions are normalized velocity commands (ND*3) in [-1, 1]; multiply by the max speeds.
"""

from __future__ import annotations

import glob
import os
import re

import numpy as np
import torch

POS_SCALE = 0.25
VEL_SCALE = 1.0 / 1.5


def latest_checkpoint(log_root: str) -> str:
    """Newest model_<iter>.pt of the newest run under ``log_root`` (e.g. logs/rsl_rl/intercept_swarm_3v2)."""
    run = sorted(d for d in glob.glob(os.path.join(log_root, "*")) if os.path.isdir(d))[-1]
    models = glob.glob(os.path.join(run, "model_*.pt"))
    return max(models, key=lambda p: int(re.search(r"model_(\d+)\.pt$", p).group(1)))


def load_actor(checkpoint: str, device: str = "cpu") -> torch.nn.Sequential:
    """Rebuild the deterministic actor MLP (Linear + ELU, linear output) from an rsl_rl checkpoint."""
    state = torch.load(checkpoint, map_location=device, weights_only=False)["model_state_dict"]
    idx = sorted({int(k.split(".")[1]) for k in state if k.startswith("actor.") and k.endswith(".weight")})
    layers = []
    for n, i in enumerate(idx):
        w = state[f"actor.{i}.weight"]
        linear = torch.nn.Linear(w.shape[1], w.shape[0])
        linear.weight.data.copy_(w)
        linear.bias.data.copy_(state[f"actor.{i}.bias"])
        layers.append(linear)
        if n < len(idx) - 1:
            layers.append(torch.nn.ELU())
    return torch.nn.Sequential(*layers).to(device).eval()


def make_obs(def_pos, def_vel, att_pos, att_vel, att_alive, def_alive=None) -> np.ndarray:
    """Build the (B, obs_dim) policy observation from arena-frame states (batched like the controllers).

    Pass ``def_alive`` (B, ND) for the Defense task, whose policy also sees which defenders are still flying.
    """
    B = def_pos.shape[0]
    alive = att_alive.astype(np.float32)[..., None]
    parts = [
        (def_pos * POS_SCALE).reshape(B, -1),
        (def_vel * VEL_SCALE).reshape(B, -1),
        (att_pos * alive * POS_SCALE).reshape(B, -1),
        (att_vel * alive * VEL_SCALE).reshape(B, -1),
        alive.reshape(B, -1),
    ]
    if def_alive is not None:
        parts.append(def_alive.astype(np.float32).reshape(B, -1))
    return np.concatenate(parts, axis=-1).astype(np.float32)


def split_obs(obs: np.ndarray, nd: int = 3, na: int = 2):
    """Inverse of :func:`make_obs`: (def_pos, def_vel, att_pos, att_vel, att_alive, def_alive) with batch dim.

    ``def_alive`` is all True when the observation has no defender-alive block (older tasks).
    """
    B = obs.shape[0]
    o = 0

    def take(n):
        nonlocal o
        out = obs[:, o : o + n]
        o += n
        return out

    def_pos = take(nd * 3).reshape(B, nd, 3) / POS_SCALE
    def_vel = take(nd * 3).reshape(B, nd, 3) / VEL_SCALE
    att_pos = take(na * 3).reshape(B, na, 3) / POS_SCALE
    att_vel = take(na * 3).reshape(B, na, 3) / VEL_SCALE
    att_alive = take(na) > 0.5
    def_alive = take(nd) > 0.5 if obs.shape[1] >= o + nd else np.ones((B, nd), dtype=bool)
    return def_pos, def_vel, att_pos, att_vel, att_alive, def_alive
