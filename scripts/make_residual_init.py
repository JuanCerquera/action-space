# Copyright (c) 2025, Action Space hackathon team.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Create a starting checkpoint for residual RL whose correction is exactly zero.

A freshly initialized actor outputs random corrections that degrade the scripted block controller
before PPO has learned anything. This takes any checkpoint of the residual task (e.g. model_0.pt of a
short run), zeroes the actor's output layer and lowers the exploration noise, and saves it as a new
run, so training resumes from "pure block controller" and only learns changes that help.

.. code-block:: bash

    python scripts/make_residual_init.py logs/rsl_rl/intercept_defense_3v2_residual/<run>/model_0.pt
    python scripts/rsl_rl/train.py --task Intercept-Defense-3v2-Residual-v0 --headless --num_envs 512 \
        --resume --load_run residual_init --checkpoint model_0.pt

"""

import argparse
import os

import torch

parser = argparse.ArgumentParser(description="Zero the residual policy's output so it starts as the block controller.")
parser.add_argument("checkpoint", help="Any rsl_rl checkpoint of Intercept-Defense-3v2-Residual-v0.")
parser.add_argument("--std", type=float, default=0.2, help="Initial exploration noise (action std).")
parser.add_argument("--run_name", default="residual_init", help="Run folder to create next to the source run.")
args = parser.parse_args()

ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
state = ckpt["model_state_dict"]
last = max(int(k.split(".")[1]) for k in state if k.startswith("actor.") and k.endswith(".weight"))
state[f"actor.{last}.weight"].zero_()
state[f"actor.{last}.bias"].zero_()
for key in ("std", "log_std"):
    if key in state:
        state[key] = torch.full_like(state[key], args.std if key == "std" else float(torch.log(torch.tensor(args.std))))
ckpt["iter"] = 0  # (rsl_rl needs the optimizer state to resume; Adam adapts to the new weights quickly)

out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(args.checkpoint))), args.run_name)
os.makedirs(out_dir, exist_ok=True)
out = os.path.join(out_dir, "model_0.pt")
torch.save(ckpt, out)
print(f"Wrote {out} (actor.{last} zeroed, std {args.std})")
