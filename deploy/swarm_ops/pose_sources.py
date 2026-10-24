"""Pose sources that put OUR drones' poses on the bus, in the judges' formation/drone<N> format.

The judges' runner does not localize drones it flies itself, so until they publish every
drone's pose, TeamDefensePolicy can publish ours (DEFENSE_POSE_SOURCE):
  judges  publish nothing (default): poses come from whoever else publishes formation/
  sim     dead-reckon each drone from the rc we command (Tello motion priors) - for dry runs
  tags    AprilTag localization from each drone's camera (tag_localizer.TagLocalizer)
SimAttackers adds scripted attackers (defense/attack_script.py) so a dry run has something to block.
"""
import math
import time

import numpy as np

from defense.attack_script import PLANS, scripted_position
from defense.blocker import rc_to_velocity

TAG10_HEIGHT_M = 1.22
SIM_STARTS_WORLD = ((5.0, -0.6, 1.0), (4.0, 0.0, 1.0), (5.0, 0.6, 1.0))  # start triangle on our side, facing tag 10


def world_to_map(w):
    X, Y, Z = w
    return [round(float(Y), 3), round(float(Z - TAG10_HEIGHT_M), 3), round(float(X), 3)]


def yaw_to_psi(yaw):
    """Inverse of psi_to_yaw: world yaw (CCW from +X) -> psi_deg (0 = facing tag 10, clockwise+)."""
    return round((math.degrees(math.pi - yaw) + 180.0) % 360.0 - 180.0, 1)


def pose_message(world_pos, yaw, rc=(0, 0, 0, 0), state="sim"):
    return {"state": state, "height_cm": round((world_pos[2]) * 100.0), "rc": list(rc),
            "p": world_to_map(world_pos), "psi_deg": yaw_to_psi(yaw), "fix_age": 0.0}


class SimPoses:
    """Dead-reckoned poses of our drones from the rc commands TeamDefensePolicy sends."""

    def __init__(self, names, cfg):
        self.cfg = cfg
        self.state = {n: [np.array(SIM_STARTS_WORLD[k % len(SIM_STARTS_WORLD)], float), math.pi]
                      for k, n in enumerate(names)}
        self.t = None

    def advance(self, rcs):
        """rcs: {name: (lr, fb, ud, yaw)} last commanded. Returns {name: pose message}."""
        now = time.time()
        dt = 0.0 if self.t is None else min(now - self.t, 0.3)
        self.t = now
        out = {}
        for name, st in self.state.items():
            rc = rcs.get(name, (0, 0, 0, 0))
            vel, turn = rc_to_velocity(rc, st[1], self.cfg)
            st[0] = st[0] + vel * dt
            st[0][2] = max(st[0][2], 0.0)
            st[1] = (st[1] + turn * dt + math.pi) % (2 * math.pi) - math.pi
            out[name] = pose_message(st[0], st[1], rc)
        return out


class SimAttackers:
    """Scripted attackers weaving toward the protected point, as extra formation/ poses."""

    STARTS_WORLD = ((1.0, -0.8), (1.0, 0.8))  # attackers start at the tag-10 wall

    def __init__(self, table_world, names=("drone91", "drone92")):
        self.table = np.asarray(table_world, float)
        self.names = names
        self.t0 = None

    def poses(self):
        now = time.time()
        self.t0 = self.t0 or now
        out = {}
        for k, name in enumerate(self.names):
            pos, arrived = scripted_position(self.STARTS_WORLD[k], self.table, now - self.t0, PLANS[k % len(PLANS)])
            if not arrived:
                out[name] = pose_message(pos, 0.0, state="sim-attacker")
        return out
