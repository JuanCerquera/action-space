"""defense: Action Space team defense (defense/team.py plan_team) as a swarm_ops policy.

Install: copy this file and our defense/ folder into swarm_ops/policies/, then register
"defense": TeamDefensePolicy in policies/__init__.py POLICIES.

Inputs (assumption): every drone's pose, ours and the attackers', on POSE_PREFIX + "drone<N>"
as published by swarm_formation.publish_telemetry: p = [x right, y up, z out of the wall] m
in the tag-10 map frame, psi_deg 0 = facing tag 10, positive clockwise seen from above.
World frame used by plan_team: X = map z (out of the wall), Y = map x, Z = map y + TAG10_HEIGHT_M.
"""
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from defense.blocker import BlockerConfig, to_rc, yaw_rate_command  # noqa: E402
from defense.team import plan_team  # noqa: E402

try:
    from .base import Policy, clip
except ImportError:                       # outside their package (tests)
    def clip(v, lim):
        return int(max(-lim, min(lim, round(v))))

    class Policy:
        name = "policy"
        bus = None

        def setup(self, drones):
            self.drones = list(drones)

        def step(self, d, t):
            raise NotImplementedError

        def describe(self, d):
            return self.name

POSE_PREFIXES = ("formation/",)           # topics "<prefix>drone<N>" carrying poses
POSE_MAX_AGE_S = 0.5                      # bus age limit for a pose message
FIX_MAX_AGE_S = 0.5                       # localization fix age limit (fix_age field)
TAG10_HEIGHT_M = 1.22                     # tag 10 centre above the floor
TARGET_MAP = (0.0, 0.0, 5.4)              # protected point, map frame: OUR side, 5.4 m out from the tag-10 wall
                                          # (attackers come from the tag-10 side). ASSUMPTION - set the real spot
FENCE_WORLD = ((0.6, 6.0), (-1.9, 1.6))   # (X out from wall, Y) limits for targets
ATTACK_MODE = os.environ.get("DEFENSE_ATTACK_MODE", "0") == "1"
FACE = os.environ.get("DEFENSE_FACE", "tag")   # "tag": keep looking at tag 10; "attacker": plan yaw
LOOK_AT_WORLD = (0.0, 0.0)                # tag 10 in world XY
TAG_YAW_MAX = 0.6                         # rad/s cap while turning back to tag 10 (gentle, keeps the fix)
REPLAN_S = 0.03
STEADY_ACC = 0.6                          # m/s^2: attackers accelerating less than this count as steady
VEL_SMOOTH = 0.3                          # weight of the new finite-difference velocity (smooths the weave)
MIN_RC_XY = 14                            # Tello ignores smaller horizontal rc
MIN_RC_DIST_M = 0.08
MAX_RC_XY, MAX_RC_Z, MAX_RC_YAW = 100, 100, 100   # no software cap: full stick = Tello's physical limit
TOPIC = "defense/"
POSE_SOURCE = os.environ.get("DEFENSE_POSE_SOURCE", "tags")     # tags | sim | judges (see pose_sources.py):
# the judges publish only THEIR attackers on formation/drone<N>, so we localize ourselves (positional.py)
SIM_ATTACKERS = os.environ.get("DEFENSE_SIM_ATTACKERS", "0") == "1"


def make_cfg():
    # standoff 0 / max_distance 4: hold the attacker's line where we are (no backing off); short leads so a
    # weaving attacker does not drag the blockers sideways.
    # Block the attacker's line to our net: nearest point to the blocker, >= standoff from the attacker, <= block_leash
    # from the goal, aimed at where the attacker will be (lead covers Tello response + tick latency). Ram near the goal,
    # or early when a steady attacker is within ram_confident_dist.
    return BlockerConfig(attack_side=(-1.0, 0.0, 0.0),  # attackers come from the tag-10 wall (-X)
                         block_forward=True,  # blockers hold their own line between the keeper and the attackers
                         standoff=0.6, max_distance=8.0, block_leash=1.8, min_lead_s=0.25, max_lead_s=0.8,
                         # ram line just in front of the blockers' line: blockers hold, never back up, and ram
                         # an attacker reaching it (backing off while sliding sideways lets weavers through)
                         ram_radius=2.0, ram_confident_dist=1.2, keeper_distance=0.6,
                         rc_per_mps=125.0, rc_per_mps_z=167.0, rc_per_radps=95.0, max_speed=0.8,
                         defender_speed=0.8, gain=1.5, z_min=0.5, z_max=2.3)


def map_to_world(p):
    x, y, z = p
    return np.array([z, x, y + TAG10_HEIGHT_M], float)


def world_to_map(w):
    X, Y, Z = w
    return [float(Y), float(Z - TAG10_HEIGHT_M), float(X)]


def psi_to_yaw(psi_deg):
    """psi 0 = facing the wall (world -X), clockwise+ -> yaw CCW from world +X in [-pi, pi]."""
    y = math.pi - math.radians(psi_deg)
    return (y + math.pi) % (2 * math.pi) - math.pi


def kick(lr, fb, dist):
    """Scale small horizontal rc up to MIN_RC_XY so the Tello actually moves."""
    m = max(abs(lr), abs(fb))
    if 0.5 < m < MIN_RC_XY and dist > MIN_RC_DIST_M:
        s = MIN_RC_XY / m
        lr, fb = clip(lr * s, 100), clip(fb * s, 100)
    return lr, fb


class TeamDefensePolicy(Policy):
    name = "defense"

    def __init__(self, attack_mode=None, face=None, pose_source=None, sim_attackers=None):
        self.attack_mode = ATTACK_MODE if attack_mode is None else attack_mode
        self.face = FACE if face is None else face
        self.pose_source = POSE_SOURCE if pose_source is None else pose_source
        self.sim_attackers = SIM_ATTACKERS if sim_attackers is None else sim_attackers
        self.cfg = make_cfg()
        self.table = map_to_world(TARGET_MAP)

    def setup(self, drones):
        super().setup(drones)
        if self.bus is None:
            from .telemetry import TelemetryBus  # only reached inside their package
            self.bus = TelemetryBus()
        self.ours = [d.name for d in self.drones]
        self.previous = None
        self.att_track = {}               # name -> (pos, vel, t)
        self.att_acc = {}                 # name -> smoothed |acceleration| (m/s^2): low = steady
        self.ram_lock = {}                # our name -> attacker name it is committed to ramming
        self.plan_t = -1.0
        self.out = {}                     # name -> (rc, info)
        # Our own pose publisher (until the judges publish every drone's pose themselves).
        from pose_sources import SimAttackers, SimPoses
        self.sim = SimPoses(self.ours, self.cfg) if self.pose_source == "sim" else None
        self.tags = None
        if self.pose_source == "tags":
            from tag_localizer import TagLocalizer
            self.tags = TagLocalizer(self.drones)
            self.tags.start()
        self.attack_sim = SimAttackers(self.table) if self.sim_attackers else None

    def _publish_own_poses(self):
        msgs = {}
        if self.sim is not None:
            msgs.update(self.sim.advance({n: rc for n, (rc, _) in self.out.items()}))
        if self.tags is not None:
            for name, pose in self.tags.poses().items():
                msgs[name] = {"state": "tags", "height_cm": None, "rc": None, **pose}
        if self.attack_sim is not None:
            msgs.update(self.attack_sim.poses())
        for name, msg in msgs.items():
            self.bus.publish(f"{POSE_PREFIXES[0]}{name}", msg)

    # ---- telemetry
    def _poses(self):
        """{name: (world pos, yaw or None, t)} for every fresh, localized pose."""
        now = time.time()
        res = {}
        for prefix in POSE_PREFIXES:
            for topic, (data, age) in self.bus.since(prefix, POSE_MAX_AGE_S).items():
                name = topic[len(prefix):]
                if not isinstance(data, dict) or data.get("p") is None:
                    continue
                fa = data.get("fix_age")
                if fa is not None and fa > FIX_MAX_AGE_S:
                    continue
                psi = data.get("psi_deg")
                res[name] = (map_to_world(data["p"]), None if psi is None else psi_to_yaw(psi), now - age)
        return res

    def _attackers(self, poses):
        out = []
        for name, (pos, _, t) in sorted(poses.items()):
            if name in self.ours:
                continue
            old = self.att_track.get(name)
            vel = np.zeros(3)
            if old is not None and t > old[2] + 1e-3:
                raw = (pos - old[0]) / (t - old[2])
                vel = VEL_SMOOTH * raw + (1 - VEL_SMOOTH) * old[1]
            elif old is not None:
                vel = old[1]
            self.att_track[name] = (pos, vel, t) if old is None or t > old[2] + 1e-3 else old
            if old is not None and t > old[2] + 1e-3:  # steadiness: smoothed |acceleration|
                acc = float(np.linalg.norm(vel - old[1])) / (t - old[2])
                self.att_acc[name] = 0.85 * self.att_acc.get(name, acc) + 0.15 * acc
            out.append((name, pos, self.att_track[name][1]))
        return out

    # ---- planning
    def _replan(self):
        poses = self._poses()
        atts = self._attackers(poses)
        def_pos, def_yaw, alive = [], [], []
        for name in self.ours:
            hit = poses.get(name)
            ok = hit is not None and hit[1] is not None
            def_pos.append(hit[0] if ok else np.zeros(3))
            def_yaw.append(hit[1] if ok else 0.0)
            alive.append(ok)
        att_pos = [a[1] for a in atts]
        att_vel = [a[2] for a in atts]
        names = [a[0] for a in atts]
        prev = None
        if self.previous is not None:     # previous is stored by attacker name -> index now
            prev = [names.index(n) if n in names else None for n in self.previous]
        plans = plan_team(def_pos, def_yaw, alive, att_pos, att_vel, [True] * len(atts), self.table, self.cfg,
                          fence=FENCE_WORLD, previous=prev, attack_mode=self.attack_mode,
                          avoid=True, face=(self.face == "attacker"),
                          steady=[self.att_acc.get(n, 9.0) < STEADY_ACC for n in names],
                          latched=[names.index(self.ram_lock[n]) if self.ram_lock.get(n) in names else None
                                   for n in self.ours])
        self.previous = [None if p.attacker is None else names[p.attacker] for p in plans]
        self.ram_lock = {n: (names[p.facing] if p.role == "intercept" and p.facing is not None else None)
                         for n, p in zip(self.ours, plans)}  # commit to a ram until the attacker is gone
        self.out = {}
        for k, name in enumerate(self.ours):
            p = plans[k]
            if not alive[k]:
                self.out[name] = ((0, 0, 0, 0), {"role": "hover", "target_map": None, "attacker": None})
                continue
            pos, yaw = def_pos[k], def_yaw[k]
            vel = np.array(p.velocity, float)
            # keep inside the height band even if the planner target is outside it
            if pos[2] < self.cfg.z_min and vel[2] < 0 or pos[2] > self.cfg.z_max and vel[2] > 0:
                vel[2] = 0.0
            turn = p.yaw_rate if self.face == "attacker" else float(np.clip(
                yaw_rate_command(pos, yaw, LOOK_AT_WORLD, self.cfg), -TAG_YAW_MAX, TAG_YAW_MAX))
            lr, fb, ud, yw = to_rc(vel, yaw, self.cfg, turn)
            dist = float(np.linalg.norm(np.asarray(p.target, float)[:2] - pos[:2])) if p.target is not None else 0.0
            lr, fb = kick(lr, fb, dist)
            rc = (clip(lr, MAX_RC_XY), clip(fb, MAX_RC_XY), clip(ud, MAX_RC_Z), clip(yw, MAX_RC_YAW))
            self.out[name] = (rc, {"role": p.role,
                                   "target_map": None if p.target is None else
                                   [round(v, 3) for v in world_to_map(p.target)],
                                   "attacker": None if p.attacker is None else names[p.attacker]})

    def step(self, d, t):
        now = time.time()
        if now - self.plan_t > REPLAN_S:
            self.plan_t = now
            self._publish_own_poses()
            self._replan()
        rc, info = self.out.get(d.name, ((0, 0, 0, 0), {"role": "hover", "target_map": None, "attacker": None}))
        try:
            self.bus.publish(f"{TOPIC}{d.name}", info)
        except Exception:
            pass
        return rc

    def describe(self, d):
        info = getattr(self, "out", {}).get(d.name)
        return f"defense:{info[1]['role'] if info else 'idle'}"
