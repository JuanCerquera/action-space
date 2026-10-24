# Copyright (c) 2026, Action Space team.
# SPDX-License-Identifier: Apache-2.0
"""Isaac Lab 3.0: 3v2 team defense — blue defenders run the Tello blocker policy, red attackers weave in.

The judges' real room (tag-10 wall and every tag from their AprilTag map), with the judges' Tello model
(assets/tello/tello.urdf, converted to USD at startup) colored by team: blue defenders, red attackers, dark grey
once knocked out. The defenders run
defense/team.py, the code the Tello test panel runs: each attacker gets a
blocker in its path to the site, the spare defender keeps goal (or with
--attack-mode rams the attacker nearest the site), targets stay apart and
commands avoid teammates. Attackers follow defense/attack_script.py (weaving,
varying heights). The Tellos physically collide; a ram counts only when the two
collision boxes (98 x 92.5 x 41 mm) touch with an impact (or stay pinned), then
BOTH_OUT_P / OUR_LOSS_P decide who loses thrust and drops. Attackers that reach the site score.

The policy runs at 10 Hz like on the real drones and outputs world-frame
velocities; a force controller tracks them (body force + attitude/yaw torque).
--record out.json writes the trajectories (30 fps, w-x-y-z quaternions) per episode.

    python scripts/03_team_defense.py --steps 8000 --attack-mode --check-finished
    python scripts/03_team_defense.py --from-policy --viz none --steps 6000 --record rec.json --record-episodes 2
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveSceneCfg
import isaaclab.sim as sim_utils
from isaaclab.utils import configclass, instantiate, replace
from isaaclab_newton.physics import NewtonCfg

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from defense.attack_script import PLANS, scripted_position  # noqa: E402
from defense.blocker import BlockerConfig, min_pairwise_distance  # noqa: E402
from defense.team import plan_team  # noqa: E402
from defense.blocker import rc_to_velocity  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "deploy" / "swarm_ops"))
import move_test  # noqa: E402  (the movetest swarm_ops policy flown on the real Tellos)


# The judges' real room, from their AprilTag map (config/judges_map.yaml = their config/map.yaml).
# World frame (same as their sim and our policy): X out from the tag-10 wall, Y = map x (right),
# Z up from the floor; map frame is x right, y up, z out of the wall, origin at tag 10.
MAP_PATH = Path(__file__).resolve().parents[1] / "config" / "judges_map.yaml"
TAGS_PATH = Path(__file__).resolve().parents[1] / "config" / "judges_tags.yaml"
TAG10_HEIGHT = 1.22  # m, tag 10 centre above the floor
CEILING = 2.74
ROOM_X = 6.0  # far wall, m out from the tag-10 wall
ROOM_Y = (-2.0, 3.7)  # side walls (map x): left from their room config (room_y_min -2.0); right covers tags out to 3.16 (their config: 1.70, unresolved)
IGNORED_TAGS = {1, 2, 4, 7, 11, 12, 22, 23, 24, 25}  # floor tags: ignored by us and the judges
P_MAP_TO_WORLD = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
TABLE = np.array((5.4, 0.0, TAG10_HEIGHT))  # protected point: OUR side, far from the tag-10 wall (attackers' side); assumption, as in team_defense.py
SITE_RADIUS = 0.4  # an attacker this close to the protected point (horizontally) scores
FENCE = ((0.6, 5.5), (-1.8, 3.4))  # X, Y limits for defender targets: inside the walls (left wall at -2.0)
DEFENDER_POSITIONS = ((5.0, -0.6, 1.0), (4.0, 0.0, 1.0), (5.0, 0.6, 1.0))  # start triangle on our side, pointing at the attackers, at hover height
ATTACKER_STARTS = ((1.0, -0.8), (1.0, 0.8))  # attackers' side: in front of the tag-10 wall; heights come from the attack plans
DT = 0.01  # 100 Hz physics (200 Hz ran the viewer at 0.4x real time)
POLICY_EVERY = round(0.1 / DT)  # physics steps per policy step: 10 Hz
BOTH_OUT_P = 0.9  # chance that a ram knocks out both drones
OUR_LOSS_P = 0.3  # otherwise: chance that only OUR drone is out (else only the attacker)
SETTLE_STEPS = round(2.0 / DT)  # after every attacker is down or scored, wait this long before the next episode
DEFAULT_TRACK_GAIN = 4.0  # 1/s; defender acceleration per m/s of velocity error
DEFAULT_MAX_ACCEL = 4.0  # m/s^2: Tello limit (judges' TelloCommandCfg max_lin_accel)
TELLO_MAX_XY, TELLO_MAX_Z = 0.8, 0.6  # m/s at full stick (judges' measured Tello motion fit): the physical cap
ATTACKER_KP, ATTACKER_KD = 6.0, 4.0  # PD on the scripted attacker path

BLUE = (0.15, 0.45, 0.95)
RED = (0.95, 0.20, 0.18)
DEAD = (0.16, 0.16, 0.17)
_NO_COLLISION = sim_utils.CollisionPropertiesCfg(collision_enabled=False)

# The judges' Tello (their simulation_infra/assets/tello/tello.urdf, copied to assets/tello/): one rigid body,
# collision box 98 x 92.5 x 41 mm, 87 g, four prop-guard discs as visuals. Converted to USD with Isaac Lab's
# URDF converter at startup, then flattened into one variant per team color (Newton's viewer takes uncolored
# shapes' primvars:displayColor).
TELLO_URDF = Path(__file__).resolve().parents[1] / "assets" / "tello" / "tello.urdf"
TELLO_USD_DIR = Path(__file__).resolve().parents[1] / "assets" / "tello" / "usd"
TELLO_SIZE = (0.098, 0.0925, 0.041)  # m, collision box (judges' URDF / kit TELLO_SIZE)
CONTACT_MARGIN = 0.01  # m: collision boxes closer than this count as touching
TELLO_TEAM_USD = {team: TELLO_USD_DIR / f"tello_{team}.usda" for team in ("blue", "red")}


def prepare_tello_usds():
    """Convert the judges' Tello URDF (cached by the converter) and write blue / red flattened variants."""
    from pxr import Gf, Usd, UsdGeom, UsdShade
    from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg
    conv = UrdfConverter(UrdfConverterCfg(asset_path=str(TELLO_URDF), usd_dir=str(TELLO_USD_DIR / "converted"),
                                          fix_base=False, merge_fixed_joints=True, joint_drive=None))
    for team, color in (("blue", BLUE), ("red", RED)):
        stage = Usd.Stage.Open(Usd.Stage.Open(conv.usd_path).Flatten())
        for prim in stage.Traverse():
            if not prim.IsA(UsdGeom.Gprim):
                continue
            UsdShade.MaterialBindingAPI(prim).UnbindAllBindings()
            name = prim.GetName()
            c = (0.05, 0.05, 0.05) if name.startswith("sphere") else tuple(0.6 * x for x in color) if name == "box_1" else color
            lin = [((x + 0.055) / 1.055) ** 2.4 if x > 0.04045 else x / 12.92 for x in c]  # displayColor is linear
            UsdGeom.Gprim(prim).CreateDisplayColorAttr([Gf.Vec3f(*lin)])
        stage.GetRootLayer().Export(str(TELLO_TEAM_USD[team]))
    print(f"[INFO] Tello: {TELLO_URDF.name} -> {conv.usd_path} -> {[p.name for p in TELLO_TEAM_USD.values()]}", flush=True)


_ID_ROT = tuple(ArticulationCfg.InitialStateCfg().rot)
W_IDX = 0 if abs(_ID_ROT[0] - 1.0) < 1e-6 else 3  # quaternion layout of this Isaac Lab: w first or last
FACING_TAG10_ROT = (0.0, 0.0, 0.0, 1.0) if W_IDX == 0 else (0.0, 0.0, 1.0, 0.0)  # yaw = pi: facing the tag-10 wall


def _team_material(color):
    return sim_utils.PreviewSurfaceCfg(
        diffuse_color=color, emissive_color=(color[0] * 0.35, color[1] * 0.35, color[2] * 0.35), roughness=0.45
    )


def _tello(prim_path: str, team: str, pos, rot=None) -> ArticulationCfg:
    """Unique spawn + init_state per drone (shared init_state would pile everyone together)."""
    spawn = sim_utils.UsdFileCfg(usd_path=str(TELLO_TEAM_USD[team]), activate_contact_sensors=False)
    init_state = ArticulationCfg.InitialStateCfg(pos=pos, joint_pos={}, joint_vel={}, **({} if rot is None else {"rot": rot}))
    return ArticulationCfg(prim_path=prim_path, spawn=spawn, init_state=init_state, actuators={})


def _static(name: str, spawn, pos) -> AssetBaseCfg:
    return AssetBaseCfg(prim_path=f"/World/Room/{name}", spawn=spawn, init_state=AssetBaseCfg.InitialStateCfg(pos=pos))


def _box(size, color, opacity=1.0):
    return sim_utils.CuboidCfg(size=size, collision_props=_NO_COLLISION, visual_material=sim_utils.PreviewSurfaceCfg(
        diffuse_color=color, emissive_color=tuple(c * 0.2 for c in color), opacity=opacity, roughness=0.8))


@configclass
class TeamDefenseSceneCfg(InteractiveSceneCfg):
    """The judges' room (tag-10 wall, mapped tags, protected point): three blue defenders, two red attackers."""

    ground = AssetBaseCfg(prim_path="/World/Ground", spawn=sim_utils.GroundPlaneCfg())
    light = AssetBaseCfg(prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=2500.0, color=(0.8, 0.8, 0.8)))
    # Room: the tag-10 wall at X = 0; side and far walls sized to cover every mapped tag
    # (Y -2.0 .. 3.7, X 0 .. 6.0); their sim's room config has Y -2.0 .. 1.7, which predates the new tags.
    floor = _static("Floor", _box((ROOM_X, ROOM_Y[1] - ROOM_Y[0], 0.01), (0.55, 0.58, 0.62)),
                    (ROOM_X / 2, sum(ROOM_Y) / 2, 0.005))
    wall = _static("Wall", _box((0.06, ROOM_Y[1] - ROOM_Y[0], CEILING), (0.85, 0.85, 0.82)),
                   (-0.03, sum(ROOM_Y) / 2, CEILING / 2))
    wall_left = _static("WallLeft", _box((ROOM_X, 0.06, CEILING), (0.75, 0.78, 0.85), 0.25),
                        (ROOM_X / 2, ROOM_Y[0] - 0.03, CEILING / 2))
    wall_right = _static("WallRight", _box((ROOM_X, 0.06, CEILING), (0.75, 0.78, 0.85), 0.25),
                         (ROOM_X / 2, ROOM_Y[1] + 0.03, CEILING / 2))
    wall_far = _static("WallFar", _box((0.06, ROOM_Y[1] - ROOM_Y[0], CEILING), (0.75, 0.78, 0.85), 0.25),
                       (ROOM_X + 0.03, sum(ROOM_Y) / 2, CEILING / 2))
    target = _static("Target", sim_utils.SphereCfg(radius=0.2, collision_props=_NO_COLLISION,
                                                   visual_material=sim_utils.PreviewSurfaceCfg(
                                                       diffuse_color=(0.12, 0.85, 0.35), emissive_color=(0.05, 0.35, 0.12),
                                                       opacity=0.5)), tuple(TABLE))

    defender_0: ArticulationCfg = _tello("{ENV_REGEX_NS}/Defender0", "blue", DEFENDER_POSITIONS[0], FACING_TAG10_ROT)
    defender_1: ArticulationCfg = _tello("{ENV_REGEX_NS}/Defender1", "blue", DEFENDER_POSITIONS[1], FACING_TAG10_ROT)
    defender_2: ArticulationCfg = _tello("{ENV_REGEX_NS}/Defender2", "blue", DEFENDER_POSITIONS[2], FACING_TAG10_ROT)
    attacker_0: ArticulationCfg = _tello("{ENV_REGEX_NS}/Attacker0", "red", (*ATTACKER_STARTS[0], PLANS[0]["z0"]))
    attacker_1: ArticulationCfg = _tello("{ENV_REGEX_NS}/Attacker1", "red", (*ATTACKER_STARTS[1], PLANS[1]["z0"]))


DEFENDER_NAMES = ("defender_0", "defender_1", "defender_2")
ATTACKER_NAMES = ("attacker_0", "attacker_1")


def reset_robot(robot):
    """Restore the asset's configured pose, velocity, and rotor state."""
    robot.write_root_pose_to_sim_index(root_pose=robot.data.default_root_pose.torch.clone())
    robot.write_root_velocity_to_sim_index(root_velocity=robot.data.default_root_vel.torch.clone())
    if robot.num_joints:
        robot.write_joint_position_to_sim_index(position=robot.data.default_joint_pos.torch.clone())
        robot.write_joint_velocity_to_sim_index(velocity=robot.data.default_joint_vel.torch.clone())
    robot.reset()


def _body_mass_and_ids(robot):
    body_ids = robot.find_bodies(".*")[0][:1]  # the Tello is one rigid body (base_link, camera merged)
    if len(body_ids) != 1:
        raise RuntimeError(f"Expected one Tello body; found {body_ids}")
    mass = robot.data.body_mass.torch.sum(dim=1)
    if not torch.isfinite(mass).all() or (mass <= 0).any():
        raise RuntimeError(f"Invalid drone mass: {mass}")
    return body_ids, mass


INERTIA = (7.4e-5, 8.2e-5, 1.32e-4)  # kg m^2, Tello EDU (judges' URDF: roll, pitch, yaw)
ATT_WN, ATT_ZETA = 20.0, 0.8  # attitude hold: natural frequency (rad/s) and damping
_W_INDEX = {}  # quaternion layout per robot (index of w in root_pose_w[3:7])
YAW_RATE_CMD = {}  # robot -> commanded yaw rate (rad/s, CCW); the attitude hold tracks it (Tello yaw channel)


def robot_yaw(robot):
    q = robot.data.root_pose_w.torch[0, 3:7].tolist()
    w, x, y, z = (q[0], q[1], q[2], q[3]) if W_IDX == 0 else (q[3], q[0], q[1], q[2])
    return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def attitude_torque(robot):
    """World torque (1, 1, 3) holding the body upright and damping spin: without it the force-only
    drones tumble while they translate."""
    q = robot.data.root_pose_w.torch[0, 3:7]
    if W_IDX == 0:
        w, x, y, z = q
    else:
        x, y, z, w = q
    body_z = torch.stack([2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)])
    up = torch.tensor([0.0, 0.0, 1.0], device=body_z.device, dtype=body_z.dtype)
    omega = robot.data.root_vel_w.torch[0, 3:6]
    inertia = torch.tensor(INERTIA, device=omega.device, dtype=omega.dtype)  # body ~ world axes while near level
    kp, kd = inertia * ATT_WN ** 2, inertia * 2 * ATT_ZETA * ATT_WN
    omega_ref = torch.tensor([0.0, 0.0, YAW_RATE_CMD.get(robot, 0.0)], device=omega.device, dtype=omega.dtype)
    return (kp * torch.linalg.cross(body_z, up) - kd * (omega - omega_ref))[None, None, :]


def set_accel_force(robot, gravity, accel):
    """Body force giving world-frame acceleration `accel` (m/s^2, tensor (3,)) with gravity compensated,
    plus the attitude-hold torque."""
    body_ids, mass = _body_mass_and_ids(robot)
    gravity_t = torch.tensor(gravity, device=robot.device, dtype=accel.dtype)
    force = (mass[:, None] * (accel[None, :] - gravity_t))[:, None, :]
    robot.permanent_wrench_composer.reset()
    robot.permanent_wrench_composer.add_forces_and_torques_index(
        forces=force, torques=attitude_torque(robot).to(force.dtype), body_ids=body_ids, is_global=True)


def tello_limits(v):
    """Clip a world velocity to what a Tello can fly: 0.8 m/s horizontal, 0.6 m/s vertical."""
    v = np.array(v, float)
    h = np.linalg.norm(v[:2])
    if h > TELLO_MAX_XY:
        v[:2] *= TELLO_MAX_XY / h
    v[2] = np.clip(v[2], -TELLO_MAX_Z, TELLO_MAX_Z)
    return v


def cut_thrust(robot):
    """Knocked out or done: no more force, the drone drops."""
    robot.permanent_wrench_composer.reset()


def _clip_norm(v, limit):
    n = torch.linalg.vector_norm(v)
    return v * (limit / n) if n > limit else v


def pos_vel(robot):
    return (robot.data.root_pose_w.torch[0, :3].cpu().numpy().astype(float),
            robot.data.root_vel_w.torch[0, :3].cpu().numpy().astype(float))


def _quat_wxyz(R):
    """Rotation matrix -> unit quaternion (w, x, y, z)."""
    R = np.asarray(R, float)
    w = np.sqrt(max(0.0, 1.0 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    x = np.copysign(np.sqrt(max(0.0, 1.0 + R[0, 0] - R[1, 1] - R[2, 2])) / 2, R[2, 1] - R[1, 2])
    y = np.copysign(np.sqrt(max(0.0, 1.0 - R[0, 0] + R[1, 1] - R[2, 2])) / 2, R[0, 2] - R[2, 0])
    z = np.copysign(np.sqrt(max(0.0, 1.0 - R[0, 0] - R[1, 1] + R[2, 2])) / 2, R[1, 0] - R[0, 1])
    q = np.array([w, x, y, z])
    return q / np.linalg.norm(q)


def spawn_map_tags():
    """Every tag in the judges' map as a thin black square at its real pose and size (white backing)."""
    import yaml
    tags = yaml.safe_load(MAP_PATH.read_text())["tags"]
    sizes = {int(k): v / 1000.0 for k, v in yaml.safe_load(TAGS_PATH.read_text())["sizes_mm"].items()}
    identity = _ID_ROT  # the quaternion layout this Isaac Lab expects
    xyzw = abs(identity[3] - 1.0) < 1e-6 and abs(identity[0]) < 1e-6
    from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
    protos, index, pos, quat = {}, [], [], []
    for tid, pose in tags.items():
        if int(tid) in IGNORED_TAGS:
            continue
        R_w = P_MAP_TO_WORLD @ np.array(pose["R"], float)  # tag axes in world; column 2 = out of the tag face
        t_w = P_MAP_TO_WORLD @ np.array(pose["t"], float) + np.array([0.0, 0.0, TAG10_HEIGHT])
        if t_w[2] < 0.2 and abs(R_w[2, 2]) < 0.9:
            # On the floor but mapped tilted / facing down (judges' map has this for tags 1, 4, 7, 22, 24):
            # draw it flat, face up, keeping its in-plane direction.
            x = R_w[:, 0] - R_w[2, 0] * np.array([0.0, 0.0, 1.0])
            x = x / np.linalg.norm(x) if np.linalg.norm(x) > 1e-6 else np.array([1.0, 0.0, 0.0])
            R_w = np.column_stack([x, np.cross([0.0, 0.0, 1.0], x), [0.0, 0.0, 1.0]])
            t_w[2] = max(t_w[2], 0.0)
        q = _quat_wxyz(R_w)
        q = (q[1], q[2], q[3], q[0]) if xyzw else tuple(q)
        s = round(sizes.get(int(tid), 0.132), 4)
        key = f"tag_{int(s * 1000)}mm"
        protos.setdefault(key, _box((s, s, 0.004), (0.02, 0.02, 0.02)))
        index.append(list(protos).index(key))
        pos.append(t_w + 0.01 * R_w[:, 2])  # just off the surface it is stuck to
        quat.append(q)
    global TAG_MARKERS
    TAG_MARKERS = VisualizationMarkers(VisualizationMarkersCfg(prim_path="/Visuals/map_tags", markers=protos))
    dev = sim_device()
    TAG_MARKERS.visualize(translations=torch.tensor(np.array(pos), dtype=torch.float32, device=dev),
                          orientations=torch.tensor(np.array(quat), dtype=torch.float32, device=dev),
                          marker_indices=torch.tensor(index))
    print(f"[INFO] room: {len(tags)} mapped tags from {MAP_PATH.name}", flush=True)


TAG_MARKERS = None


MARKERS = None  # move-test only: takeoff spots and 1 m forward points


def make_markers():
    try:
        from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
        sphere = lambda color: sim_utils.SphereCfg(radius=0.04, visual_material=_team_material(color))  # noqa: E731
        return VisualizationMarkers(VisualizationMarkersCfg(prim_path="/Visuals/drones", markers={
            "defender": sphere(BLUE), "attacker": sphere(RED), "out": sphere((0.4, 0.4, 0.4)),
            "intercept": sphere((1.0, 0.55, 0.1))}))
    except Exception as exc:  # markers are cosmetic; never block the run
        print(f"[WARN] no drone markers: {exc}", flush=True)
        return None


def _srgb(c):
    return tuple(12.92 * x if x <= 0.0031308 else 1.055 * x ** (1 / 2.4) - 0.055 for x in c)


class DroneColors:
    """Team colors live on the drones' own shapes: Newton's model.shape_color, which the viewer re-syncs every
    frame. A knocked-out drone turns dark grey; a new episode restores the team colors."""

    PRIM = {"defender_0": "Defender0", "defender_1": "Defender1", "defender_2": "Defender2",
            "attacker_0": "Attacker0", "attacker_1": "Attacker1"}

    def __init__(self):
        self.model, self.shapes, self.base, self.dead = None, {}, None, set()
        try:
            from isaaclab_newton.physics.newton_manager import NewtonManager
            self.model = NewtonManager.get_model()
            labels = list(self.model.shape_label)
            for name, prim in self.PRIM.items():
                self.shapes[name] = [k for k, lab in enumerate(labels) if f"/{prim}/" in str(lab)]
            self.base = self.model.shape_color.numpy().copy()
            print(f"[INFO] drone colors: {sum(map(len, self.shapes.values()))} Tello shapes recolorable at runtime",
                  flush=True)
            self._collision_report(labels)
        except Exception as exc:  # cosmetic; never block the run
            print(f"[WARN] runtime drone recolor unavailable ({exc!r}); team colors from spawn only", flush=True)
            self.model = None

    def _collision_report(self, labels):
        """Log the drones' shapes' collision group / flags: the collision boxes must collide with each other."""
        groups = self.model.shape_collision_group.numpy()
        flags = self.model.shape_flags.numpy()
        coll = [(str(labels[k]).split("/")[-1], int(groups[k]), int(flags[k])) for k in self.shapes.get("defender_0", [])]
        print(f"[INFO] defender_0 shapes (name, collision_group, flags): {coll}", flush=True)

    def set(self, name, alive):
        if self.model is None or (name in self.dead) == (not alive):
            return
        (self.dead.discard if alive else self.dead.add)(name)
        colors = self.model.shape_color.numpy()
        idx = self.shapes.get(name, [])
        colors[idx] = self.base[idx] if alive else np.array(_srgb(DEAD), dtype=colors.dtype)
        self.model.shape_color.assign(colors)
        print(f"[INFO] {name}: {'team color' if alive else 'knocked out, dark grey'} ({len(idx)} shapes)", flush=True)


DRONE_COLORS = None


def show_status(ep):
    if DRONE_COLORS is None:
        return
    for i, name in enumerate(DEFENDER_NAMES):
        DRONE_COLORS.set(name, ep.def_alive[i])
    for j, name in enumerate(ATTACKER_NAMES):
        DRONE_COLORS.set(name, ep.att_status[j] != "down")


def tello_contact(p1, yaw1, p2, yaw2, margin=CONTACT_MARGIN):
    """True when the two Tello collision boxes (yawed, kept level by the attitude hold) are within `margin`:
    separating-axis test in the horizontal plane plus vertical overlap."""
    hx, hy, hz = (v / 2 for v in TELLO_SIZE)
    if abs(float(p2[2]) - float(p1[2])) > 2 * hz + margin:
        return False
    d = np.asarray(p2[:2], float) - np.asarray(p1[:2], float)
    axes = [(np.array([np.cos(y), np.sin(y)]), np.array([-np.sin(y), np.cos(y)])) for y in (yaw1, yaw2)]
    for ax in (axes[0][0], axes[0][1], axes[1][0], axes[1][1]):
        r = sum(hx * abs(ax @ a) + hy * abs(ax @ b) for a, b in axes)
        if abs(d @ ax) > r + margin:
            return False
    return True


def quat_wxyz(robot):
    q = robot.data.root_pose_w.torch[0, 3:7].tolist()
    return [q[0], q[1], q[2], q[3]] if W_IDX == 0 else [q[3], q[0], q[1], q[2]]


class Recorder:
    """--record: trajectories at 30 fps in our world frame, quaternions always w, x, y, z."""

    FPS = 30

    def __init__(self, path, max_episodes):
        self.path, self.max = Path(path), max_episodes
        self.data = {"fps": self.FPS,
                     "room": {"wall_x": 0.0, "room_x": ROOM_X, "room_y": list(ROOM_Y), "ceiling": CEILING,
                              "tag10_height": TAG10_HEIGHT},
                     "goal": [float(v) for v in TABLE], "episodes": []}
        self.cur, self.next_t = None, 0.0

    def start(self, seed):
        self.cur, self.next_t = None, 0.0
        if len(self.data["episodes"]) < self.max:
            self.cur = {"seed": int(seed), "frames": [], "events": []}

    def frame(self, robots, ep):
        if self.cur is None or ep.t + 1e-9 < self.next_t:
            return
        self.next_t += 1.0 / self.FPS
        drones = {}
        for i, n in enumerate(DEFENDER_NAMES):
            drones[n] = {"team": "defender", "pos": [round(float(v), 4) for v in pos_vel(robots[n])[0]],
                         "quat_wxyz": [round(v, 5) for v in quat_wxyz(robots[n])], "alive": bool(ep.def_alive[i]),
                         "role": str(ep.roles[i]) if ep.def_alive[i] else "out"}
        for j, n in enumerate(ATTACKER_NAMES):
            drones[n] = {"team": "attacker", "pos": [round(float(v), 4) for v in pos_vel(robots[n])[0]],
                         "quat_wxyz": [round(v, 5) for v in quat_wxyz(robots[n])],
                         "alive": ep.att_status[j] != "down", "role": ep.att_status[j]}
        self.cur["frames"].append({"t": round(ep.t, 3), "drones": drones})

    def event(self, t, kind, who, text):
        if self.cur is not None:
            self.cur["events"].append({"t": round(t, 3), "type": kind, "who": list(who), "text": text})

    def finish(self):
        if self.cur is None:
            return
        self.data["episodes"].append(self.cur)
        self.cur = None
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(self.data, separators=(",", ":")))
        tmp.replace(self.path)
        print(f"[INFO] recorded episode {len(self.data['episodes'])}/{self.max} -> {self.path}", flush=True)


def sim_device():
    return "cuda:0" if torch.cuda.is_available() else "cpu"


class Episode:
    count = 0
    """Per-episode game state: who is in play, the scripted clock, and results."""

    def __init__(self):
        self.t = 0.0
        self.contact = {}  # (defender, attacker) -> seconds in contact
        # Each episode varies the attack: start spot, speed, weave size and timing (seeded per episode count).
        self.seed = Episode.count
        rng = np.random.default_rng(Episode.count)
        self.rng = np.random.default_rng(1000 + Episode.count)  # ram outcomes
        Episode.count += 1
        self.starts = [(x, y + rng.uniform(-0.5, 0.5)) for x, y in ATTACKER_STARTS]
        self.plans = [dict(pl, speed=pl["speed"] * rng.uniform(0.8, 1.25), weave=pl["weave"] * rng.uniform(0.5, 1.4),
                           phase=rng.uniform(0, 2 * np.pi), period=pl["period"] * rng.uniform(0.8, 1.3)) for pl in PLANS]
        self.def_alive = [True] * len(DEFENDER_NAMES)
        self.att_status = ["active"] * len(ATTACKER_NAMES)  # active / down / scored
        self.previous = None
        self.velocity_cmd = [np.zeros(3) for _ in DEFENDER_NAMES]
        self.roles = ["block"] * len(DEFENDER_NAMES)
        self.events = []
        self.min_teammate_dist = float("inf")
        self.finished_at = None

    def finished(self) -> bool:
        return all(s != "active" for s in self.att_status)


def begin_episode(robots, reason):
    for robot in robots.values():
        reset_robot(robot)
    print(f"[INFO] Episode reset ({reason}).", flush=True)
    return Episode()



# --from-policy: our swarm_ops policy (dist/policy.py) flies the defenders exactly as on the real drones:
# poses go on a telemetry bus as formation/drone<N> (judges' format), TeamDefensePolicy.step() returns Tello rc
# every 0.074 s (their runner's tick), and the rc becomes motion through the judges' measured Tello model.
PERIOD_STEPS = max(1, round(0.074 / DT))
POLICY_FILE = Path(__file__).resolve().parents[1] / "dist" / "policy.py"
POL_CFG = BlockerConfig(rc_per_mps=125.0, rc_per_mps_z=167.0, rc_per_radps=95.0)  # Tello: 0.8 m/s, 0.6 m/s, 60 deg/s at rc 100
_POLICY_MOD = None


class _SimClock:
    """Stands in for the `time` module inside the policy so it runs on sim time."""
    t = 0.0

    @classmethod
    def time(cls):
        return cls.t

    monotonic = time

    @staticmethod
    def sleep(_s):
        pass


class _SimBus:
    """Same calls as the judges' TelemetryBus (publish / latest / since), on sim time."""

    def __init__(self):
        self.data = {}

    def publish(self, topic, data, t=None, src=None):
        self.data[topic] = (data, _SimClock.t if t is None else t)

    def latest(self, topic, max_age=None):
        hit = self.data.get(topic)
        if hit is None:
            return None
        age = _SimClock.t - hit[1]
        return None if max_age is not None and age > max_age else (hit[0], age)

    def since(self, prefix="", max_age=None):
        now = _SimClock.t
        return {k: (d, now - t) for k, (d, t) in self.data.items()
                if k.startswith(prefix) and (max_age is None or now - t <= max_age)}


class _FakeTello:
    def __init__(self, num, robot):
        self.num, self.name, self.robot = num, f"drone{num}", robot

    def height_cm(self):
        return float(self.robot.data.root_pose_w.torch[0, 2]) * 100.0

    def battery(self):
        return 90


def _policy_module():
    global _POLICY_MOD
    if _POLICY_MOD is None:
        import importlib.util
        spec = importlib.util.spec_from_file_location("as_policy", POLICY_FILE)
        _POLICY_MOD = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_POLICY_MOD)
        for name in ("team_defense", "pose_sources", "tag_localizer"):
            sys.modules[name].time = _SimClock  # the policy's clock = sim time
        print(f"[INFO] driving defenders from {POLICY_FILE.name} (TeamDefensePolicy, 13.5 Hz rc)", flush=True)
    return _POLICY_MOD



# Threshold spheres (dotted: 3 great circles each): around the goal = max blocking distance, ram line, keeper spot;
# around each live attacker = minimum blocking distance and confident-ram range.
ZONE_DOTS = 36
ZONE_STYLE = {"leash": (0.25, 0.55, 1.0), "ram": (1.0, 0.25, 0.2), "keeper": (0.2, 0.9, 0.35), "keeper_ram": (0.7, 0.3, 0.95),
              "standoff": (1.0, 0.85, 0.1), "confident": (1.0, 0.5, 0.0)}
ZONE_MARKERS = None


def _dotted_sphere(center, radius):
    """A dotted ring on the floor under `center` (was a 3-ring sphere around it)."""
    a = np.linspace(0, 2 * np.pi, 3 * ZONE_DOTS, endpoint=False)
    ring = np.stack([np.cos(a), np.sin(a), 0 * a], 1) * radius
    return ring + np.array([center[0], center[1], 0.02])


def show_zones(cfg, att_pos, att_alive):
    global ZONE_MARKERS
    try:
        if ZONE_MARKERS is None:
            from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
            dot = lambda color: sim_utils.SphereCfg(radius=0.015, visual_material=_team_material(color))  # noqa: E731
            ZONE_MARKERS = VisualizationMarkers(VisualizationMarkersCfg(
                prim_path="/Visuals/zones", markers={k: dot(v) for k, v in ZONE_STYLE.items()}))
        keys = list(ZONE_STYLE)
        pts, idx = [], []
        for key, radius in (("leash", cfg.block_leash), ("ram", cfg.ram_radius), ("keeper", cfg.keeper_distance),
                            ("keeper_ram", cfg.keeper_ram_radius)):
            ring = _dotted_sphere(TABLE, radius)
            pts.append(ring); idx += [keys.index(key)] * len(ring)
        ZONE_MARKERS.visualize(translations=torch.tensor(np.concatenate(pts), dtype=torch.float32, device=sim_device()),
                               marker_indices=torch.tensor(idx))
    except Exception as exc:
        print(f"[WARN] zones not drawn: {exc}", flush=True)


class PolicyDriver:
    def __init__(self):
        mod = _policy_module()
        self.ps = sys.modules["pose_sources"]
        self.policy = mod.TeamDefensePolicy(pose_source="judges", sim_attackers=False)
        self.policy.bus = _SimBus()
        self.drones = None

    def tick(self, robots, ep, att_state):
        _SimClock.t = ep.t
        if self.drones is None:
            self.drones = [_FakeTello(i + 1, robots[n]) for i, n in enumerate(DEFENDER_NAMES)]
            self.policy.setup(self.drones)
        bus = self.policy.bus
        for i, d in enumerate(self.drones):  # our drones' poses (a knocked-out one stops publishing)
            if ep.def_alive[i]:
                pos, _ = pos_vel(d.robot)
                bus.publish(f"formation/{d.name}", self.ps.pose_message(pos, robot_yaw(d.robot), state="sim"))
        for j, (pos, _) in enumerate(att_state):  # attackers, as the judges' formation would publish them
            if ep.att_status[j] == "active":
                bus.publish(f"formation/drone9{j + 1}", self.ps.pose_message(pos, 0.0, state="attacker"))
        vels, roles = [], []
        for i, d in enumerate(self.drones):
            if not ep.def_alive[i]:
                vels.append(np.zeros(3))
                roles.append("out")
                YAW_RATE_CMD[d.robot] = 0.0
                continue
            rc = self.policy.step(d, ep.t)
            vel, turn = rc_to_velocity(rc, robot_yaw(d.robot), POL_CFG)
            YAW_RATE_CMD[d.robot] = turn
            vels.append(vel)
            roles.append(self.policy.describe(d).split(":")[-1])
        self.n = getattr(self, "n", 0) + 1
        if self.n % 2 == 0:  # threshold spheres, ~7 Hz
            show_zones(self.policy.cfg, [p for p, _ in att_state], [st == "active" for st in ep.att_status])
        if self.n % 4 == 0 and ep.t < 9.0:  # trace for diagnosis: positions, roles, targets every ~0.3 s
            info = self.policy.out
            parts = [f"t={ep.t:.2f}"]
            for i, d in enumerate(self.drones):
                pos, _ = pos_vel(d.robot)
                tgt = info.get(d.name, (None, {}))[1].get("target_map") if d.name in info else None
                parts.append(f"D{i+1}:{roles[i]}@{np.round(pos[:2], 2).tolist()} tgt_map={tgt} rc={info.get(d.name, ((0,0,0,0),))[0]}")
            for j, (pos, vel) in enumerate(att_state):
                parts.append(f"A{j+1}:{ep.att_status[j]}@{np.round(pos[:2], 2).tolist()} v={np.round(vel[:2], 2).tolist()}")
            print("[TRACE] " + " | ".join(parts), flush=True)
        return vels, roles


def run_scene(sim, robots, args, cfg: BlockerConfig):
    gravity = sim.cfg.gravity
    steps, steps_in_episode = 0, 0
    episodes = []
    ep = None
    rec = Recorder(args.record, args.record_episodes) if args.record else None

    while sim.is_running() and (args.steps == 0 or steps < args.steps):
        ui_reset = sim.consume_reset_request()
        settled = ep is not None and ep.finished_at is not None and steps_in_episode - ep.finished_at >= SETTLE_STEPS
        if ep is None or ui_reset or (settled and args.auto_reset):
            if ep is not None:
                episodes.append(ep)
                if rec:
                    rec.finish()
            ep = begin_episode(robots, "start" if ep is None else "Reset Episode" if ui_reset else "episode over")
            show_status(ep)
            if rec:
                rec.start(ep.seed)
            steps_in_episode = 0

        att_state = [pos_vel(robots[n]) for n in ATTACKER_NAMES]
        def_state = [pos_vel(robots[n]) for n in DEFENDER_NAMES]

        # Attackers: PD along their scripted path; done ones drop.
        for j, name in enumerate(ATTACKER_NAMES):
            if ep.att_status[j] != "active":
                cut_thrust(robots[name])
                continue
            target, arrived = scripted_position(ep.starts[j], TABLE, ep.t, ep.plans[j])
            ahead, _ = scripted_position(ep.starts[j], TABLE, ep.t + DT, ep.plans[j])
            target_vel = (ahead - target) / DT
            pos, vel = att_state[j]
            accel = torch.tensor(ATTACKER_KP * (target - pos) + ATTACKER_KD * (target_vel - vel), device=sim.device,
                                 dtype=torch.float32)
            set_accel_force(robots[name], gravity, _clip_norm(accel, args.max_accel))
            if arrived and np.linalg.norm(pos[:2] - TABLE[:2]) < SITE_RADIUS:
                ep.att_status[j] = "scored"
                ep.events.append({"t": round(ep.t, 2), "event": f"{name} scored"})
                if rec:
                    rec.event(ep.t, "score", [name], f"{name} reached the site")
                print(f"[INFO] t={ep.t:.1f}s {name} reached the site.", flush=True)

        # Defenders: either our swarm_ops policy.py exactly as on the real drones (--from-policy), or the
        # shared team step at 10 Hz; a velocity-tracking force every physics step.
        if args.from_policy:
            if steps_in_episode == 0:
                ep.policy = PolicyDriver()
            if steps_in_episode % PERIOD_STEPS == 0:
                ep.velocity_cmd, roles = ep.policy.tick(robots, ep, att_state)
                if roles != ep.roles:
                    ep.roles = roles
                    print(f"[INFO] t={ep.t:.1f}s roles {ep.roles}", flush=True)
        elif steps_in_episode % POLICY_EVERY == 0:
            plans = plan_team(
                [p for p, _ in def_state], [0.0] * len(DEFENDER_NAMES), ep.def_alive,
                [p for p, _ in att_state], [v for _, v in att_state], [s == "active" for s in ep.att_status],
                TABLE, cfg, fence=FENCE, previous=ep.previous, attack_mode=args.attack_mode, face=False,
            )
            ep.previous = [p.attacker for p in plans]
            ep.velocity_cmd = [p.velocity for p in plans]
            if [p.role for p in plans] != ep.roles:
                ep.roles = [p.role for p in plans]
                print(f"[INFO] t={ep.t:.1f}s roles {ep.roles}", flush=True)
        for i, name in enumerate(DEFENDER_NAMES):
            if not ep.def_alive[i]:
                cut_thrust(robots[name])
                continue
            v_cmd_i = tello_limits(ep.velocity_cmd[i])
            accel = torch.tensor(args.track_gain * (v_cmd_i - def_state[i][1]), device=sim.device,
                                 dtype=torch.float32)
            set_accel_force(robots[name], gravity, _clip_norm(accel, args.max_accel))

        for robot in robots.values():
            robot.write_data_to_sim()
        sim.step()
        steps += 1
        steps_in_episode += 1
        ep.t += DT
        for robot in robots.values():
            robot.update(sim.get_physics_dt())

        for name, robot in robots.items():
            if not torch.isfinite(robot.data.root_pose_w.torch).all() or not torch.isfinite(robot.data.root_vel_w.torch).all():
                raise RuntimeError(f"{name}: non-finite state at step {steps}")

        def_pos = [pos_vel(robots[n])[0] for n in DEFENDER_NAMES]
        att_pos = [pos_vel(robots[n])[0] for n in ATTACKER_NAMES]
        if rec:
            rec.frame(robots, ep)
        live = [p for p, ok in zip(def_pos, ep.def_alive) if ok]
        ep.min_teammate_dist = min(ep.min_teammate_dist, min_pairwise_distance(live))
        # Contact = the two Tello collision boxes touching (within CONTACT_MARGIN); the physics bumps them apart.
        # A ram needs an impact (closing speed >= hit_min_speed, from the velocities just before this step, i.e.
        # before the collision response) or contact held for hit_hold_s.
        def_yaw = [robot_yaw(robots[n]) for n in DEFENDER_NAMES]
        att_yaw = [robot_yaw(robots[n]) for n in ATTACKER_NAMES]
        hits = []
        for i in range(len(def_pos)):
            for j in range(len(att_pos)):
                touching = (ep.def_alive[i] and ep.att_status[j] == "active"
                            and tello_contact(def_pos[i], def_yaw[i], att_pos[j], att_yaw[j]))
                ep.contact[(i, j)] = ep.contact.get((i, j), 0.0) + DT if touching else 0.0
                if not touching or any(i == a or j == b for a, b in hits):
                    continue
                off = att_pos[j] - def_pos[i]
                off = off / max(np.linalg.norm(off), 1e-6)
                closing = float(np.dot(def_state[i][1] - att_state[j][1], off))
                if closing >= cfg.hit_min_speed or ep.contact[(i, j)] >= cfg.hit_hold_s:
                    hits.append((i, j))
                    print(f"[INFO] t={ep.t:.2f}s contact {DEFENDER_NAMES[i]}-{ATTACKER_NAMES[j]}: centres "
                          f"{np.linalg.norm(att_pos[j] - def_pos[i]):.3f} m apart, closing {closing:.2f} m/s, "
                          f"held {ep.contact[(i, j)]:.2f} s", flush=True)
        for i, j in hits:  # ram outcome: both out (BOTH_OUT_P), else ONE drone: ours with OUR_LOSS_P, else the attacker
            if not (ep.def_alive[i] and ep.att_status[j] == "active"):
                continue
            ep.contact = {}  # the survivor separates before it can be hit again
            if ep.rng.random() < BOTH_OUT_P:
                ep.def_alive[i] = False
                ep.att_status[j] = "down"
                outcome = f"{DEFENDER_NAMES[i]} rammed {ATTACKER_NAMES[j]}: both out"
            elif ep.rng.random() < OUR_LOSS_P:
                ep.def_alive[i] = False
                outcome = f"{DEFENDER_NAMES[i]} rammed {ATTACKER_NAMES[j]}: OUR drone is out, the attacker flies on"
            else:
                ep.att_status[j] = "down"
                outcome = f"{DEFENDER_NAMES[i]} rammed {ATTACKER_NAMES[j]}: attacker down"
            ep.events.append({"t": round(ep.t, 2), "event": outcome})
            print(f"[INFO] t={ep.t:.1f}s {outcome}.", flush=True)
            show_status(ep)
            if rec:
                rec.event(ep.t, "ram", [DEFENDER_NAMES[i], ATTACKER_NAMES[j]], outcome)
                for name, out in ((DEFENDER_NAMES[i], not ep.def_alive[i]), (ATTACKER_NAMES[j], ep.att_status[j] == "down")):
                    if out:
                        rec.event(ep.t, "down", [name], f"{name} knocked out")
        if ep.finished() and ep.finished_at is None:
            ep.finished_at = steps_in_episode
            print(f"[INFO] t={ep.t:.1f}s episode decided: attackers {ep.att_status}", flush=True)

        if steps % 400 == 0:
            print(f"[INFO] step={steps} t={ep.t:.1f}s roles={ep.roles} attackers={ep.att_status} "
                  f"min_teammate_dist={ep.min_teammate_dist:.2f} m", flush=True)
            for name in DEFENDER_NAMES + ATTACKER_NAMES:  # diagnostics: attitude and tracking
                r = robots[name]
                gz = float(r.data.projected_gravity_b.torch[0, 2])
                tilt = float(np.degrees(np.arccos(np.clip(-gz, -1.0, 1.0))))
                spin = float(torch.linalg.vector_norm(r.data.root_vel_w.torch[0, 3:6]))
                p, v = pos_vel(r)
                i = DEFENDER_NAMES.index(name) if name in DEFENDER_NAMES else None
                err = np.linalg.norm(ep.velocity_cmd[i] - v) if i is not None else float("nan")
                print(f"[DIAG] {name} pos={np.round(p, 2).tolist()} vel={np.round(v, 2).tolist()} "
                      f"tilt={tilt:.0f}deg spin={spin:.1f}rad/s vel_err={err:.2f}", flush=True)

    episodes.append(ep)
    if rec:
        rec.finish()
    if args.steps and steps != args.steps:
        raise RuntimeError(f"Simulation stopped after {steps}/{args.steps} steps")
    first = episodes[0]
    if args.check_finished and not first.finished():
        raise RuntimeError(f"first episode undecided after {first.t:.1f} s: attackers {first.att_status}")
    if args.check_separation and first.min_teammate_dist < args.min_separation:
        raise RuntimeError(f"defenders came within {first.min_teammate_dist:.3f} m (< {args.min_separation} m)")

    print("[INFO] Scene result: " + json.dumps({
        "steps": steps,
        "simulated_seconds": steps * sim.get_physics_dt(),
        "attack_mode": args.attack_mode,
        "episodes": [{"seconds": round(e.t, 2), "attackers": e.att_status, "defenders_alive": e.def_alive,
                      "events": e.events, "min_teammate_dist_m": round(e.min_teammate_dist, 3)} for e in episodes],
        "device": str(sim.device),
    }), flush=True)


# --move-test: preview of deploy/swarm_ops/move_test.py on our three drones in parallel; attackers parked.
MOVE_TEST_DRONES = DEFENDER_NAMES  # = movetest:1,2,3
MOVE_TEST_STARTS = ((3.0, -0.6), (2.0, 0.0), (3.0, 0.6))  # world XY on the floor: triangle pointing at tag 10 (front 2.0 m out, rear pair 3.0 m, 1.2 m apart)
MOVE_TEST_YAW = float(np.pi)  # heading at takeoff: facing tag 10 (-X), as the judges place them: forward = -X, right = +Y
MOVE_TEST_PARK = {"attacker_0": (5.5, -0.5), "attacker_1": (5.5, 3.3)}
MOVE_TAKEOFF_Z, MOVE_TAKEOFF_S = 1.0, 2.0  # Tello takes off to about 0.8-1 m
MOVE_LAND_SPEED, MOVE_WAIT_S = 0.4, 3.0
MOVE_PERIOD_S = 0.074  # swarm_policies.py tick
MOVE_CFG = BlockerConfig(rc_per_mps=125, rc_per_mps_z=167, rc_per_radps=95)  # Tello priors, as move_test.py


def place(robot, xyz):
    pose = robot.data.default_root_pose.torch.clone()
    pose[:, :3] = torch.tensor(xyz, device=pose.device, dtype=pose.dtype)
    robot.write_root_pose_to_sim_index(root_pose=pose)
    robot.write_root_velocity_to_sim_index(root_velocity=torch.zeros_like(robot.data.default_root_vel.torch))
    robot.reset()


class _SimTello:
    """What MoveTestPolicy reads from a drone: num, name, height_cm()."""

    def __init__(self, robot, num):
        self.robot, self.num, self.name = robot, num, f"drone{num}"

    def height_cm(self):
        return float(self.robot.data.root_pose_w.torch[0, 2]) * 100.0


def run_move_test(sim, robots, args):
    gravity = sim.cfg.gravity
    drones = [robots[n] for n in MOVE_TEST_DRONES]
    fakes = [_SimTello(r, k + 1) for k, r in enumerate(drones)]
    starts = [np.array((*xy, 0.0)) for xy in MOVE_TEST_STARTS]
    steps, cycle = 0, 0

    def restart():
        nonlocal cycle
        cycle += 1
        for name, xy in MOVE_TEST_PARK.items():
            place(robots[name], (*xy, 0.05))
        for r, xy in zip(drones, MOVE_TEST_STARTS):
            place(r, (*xy, 0.05))
        policy = move_test.MoveTestPolicy()
        policy.setup(fakes)  # one instance, all drones, like the umbrella
        print(f"[MOVE] cycle {cycle}: takeoff from {MOVE_TEST_STARTS}, facing tag 10 (forward = -X toward the wall, right = +Y)", flush=True)
        return {"phase": "ground", "t": 0.0, "policy": policy, "leg": None, "rc": [(0, 0, 0, 0)] * len(drones),
                "next_tick": 0.0, "landed": [False] * len(drones), "min_sep": float("inf")}

    def fmt(ps):
        return " ".join(f"d{k + 1}={np.round(p - s0, 2).tolist()}" for k, (p, s0) in enumerate(zip(ps, starts)))

    st = restart()
    while sim.is_running() and (args.steps == 0 or steps < args.steps):
        if sim.consume_reset_request():
            st = restart()
        for name in MOVE_TEST_PARK:
            cut_thrust(robots[name])
        state = [pos_vel(r) for r in drones]
        ps = [p for p, _ in state]
        v_cmd = [np.zeros(3) for _ in drones]
        ph = st["phase"]
        if ph == "ground":
            if st["t"] >= 0.5:
                st.update(phase="takeoff", t=0.0, z0=[p[2] for p in ps])
                print("[MOVE] takeoff", flush=True)
        elif ph == "takeoff":
            s = min(1.0, st["t"] / MOVE_TAKEOFF_S)
            for k, p in enumerate(ps):
                z0 = st["z0"][k]
                rate = (MOVE_TAKEOFF_Z - z0) / MOVE_TAKEOFF_S if s < 1.0 else 0.0
                target = np.array((starts[k][0], starts[k][1], z0 + (MOVE_TAKEOFF_Z - z0) * s))
                v_cmd[k] = np.array((0.0, 0.0, rate)) + 2.0 * (target - p)
            if st["t"] >= MOVE_TAKEOFF_S + 0.5:
                st.update(phase="policy", t=0.0, next_tick=0.0)
                print(f"[MOVE] airborne rel {fmt(ps)}", flush=True)
        elif ph == "policy":
            st["min_sep"] = min(st["min_sep"], min_pairwise_distance(ps))
            if st["t"] >= st["next_tick"]:
                t_pol = st["next_tick"]
                rcs = [st["policy"].step(f, t_pol) for f in fakes]
                st["next_tick"] += MOVE_PERIOD_S
                if any(rc is None for rc in rcs):  # same clock: all finish on the same tick
                    st.update(phase="land", t=0.0, land_xy=[p[:2].copy() for p in ps])
                    print(f"[MOVE] t={t_pol:.2f}s leg=done -> land, rel {fmt(ps)} "
                          f"min_sep={st['min_sep']:.2f} m", flush=True)
                else:
                    st["rc"] = rcs
                    leg = st["policy"].describe(fakes[0])
                    if leg != st["leg"]:
                        st["leg"] = leg
                        print(f"[MOVE] t={t_pol:.2f}s leg={leg} rc={list(rcs[0])} rel {fmt(ps)} "
                              f"min_sep={st['min_sep']:.2f}", flush=True)
            if st["phase"] == "policy":
                v_cmd = [rc_to_velocity(rc, MOVE_TEST_YAW, MOVE_CFG)[0] for rc in st["rc"]]
        if st["phase"] == "land":
            for k, p in enumerate(ps):
                v_cmd[k] = np.array((*(2.0 * (st["land_xy"][k] - p[:2])), -MOVE_LAND_SPEED))
                if p[2] < 0.08:
                    st["landed"][k] = True
            if all(st["landed"]):
                st.update(phase="wait", t=0.0)
                print(f"[MOVE] landed, rel {fmt(ps)}", flush=True)
        if st["phase"] == "wait" and st["t"] >= MOVE_WAIT_S:
            st = restart()
        flying = st["phase"] in ("takeoff", "policy", "land")
        for k, r in enumerate(drones):
            if not flying or st["landed"][k]:
                cut_thrust(r)
                continue
            accel = torch.tensor(args.track_gain * (v_cmd[k] - state[k][1]), device=sim.device, dtype=torch.float32)
            set_accel_force(r, gravity, _clip_norm(accel, args.max_accel))

        for robot in robots.values():
            robot.write_data_to_sim()
        sim.step()
        steps += 1
        st["t"] += DT
        for robot in robots.values():
            robot.update(sim.get_physics_dt())
        for name in MOVE_TEST_DRONES:
            if not torch.isfinite(robots[name].data.root_pose_w.torch).all():
                raise RuntimeError(f"{name}: non-finite state at step {steps}")
        if MARKERS is not None and steps % 4 == 0:
            marks = [pos_vel(r)[0] + np.array((0.0, 0.0, 0.12)) for r in drones]
            marks += [s0 + np.array((0.0, 0.0, 0.02)) for s0 in starts]  # takeoff spots
            ahead = np.array((np.cos(MOVE_TEST_YAW), np.sin(MOVE_TEST_YAW), 0.0)) * move_test.FORWARD_M
            marks += [s0 + ahead + np.array((0.0, 0.0, MOVE_TAKEOFF_Z)) for s0 in starts]  # 1 m forward points
            idx = [0] * len(drones) + [3] * (2 * len(drones))
            try:
                MARKERS.visualize(translations=torch.tensor(np.array(marks), dtype=torch.float32, device=sim_device()),
                                  marker_indices=torch.tensor(idx))
            except Exception as exc:
                print(f"[WARN] marker update failed: {exc}", flush=True)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_launcher_args(parser)
    parser.add_argument("--steps", type=int, default=8000, help="Physics steps (dt 5 ms); 0 runs until interrupted.")
    parser.add_argument("--attack-mode", action="store_true", help="The spare defender rams instead of keeping goal.")
    parser.add_argument("--from-policy", action="store_true",
                        help="Fly the defenders from dist/policy.py exactly as on the real drones (poses -> bus -> rc -> Tello model).")
    parser.add_argument("--max-speed", type=float, default=TELLO_MAX_XY, help="Defender speed cap (m/s); default = Tello full stick.")
    parser.add_argument("--track-gain", type=float, default=DEFAULT_TRACK_GAIN, help="Velocity tracking gain (1/s).")
    parser.add_argument("--max-accel", type=float, default=DEFAULT_MAX_ACCEL, help="Max commanded acceleration (m/s^2).")
    parser.add_argument("--check-finished", action="store_true", help="Fail unless every attacker is down or scored in the first episode.")
    parser.add_argument("--check-separation", action="store_true", help="Fail if live defenders come within --min-separation.")
    parser.add_argument("--min-separation", type=float, default=0.4, help="Separation check threshold (m).")
    parser.add_argument("--move-test", action="store_true",
                        help="Preview deploy/swarm_ops/move_test.py: our three drones fly the movetest sequence, in a loop.")
    parser.add_argument("--viser-port", type=int, default=None,
                        help="Serve the Viser viewer on this port instead of 8080 (use with --viz viser).")
    parser.add_argument("--record", type=str, default=None, help="Write trajectories (30 fps JSON) to this path.")
    parser.add_argument("--record-episodes", type=int, default=3, help="Episodes to record with --record.")
    parser.add_argument("--auto-reset", action=argparse.BooleanOptionalAction, default=True,
                        help="Start a new episode once the last one is decided.")
    args = parser.parse_args()
    if args.steps < 0:
        parser.error("--steps must be nonnegative")
    for name in ("max_speed", "track_gain", "max_accel", "min_separation"):
        if not 0 < getattr(args, name) < float("inf"):
            parser.error(f"--{name.replace('_', '-')} must be positive and finite")
    if (args.check_finished or args.check_separation) and args.steps == 0:
        parser.error("checks require a finite --steps value")

    cfg = BlockerConfig(max_speed=args.max_speed, defender_speed=args.max_speed, gain=1.5)
    sim_cfg = sim_utils.SimulationCfg(dt=DT, render_interval=2, device=args.device, physics=NewtonCfg())  # viewer at 50 Hz
    if args.viser_port:
        from isaaclab_visualizers.viser import ViserVisualizerCfg
        sim_cfg.visualizer_cfgs = [ViserVisualizerCfg(port=args.viser_port)]
    with launch_simulation(sim_cfg, args):
        sim = sim_utils.SimulationContext(sim_cfg)
        if args.move_test:
            sim.set_camera_view(eye=[4.5, -2.5, 2.5], target=[2.0, 0.6, 0.8])
        else:
            sim.set_camera_view(eye=[7.5, -3.0, 3.5], target=[2.0, 1.2, 1.0])
        prepare_tello_usds()
        scene = instantiate(TeamDefenseSceneCfg(num_envs=1, env_spacing=12.0))
        robots = {name: scene[name] for name in DEFENDER_NAMES + ATTACKER_NAMES}
        global MARKERS, DRONE_COLORS
        if args.move_test:
            MARKERS = make_markers()  # takeoff spots / forward points for the move-test preview
        sim.reset()
        DRONE_COLORS = DroneColors()
        try:
            spawn_map_tags()
        except Exception as exc:  # cosmetic; the game does not depend on it
            print(f"[WARN] map tags not drawn: {exc}", flush=True)
        print("[INFO] Scene ready (3v2 Tellos: blue defenders, red scripted attackers).", flush=True)
        if args.move_test:
            run_move_test(sim, robots, args)
        else:
            run_scene(sim, robots, args, cfg)


if __name__ == "__main__":
    main()
