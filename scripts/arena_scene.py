# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Build the hackathon arena (room, goal zone, textured AprilTags, lights) from config/arena.json.

Import only after the Isaac Sim app is launched (used by arena_twin.py and play_arena.py).
"""

import os
import sys

import cv2
import numpy as np

import isaaclab.sim as sim_utils
from pxr import Gf, Sdf, UsdGeom, UsdShade

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO)
from tello.localizer import make_tag_image, tag_axes  # noqa: E402


def spawn_textured_tag(stage, path: str, tag: dict, texture: str, black_frac: float):
    """Author a quad with the tag texture so its black square matches the tag pose and size."""
    n, u, r = tag_axes(tag)
    c = np.asarray(tag["center"], dtype=np.float64) + 0.002 * n  # sit just off the surface
    h = 0.5 * tag["size_m"] / black_frac  # half side of the full image incl. white border
    corners = [c + h * (-r - u), c + h * (r - u), c + h * (r + u), c + h * (-r + u)]  # BL, BR, TR, TL

    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr([Gf.Vec3f(*map(float, p)) for p in corners])
    mesh.CreateFaceVertexCountsAttr([4])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    mesh.CreateNormalsAttr([Gf.Vec3f(*map(float, n))] * 4)
    mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
    mesh.CreateDoubleSidedAttr(True)
    st = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex)
    st.Set([Gf.Vec2f(0, 0), Gf.Vec2f(1, 0), Gf.Vec2f(1, 1), Gf.Vec2f(0, 1)])

    mat = UsdShade.Material.Define(stage, path + "_mat")
    shader = UsdShade.Shader.Define(stage, path + "_mat/surface")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.9)
    reader = UsdShade.Shader.Define(stage, path + "_mat/st_reader")
    reader.CreateIdAttr("UsdPrimvarReader_float2")
    reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
    tex = UsdShade.Shader.Define(stage, path + "_mat/texture")
    tex.CreateIdAttr("UsdUVTexture")
    tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(texture)
    tex.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB")
    tex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(reader.ConnectableAPI(), "result")
    tex_rgb = tex.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(tex_rgb)
    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(mat)


def design_scene(arena: dict, ground: bool = True, back_wall: bool = True, wall_opacity: float = 1.0):
    """Room, goal zone, AprilTags and lights.

    Args:
        arena: The arena map (config/arena.json).
        ground: Spawn a ground plane (skip it when the env already has one).
        back_wall: Spawn the wall opposite the goal (skip it to keep the camera view open).
        wall_opacity: Opacity of the side walls, so drones near them stay visible.
    """
    stage = sim_utils.get_current_stage()
    (x0, x1), (y0, y1), (z0, z1) = arena["room"]["x"], arena["room"]["y"], arena["room"]["z"]
    t = 0.05  # wall thickness

    if ground:
        ground_cfg = sim_utils.GroundPlaneCfg(color=(0.35, 0.37, 0.42))
        ground_cfg.func("/World/ground", ground_cfg)
    walls = {
        "goal_wall": ((x0 - t / 2, (y0 + y1) / 2, z1 / 2), (t, y1 - y0, z1)),
        "back_wall": ((x1 + t / 2, (y0 + y1) / 2, z1 / 2), (t, y1 - y0, z1)),
        "left_wall": (((x0 + x1) / 2, y0 - t / 2, z1 / 2), (x1 - x0, t, z1)),
        "right_wall": (((x0 + x1) / 2, y1 + t / 2, z1 / 2), (x1 - x0, t, z1)),
    }
    if not back_wall:
        walls.pop("back_wall")
    for name, (pos, size) in walls.items():
        opacity = 1.0 if name == "goal_wall" else wall_opacity
        wall_mat = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.85, 0.85, 0.85), roughness=0.8, opacity=opacity)
        wall = sim_utils.CuboidCfg(size=size, visual_material=wall_mat)
        wall.func(f"/World/Arena/{name}", wall, translation=pos)

    zone = arena["goal"]["zone"]
    zone_size = tuple(zone[a][1] - zone[a][0] for a in "xyz")
    zone_pos = tuple((zone[a][0] + zone[a][1]) / 2 for a in "xyz")
    zone_cfg = sim_utils.CuboidCfg(
        size=zone_size, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.2, 0.2), opacity=0.15)
    )
    zone_cfg.func("/World/Arena/goal_zone", zone_cfg, translation=zone_pos)

    tex_dir = os.path.join(REPO, "media", "tags")
    os.makedirs(tex_dir, exist_ok=True)
    for tag in arena["tags"]:
        img, black_frac = make_tag_image(tag["id"])
        texture = os.path.join(tex_dir, f"tag_{tag['id']}.png")
        cv2.imwrite(texture, img)
        spawn_textured_tag(stage, f"/World/Arena/tag_{tag['id']}", tag, texture.replace("\\", "/"), black_frac)

    dome = sim_utils.DomeLightCfg(intensity=1500.0, color=(1.0, 1.0, 1.0))
    dome.func("/World/DomeLight", dome)
    distant = sim_utils.DistantLightCfg(intensity=1500.0, angle=0.5)
    distant.func("/World/DistantLight", distant, orientation=(0.92, 0.0, 0.38, 0.0))


def make_tello_usd(path: str, body_color: tuple[float, float, float], scale: float = 1.5) -> str:
    """Write a simple Tello-shaped model (team-colored body, 4 motors, 4 prop discs) to a USD file.

    Real Tello: about 98 x 93 x 41 mm with props close to the body. ``scale`` enlarges it so it reads
    at room scale. Returns ``path`` for use as a marker prototype.
    """
    from pxr import Usd

    os.makedirs(os.path.dirname(path), exist_ok=True)
    stage = Usd.Stage.CreateNew(path) if not os.path.exists(path) else Usd.Stage.Open(path)
    for prim in list(stage.GetPseudoRoot().GetChildren()):
        stage.RemovePrim(prim.GetPath())
    root = UsdGeom.Xform.Define(stage, "/Tello")
    stage.SetDefaultPrim(root.GetPrim())
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    def material(name, color, opacity=1.0):
        mat = UsdShade.Material.Define(stage, f"/Tello/Looks/{name}")
        shader = UsdShade.Shader.Define(stage, f"/Tello/Looks/{name}/surface")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.5)
        shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(opacity)
        mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        return mat

    body_mat = material("body", body_color)
    dark_mat = material("dark", (0.15, 0.16, 0.18))
    prop_mat = material("prop", (0.25, 0.26, 0.28), opacity=0.6)

    def place(prim, translate, size, mat):
        xf = UsdGeom.Xformable(prim)
        xf.AddTranslateOp().Set(Gf.Vec3d(*(scale * t for t in translate)))
        xf.AddScaleOp().Set(Gf.Vec3f(*(scale * s for s in size)))
        UsdShade.MaterialBindingAPI.Apply(prim.GetPrim()).Bind(mat)

    # body: unit cube scaled to Tello's ~60 x 45 x 35 mm core
    body = UsdGeom.Cube.Define(stage, "/Tello/body")
    body.CreateSizeAttr(1.0)
    place(body, (0.0, 0.0, 0.0), (0.06, 0.045, 0.035), body_mat)
    # front camera nub so the heading is visible
    cam = UsdGeom.Cube.Define(stage, "/Tello/camera")
    cam.CreateSizeAttr(1.0)
    place(cam, (0.032, 0.0, 0.004), (0.006, 0.014, 0.012), dark_mat)
    for k, (sx, sy) in enumerate([(1, 1), (1, -1), (-1, 1), (-1, -1)]):
        px, py = 0.036 * sx, 0.036 * sy
        motor = UsdGeom.Cylinder.Define(stage, f"/Tello/motor_{k}")
        motor.CreateRadiusAttr(1.0)
        motor.CreateHeightAttr(1.0)
        place(motor, (px, py, 0.008), (0.009, 0.009, 0.018), dark_mat)
        prop = UsdGeom.Cylinder.Define(stage, f"/Tello/prop_{k}")
        prop.CreateRadiusAttr(1.0)
        prop.CreateHeightAttr(1.0)
        place(prop, (px, py, 0.019), (0.034, 0.034, 0.002), prop_mat)
    stage.GetRootLayer().Save()
    return path
