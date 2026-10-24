"""Cinematic Cycles renderer for drone-defense recordings.

Runs inside Blender (tested with 4.2 LTS):
    blender -b -P scripts/render_blender.py -- --input rec.json --episode 0 --out out.mp4 \
        [--preview] [--camera orbit|follow|wide|top] [--still FRAME] [--samples N] [--drone-scale S]

--still FRAME renders a single PNG (``--out`` should end in .png) instead of the mp4.
Tag assets (assets/render/tags/*.png + tags_world.json) are made by scripts/make_tag_assets.py.

World frame: X out of the tag-10 wall (wall at X=0), Y right when facing the wall, Z up, metres.
"""
import json
import math
import os
import sys
import argparse

import bpy
from mathutils import Matrix, Quaternion, Vector

HERE = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.join(os.path.dirname(HERE), "assets", "render")


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--out", default="out.mp4")
    ap.add_argument("--preview", action="store_true")
    ap.add_argument("--camera", default="wide", choices=["orbit", "follow", "wide", "top"])
    ap.add_argument("--still", type=int, default=None, help="render only this (0-based) frame to PNG")
    ap.add_argument("--samples", type=int, default=None)
    ap.add_argument("--drone-scale", type=float, default=1.0, help=">1 exaggerates drones for wide shots")
    ap.add_argument("--assets", default=ASSETS)
    ap.add_argument("--cpu", action="store_true")
    return ap.parse_args(argv)


# ----------------------------------------------------------------------------- materials
def new_mat(name):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    bsdf = nt.nodes["Principled BSDF"]
    return m, nt, bsdf


def mat_simple(name, color, rough=0.5, metal=0.0, coat=0.0, emit=None, emit_str=0.0, alpha=1.0, trans=0.0):
    m, nt, b = new_mat(name)
    b.inputs["Base Color"].default_value = (*color, 1)
    b.inputs["Roughness"].default_value = rough
    b.inputs["Metallic"].default_value = metal
    b.inputs["Coat Weight"].default_value = coat
    if emit is not None:
        b.inputs["Emission Color"].default_value = (*emit, 1)
        b.inputs["Emission Strength"].default_value = emit_str
    if alpha < 1.0:
        b.inputs["Alpha"].default_value = alpha
    if trans:
        b.inputs["Transmission Weight"].default_value = trans
    return m


def mat_floor():
    """Light grey polished-concrete tiles with roughness variation and subtle bump."""
    m, nt, b = new_mat("Floor")
    N, L = nt.nodes, nt.links
    tc = N.new("ShaderNodeTexCoord")
    brick = N.new("ShaderNodeTexBrick")
    brick.offset = 0.0
    brick.inputs["Scale"].default_value = 1.0 / 0.61
    brick.inputs["Mortar Size"].default_value = 0.004
    brick.inputs["Brick Width"].default_value = 1.0
    brick.inputs["Row Height"].default_value = 1.0
    brick.inputs["Color1"].default_value = (0.62, 0.62, 0.6, 1)
    brick.inputs["Color2"].default_value = (0.55, 0.555, 0.54, 1)
    brick.inputs["Mortar"].default_value = (0.25, 0.25, 0.25, 1)
    brick.inputs["Bias"].default_value = 0.0
    noise = N.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = 3.0
    noise.inputs["Detail"].default_value = 10.0
    noise.inputs["Roughness"].default_value = 0.6
    fine = N.new("ShaderNodeTexNoise")
    fine.inputs["Scale"].default_value = 180.0
    fine.inputs["Detail"].default_value = 4.0
    mix = N.new("ShaderNodeMix")
    mix.data_type = "RGBA"
    mix.blend_type = "MULTIPLY"
    mix.inputs[0].default_value = 0.35
    L.new(tc.outputs["Object"], brick.inputs["Vector"])
    L.new(tc.outputs["Object"], noise.inputs["Vector"])
    L.new(tc.outputs["Object"], fine.inputs["Vector"])
    L.new(brick.outputs["Color"], mix.inputs[6])
    L.new(noise.outputs["Color"], mix.inputs[7])
    L.new(mix.outputs[2], b.inputs["Base Color"])
    rr = N.new("ShaderNodeMapRange")
    rr.inputs["To Min"].default_value = 0.18
    rr.inputs["To Max"].default_value = 0.55
    L.new(noise.outputs["Fac"], rr.inputs["Value"])
    L.new(rr.outputs["Result"], b.inputs["Roughness"])
    bump = N.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.08
    bump.inputs["Distance"].default_value = 0.002
    L.new(fine.outputs["Fac"], bump.inputs["Height"])
    bm = N.new("ShaderNodeBump")
    bm.inputs["Strength"].default_value = 0.5
    bm.inputs["Distance"].default_value = 0.001
    L.new(brick.outputs["Fac"], bm.inputs["Height"])
    L.new(bump.outputs["Normal"], bm.inputs["Normal"])
    L.new(bm.outputs["Normal"], b.inputs["Normal"])
    b.inputs["Coat Weight"].default_value = 0.15
    return m


def mat_plaster(name, color):
    m, nt, b = new_mat(name)
    N, L = nt.nodes, nt.links
    noise = N.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = 60.0
    noise.inputs["Detail"].default_value = 8.0
    bump = N.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.06
    L.new(noise.outputs["Fac"], bump.inputs["Height"])
    L.new(bump.outputs["Normal"], b.inputs["Normal"])
    b.inputs["Base Color"].default_value = (*color, 1)
    b.inputs["Roughness"].default_value = 0.85
    return m


def mat_image(name, path):
    m, nt, b = new_mat(name)
    tex = nt.nodes.new("ShaderNodeTexImage")
    tex.image = bpy.data.images.load(path)
    tex.interpolation = "Closest"
    nt.links.new(tex.outputs["Color"], b.inputs["Base Color"])
    b.inputs["Roughness"].default_value = 0.7
    return m


# ----------------------------------------------------------------------------- geometry helpers
def link(obj, coll=None):
    (coll or bpy.context.scene.collection).objects.link(obj)
    return obj


def plane(name, size_x, size_y, mat, loc=(0, 0, 0), rot=None, coll=None):
    import bmesh
    me = bpy.data.meshes.new(name)
    bm = bmesh.new()
    hx, hy = size_x / 2, size_y / 2
    vs = [bm.verts.new(v) for v in [(-hx, -hy, 0), (hx, -hy, 0), (hx, hy, 0), (-hx, hy, 0)]]
    f = bm.faces.new(vs)
    uv = bm.loops.layers.uv.new()
    for lp, c in zip(f.loops, [(0, 0), (1, 0), (1, 1), (0, 1)]):
        lp[uv].uv = c
    bm.to_mesh(me)
    bm.free()
    me.materials.append(mat)
    ob = bpy.data.objects.new(name, me)
    ob.location = loc
    if rot is not None:
        ob.matrix_world = Matrix.Translation(Vector(loc)) @ rot.to_4x4()
    return link(ob, coll)


def add_prim(kind, name, mat=None, coll=None, **kw):
    getattr(bpy.ops.mesh, f"primitive_{kind}_add")(**kw)
    ob = bpy.context.active_object
    ob.name = name
    if mat:
        ob.data.materials.append(mat)
    if coll:
        for c in ob.users_collection:
            c.objects.unlink(ob)
        coll.objects.link(ob)
    return ob


def bevel(ob, width, segs=3):
    md = ob.modifiers.new("bevel", "BEVEL")
    md.width = width
    md.segments = segs
    md.limit_method = "NONE"
    return md


def smooth_shade(ob):
    for p in ob.data.polygons:
        p.use_smooth = True


# ----------------------------------------------------------------------------- scene parts
def build_room(room, goal):
    x0, x1 = room["wall_x"], room["room_x"]
    y0, y1 = room["room_y"]
    zc = room["ceiling"]
    cx, cy, w, d = (x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0
    coll = bpy.data.collections.new("Room")
    bpy.context.scene.collection.children.link(coll)
    plane("Floor", w + 0.02, d + 0.02, mat_floor(), (cx, cy, 0), coll=coll)
    wall = mat_plaster("WallPaint", (0.78, 0.77, 0.74))
    tagwall = mat_plaster("TagWallPaint", (0.72, 0.74, 0.76))
    R = lambda ax, ang: Matrix.Rotation(math.radians(ang), 3, ax)
    plane("Wall_tag10", d, zc, tagwall, (x0, cy, zc / 2), R("Y", 90) @ R("Z", 90), coll)
    plane("Wall_far", d, zc, wall, (x1, cy, zc / 2), R("Y", -90) @ R("Z", 90), coll)
    plane("Wall_left", w, zc, wall, (cx, y0, zc / 2), R("X", 90), coll)
    plane("Wall_right", w, zc, wall, (cx, y1, zc / 2), R("X", -90), coll)
    ceil = plane("Ceiling", w, d, mat_plaster("Ceiling", (0.85, 0.85, 0.85)), (cx, cy, zc), R("X", 180), coll)
    ceil.visible_camera = False
    # baseboards for scale/realism
    bb = mat_simple("Baseboard", (0.12, 0.12, 0.12), rough=0.4)
    for nm, loc, sc in [("bb_l", (cx, y0 + 0.006, 0.05), (w, 0.012, 0.1)),
                        ("bb_r", (cx, y1 - 0.006, 0.05), (w, 0.012, 0.1)),
                        ("bb_f", (x1 - 0.006, cy, 0.05), (0.012, d, 0.1)),
                        ("bb_t", (x0 + 0.006, cy, 0.05), (0.012, d, 0.1))]:
        ob = add_prim("cube", nm, bb, coll, size=1, location=loc)
        ob.scale = sc
    # ceiling area lights (soft) - invisible to camera
    for i, (lx, ly) in enumerate([(1.5, cy - 1.2), (1.5, cy + 1.2), (3.0, cy), (4.5, cy - 1.2), (4.5, cy + 1.2)]):
        ld = bpy.data.lights.new(f"Panel{i}", "AREA")
        ld.shape = "RECTANGLE"
        ld.size, ld.size_y = 1.2, 0.6
        ld.energy = 220
        ld.color = (1.0, 0.96, 0.9)
        lo = bpy.data.objects.new(f"Panel{i}", ld)
        lo.location = (lx, ly, zc - 0.02)
        lo.visible_camera = False
        link(lo, coll)
        # emissive fixture face so reflections look right
        fx = plane(f"Fixture{i}", 1.2, 0.6, mat_simple("Fixture", (1, 1, 1), emit=(1, 0.97, 0.92), emit_str=4.0),
                   (lx, ly, zc - 0.005), R("X", 180), coll)
        fx.visible_shadow = False
    # warm key from the goal side
    kd = bpy.data.lights.new("Key", "AREA")
    kd.size = 2.0
    kd.energy = 300
    kd.color = (1.0, 0.92, 0.82)
    ko = link(bpy.data.objects.new("Key", kd), coll)
    ko.location = (x1 - 0.3, y0 + 0.4, zc - 0.3)
    ko.rotation_euler = Vector((cx, cy, 0.5)) .__sub__(ko.location).to_track_quat("-Z", "Y").to_euler()
    ko.visible_camera = False
    # world: soft neutral ambient
    world = bpy.data.worlds.new("World")
    bpy.context.scene.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes["Background"]
    bg.inputs["Color"].default_value = (0.55, 0.58, 0.62, 1)
    bg.inputs["Strength"].default_value = 0.35
    build_goal(goal, coll)
    return coll


def build_goal(goal, coll):
    gx, gy, gz = goal
    green = mat_simple("GoalFrame", (0.05, 0.6, 0.18), rough=0.25, coat=0.6, emit=(0.1, 1.0, 0.3), emit_str=0.6)
    film = mat_simple("GoalFilm", (0.2, 0.9, 0.4), rough=0.05, alpha=0.12)
    pole = mat_simple("GoalPole", (0.08, 0.08, 0.09), rough=0.3, metal=0.9)
    r = 0.5
    ring = add_prim("torus", "GoalHoop", green, coll, major_radius=r, minor_radius=0.022,
                    major_segments=96, minor_segments=16, location=(gx, gy, gz), rotation=(0, math.radians(90), 0))
    smooth_shade(ring)
    disk = add_prim("circle", "GoalFilm", film, coll, radius=r, vertices=96, fill_type="NGON",
                    location=(gx + 0.001, gy, gz), rotation=(0, math.radians(90), 0))
    disk.visible_shadow = False
    p = add_prim("cylinder", "GoalPole", pole, coll, radius=0.018, depth=gz - r, vertices=32,
                 location=(gx, gy, (gz - r) / 2))
    smooth_shade(p)
    base = add_prim("cylinder", "GoalBase", pole, coll, radius=0.22, depth=0.025, vertices=64,
                    location=(gx, gy, 0.0125))
    bevel(base, 0.006)
    gl = bpy.data.lights.new("GoalGlow", "POINT")
    gl.energy = 5
    gl.color = (0.2, 1.0, 0.4)
    gl.shadow_soft_size = 0.4
    glo = link(bpy.data.objects.new("GoalGlow", gl), coll)
    glo.location = (gx - 0.05, gy, gz)
    return glo


def build_tags(assets, coll):
    path = os.path.join(assets, "tags_world.json")
    if not os.path.exists(path):
        print("[render] no tags_world.json, skipping tags")
        return
    tags = json.load(open(path))
    for tid, tg in tags.items():
        img = os.path.join(assets, "tags", f"tag36h11_{tid}.png")
        if not os.path.exists(img):
            continue
        t = Vector(tg["t"])
        R = Matrix(tg["R"])
        size = tg["size_black_m"] * 10.0 / 8.0  # image has a 1-cell white margin around 8-cell black square
        if t.z < 0.2:  # floor tag: lie flat face-up, keep in-plane rotation
            xa = Vector((R[0][0], R[1][0], 0.0))
            if xa.length < 1e-3:
                xa = Vector((R[0][1], R[1][1], 0.0))
            xa.normalize()
            za = Vector((0, 0, 1))
            ya = za.cross(xa)
            R = Matrix((xa, ya, za)).transposed()
            t = Vector((t.x, t.y, 0.0015))
        else:
            n = R.col[2]
            t = t + n * 0.002
        # tag image: +x right, +y up in the tag plane; plane UV (0,0) bottom-left
        plane(f"Tag{tid}", size, size, mat_image(f"TagMat{tid}", img), tuple(t), R, coll)
        # thin backing card for wall tags (paper thickness look)
        if tg["t"][2] >= 0.2:
            pass


def build_tello(name, team_mat, frame_mats, coll, scale=1.0):
    """Procedural DJI Tello look-alike, ~98 x 92.5 x 41 mm. Forward = +X (camera lens)."""
    root = bpy.data.objects.new(name, None)
    root.empty_display_size = 0.05
    link(root, coll)
    parts = []
    white, dark, lens_m, prop_m, guard_m, seam_m, _led = frame_mats

    def own(ob):
        ob.parent = root
        parts.append(ob)
        return ob

    # main body: lower dark chassis + coloured top shell
    body = own(add_prim("cube", f"{name}_chassis", dark, coll, size=1, location=(0, 0, 0.012)))
    body.scale = (0.058, 0.044, 0.018)
    bevel(body, 0.006, 4)
    shell = own(add_prim("cube", f"{name}_shell", team_mat, coll, size=1, location=(0, 0, 0.026)))
    shell.scale = (0.062, 0.047, 0.016)
    bevel(shell, 0.0075, 5)
    # battery seam (rear) + battery tab
    seam = own(add_prim("cube", f"{name}_seam", seam_m, coll, size=1, location=(-0.016, 0, 0.0262)))
    seam.scale = (0.0012, 0.0472, 0.0162)
    tab = own(add_prim("cube", f"{name}_battery", dark, coll, size=1, location=(-0.0318, 0, 0.02)))
    tab.scale = (0.003, 0.03, 0.012)
    bevel(tab, 0.0012)
    # front camera lens
    lens_ring = own(add_prim("cylinder", f"{name}_lensring", dark, coll, radius=0.0068, depth=0.004, vertices=32,
                             location=(0.0315, 0, 0.019), rotation=(0, math.radians(90), 0)))
    smooth_shade(lens_ring)
    lens = own(add_prim("uv_sphere", f"{name}_lens", lens_m, coll, radius=0.0048, segments=24, ring_count=12,
                        location=(0.0325, 0, 0.019)))
    lens.scale = (0.45, 1, 1)
    smooth_shade(lens)
    # status LED (front-left)
    led = own(add_prim("uv_sphere", f"{name}_led", frame_mats[6], coll, radius=0.0016, location=(0.031, 0.012, 0.019)))
    # arms, motors, guards, props
    props = []
    mx, my, rg = 0.0285, 0.0265, 0.0205
    for k, (sx, sy) in enumerate([(1, 1), (1, -1), (-1, 1), (-1, -1)]):
        cx, cy = sx * mx, sy * my
        arm = own(add_prim("cube", f"{name}_arm{k}", white, coll, size=1, location=(cx * 0.55, cy * 0.55, 0.012)))
        arm.scale = (math.hypot(mx, my) * 0.9, 0.008, 0.005)
        arm.rotation_euler.z = math.atan2(cy, cx)
        bevel(arm, 0.002)
        motor = own(add_prim("cylinder", f"{name}_motor{k}", dark, coll, radius=0.0052, depth=0.011, vertices=24,
                             location=(cx, cy, 0.012)))
        bevel(motor, 0.001)
        smooth_shade(motor)
        guard = own(add_prim("torus", f"{name}_guard{k}", guard_m, coll, major_radius=rg, minor_radius=0.0018,
                             major_segments=48, minor_segments=8, location=(cx, cy, 0.0135)))
        guard.scale = (1, 1, 2.6)
        smooth_shade(guard)
        # guard spokes
        for s in range(2):
            sp = own(add_prim("cube", f"{name}_spoke{k}{s}", guard_m, coll, size=1, location=(cx, cy, 0.0085)))
            sp.scale = (2 * rg, 0.0016, 0.0016)
            sp.rotation_euler.z = math.atan2(cy, cx) + (math.pi / 4 if s else -math.pi / 4)
        # two-blade prop
        prop = bpy.data.objects.new(f"{name}_prop{k}", None)
        prop.location = (cx, cy, 0.0195)
        prop.empty_display_size = 0.01
        link(prop, coll)
        own(prop)
        for bld in range(2):
            b = add_prim("cube", f"{name}_blade{k}{bld}", prop_m, coll, size=1)
            b.scale = (0.0175, 0.0042, 0.0008)
            b.location = ((1 if bld else -1) * 0.0098, 0, 0)
            b.rotation_euler = ((1 if bld else -1) * math.radians(14) * sx * sy, 0, 0)
            bevel(b, 0.0015, 2)
            b.parent = prop
            parts.append(b)
        hub = add_prim("cylinder", f"{name}_hub{k}", dark, coll, radius=0.0028, depth=0.003, vertices=16)
        hub.parent = prop
        parts.append(hub)
        props.append((prop, sx * sy))
    root.scale = (scale, scale, scale)
    return root, props, shell


def drone_materials():
    white = mat_simple("TelloWhite", (0.86, 0.86, 0.84), rough=0.35, coat=0.3)
    dark = mat_simple("TelloDark", (0.035, 0.035, 0.04), rough=0.45)
    lens = mat_simple("TelloLens", (0.01, 0.01, 0.015), rough=0.02, coat=1.0)
    prop = mat_simple("TelloProp", (0.06, 0.06, 0.065), rough=0.35)
    guard = mat_simple("TelloGuard", (0.8, 0.8, 0.79), rough=0.4)
    seam = mat_simple("TelloSeam", (0.01, 0.01, 0.01), rough=0.8)
    led = mat_simple("TelloLED", (0.1, 1.0, 0.2), emit=(0.1, 1.0, 0.2), emit_str=15)
    return [white, dark, lens, prop, guard, seam, led]


TEAM_RGB = {"defender": (0.03, 0.18, 0.75), "attacker": (0.78, 0.05, 0.03)}
DEAD_RGB = (0.07, 0.07, 0.075)


def shell_material(name, team):
    """Glossy plastic; Mix factor 0=team colour, 1=dead grey (keyframed)."""
    m, nt, b = new_mat(name)
    mix = nt.nodes.new("ShaderNodeMix")
    mix.data_type = "RGBA"
    mix.inputs[6].default_value = (*TEAM_RGB.get(team, (0.6, 0.6, 0.6)), 1)
    mix.inputs[7].default_value = (*DEAD_RGB, 1)
    nt.links.new(mix.outputs[2], b.inputs["Base Color"])
    b.inputs["Roughness"].default_value = 0.28
    b.inputs["Coat Weight"].default_value = 0.6
    b.inputs["Coat Roughness"].default_value = 0.08
    return m, mix


# ----------------------------------------------------------------------------- animation helpers
def set_fcurve(id_owner, path, index, frames, values, interp=1):
    """Bulk-insert keyframes. interp: 0 CONSTANT, 1 LINEAR."""
    if id_owner.animation_data is None:
        id_owner.animation_data_create()
    ad = id_owner.animation_data
    if ad.action is None:
        ad.action = bpy.data.actions.new(f"{id_owner.name}_act")
    fc = ad.action.fcurves.find(path, index=index) or ad.action.fcurves.new(path, index=index)
    fc.keyframe_points.add(len(frames))
    co = [c for fv in zip(frames, values) for c in fv]
    fc.keyframe_points.foreach_set("co", co)
    fc.keyframe_points.foreach_set("interpolation", [interp] * len(frames))
    fc.update()
    return fc


def frame_of(t, fps):
    return 1 + t * fps


# ----------------------------------------------------------------------------- main
def main():
    a = parse_args()
    rec = json.load(open(a.input))
    fps = int(round(rec.get("fps", 30)))
    room = rec.get("room", {"wall_x": 0.0, "room_x": 6.0, "room_y": [-2.0, 3.7], "ceiling": 2.74})
    goal = rec.get("goal", [5.4, 0.0, 1.22])
    ep = rec["episodes"][a.episode]
    frames = ep["frames"]
    events = ep.get("events", [])

    bpy.ops.wm.read_factory_settings(use_empty=True)
    sc = bpy.context.scene
    sc.render.fps = fps
    nfr = len(frames)
    sc.frame_start, sc.frame_end = 1, nfr

    coll = build_room(room, goal)
    tag_coll = bpy.data.collections.new("Tags")
    sc.collection.children.link(tag_coll)
    build_tags(a.assets, tag_coll)

    # ---- drones
    dcoll = bpy.data.collections.new("Drones")
    sc.collection.children.link(dcoll)
    fm = drone_materials()
    names = sorted({n for f in frames for n in f["drones"]})
    fr_idx = [frame_of(f["t"] - frames[0]["t"], fps) for f in frames]
    tracks = {}
    for nm in names:
        first = next(f["drones"][nm] for f in frames if nm in f["drones"])
        team = first.get("team", "defender")
        smat, mix = shell_material(f"Shell_{nm}", team)
        root, props, shell = build_tello(nm, smat, fm, dcoll, a.drone_scale)
        root.rotation_mode = "QUATERNION"
        fx, loc, rot, dead = [], [[], [], []], [[], [], [], []], []
        prev_q = None
        last = first
        for fi, f in zip(fr_idx, frames):
            st = f["drones"].get(nm, last)
            last = st
            q = Quaternion(st.get("quat_wxyz", [1, 0, 0, 0]))
            if prev_q is not None and prev_q.dot(q) < 0:
                q.negate()
            prev_q = q
            fx.append(fi)
            for i in range(3):
                loc[i].append(st["pos"][i])
            for i in range(4):
                rot[i].append(q[i])
            dead.append(0.0 if st.get("alive", True) else 1.0)
        for i in range(3):
            set_fcurve(root, "location", i, fx, loc[i])
        for i in range(4):
            set_fcurve(root, "rotation_quaternion", i, fx, rot[i])
        set_fcurve(smat.node_tree, f'nodes["{mix.name}"].inputs[0].default_value', 0, fx, dead, interp=0)
        # props: accumulate spin while alive (rad per frame), stop when dead
        ang, spin = 0.0, []
        for d in dead:
            spin.append(ang)
            ang += 0.0 if d else 2.2
        for prop, sgn in props:
            set_fcurve(prop, "rotation_euler", 2, fx, [sgn * s for s in spin])
        tracks[nm] = {"team": team, "loc": list(zip(*loc)), "dead": dead}

    # ---- event FX: sparks at ram, glow pulse on score
    fxcoll = bpy.data.collections.new("FX")
    sc.collection.children.link(fxcoll)
    spark_mat = mat_simple("Spark", (1, 0.7, 0.3), emit=(1.0, 0.65, 0.25), emit_str=0.0)
    for k, ev in enumerate(events):
        f0 = frame_of(ev["t"] - frames[0]["t"], fps)
        if ev["type"] == "ram":
            if "pos" in ev:
                p = Vector(ev["pos"])
            else:
                i = max(0, min(nfr - 1, int(round(f0 - 1)) - 1))
                pts = [Vector(tracks[w]["loc"][i]) for w in ev.get("who", []) if w in tracks]
                if not pts:
                    continue
                p = sum(pts, Vector()) / len(pts)
            ld = bpy.data.lights.new(f"RamFlash{k}", "POINT")
            ld.color = (1.0, 0.7, 0.35)
            ld.shadow_soft_size = 0.02
            lo = link(bpy.data.objects.new(f"RamFlash{k}", ld), fxcoll)
            lo.location = p
            set_fcurve(ld, "energy", 0, [f0 - 1, f0, f0 + 2, f0 + 7], [0, 60, 25, 0])
            m = spark_mat.copy()
            set_fcurve(m.node_tree, 'nodes["Principled BSDF"].inputs["Emission Strength"].default_value', 0,
                       [f0 - 1, f0, f0 + 3, f0 + 6], [0, 80, 20, 0])
            import random
            rng = random.Random(k)
            for s in range(14):
                d = Vector((rng.uniform(-1, 1), rng.uniform(-1, 1), rng.uniform(-0.3, 1))).normalized()
                sp = add_prim("uv_sphere", f"spark{k}_{s}", m, fxcoll, radius=0.004, segments=8, ring_count=4,
                              location=p)
                sp.scale = (1, 1, 1)
                sp.visible_shadow = False
                v = d * rng.uniform(0.6, 1.4)
                fs = [f0 - 1, f0, f0 + 8]
                for i in range(3):
                    set_fcurve(sp, "location", i, fs,
                               [p[i], p[i], p[i] + v[i] * 8 / fps + (-0.5 * 9.8 * (8 / fps) ** 2 if i == 2 else 0)])
                set_fcurve(sp, "scale", 0, [f0 - 1, f0, f0 + 8], [0, 1, 0.2])
                set_fcurve(sp, "scale", 1, [f0 - 1, f0, f0 + 8], [0, 1, 0.2])
                set_fcurve(sp, "scale", 2, [f0 - 1, f0, f0 + 8], [0, 3, 0.4])
        elif ev["type"] == "score":
            gl = bpy.data.lights["GoalGlow"]
            set_fcurve(gl, "energy", 0, [1, f0 - 1, f0 + 2, f0 + 30], [5, 5, 120, 15])

    # ---- camera
    cam_d = bpy.data.cameras.new("Cam")
    cam = link(bpy.data.objects.new("Cam", cam_d))
    sc.camera = cam
    tgt = link(bpy.data.objects.new("CamTarget", None))
    con = cam.constraints.new("TRACK_TO")
    con.target = tgt
    con.track_axis, con.up_axis = "TRACK_NEGATIVE_Z", "UP_Y"
    x0, x1 = room["wall_x"], room["room_x"]
    y0, y1 = room["room_y"]
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    cam_d.dof.use_dof = False
    if a.camera == "wide":
        cam.location = (x1 - 0.25, y0 + 0.25, 2.45)
        tgt.location = (2.4, cy + 0.2, 0.75)
        cam_d.lens = 16
    elif a.camera == "top":
        cam.constraints.remove(con)
        cam_d.type = "ORTHO"
        cam_d.ortho_scale = max((x1 - x0), (y1 - y0) * 16 / 9) * 1.02
        cam.location = (cx, cy, 8.0)
        cam.rotation_euler = (0, 0, math.radians(90))
        cam_d.clip_end = 20
    elif a.camera == "orbit":
        tgt.location = (cx, cy, 0.9)
        cam_d.lens = 20
        r = min(x1 - x0, y1 - y0) / 2 - 0.25
        fs = list(range(1, nfr + 1))
        th = [math.radians(-120) + 2 * math.pi * (f - 1) / (fps * 40.0) for f in fs]
        set_fcurve(cam, "location", 0, fs, [cx + r * math.cos(t) for t in th])
        set_fcurve(cam, "location", 1, fs, [cy + r * math.sin(t) for t in th])
        set_fcurve(cam, "location", 2, fs, [2.1 + 0.15 * math.sin(t * 2) for t in th])
    elif a.camera == "follow":
        cam_d.lens = 30
        cam_d.dof.use_dof = True
        cam_d.dof.aperture_fstop = 4.0
        cam_d.dof.focus_object = tgt
        gv = Vector(goal)
        att = [n for n in names if tracks[n]["team"] == "attacker"] or names
        tp, cp = [], []
        s_t = s_c = None
        for i in range(nfr):
            alive = [n for n in att if not tracks[n]["dead"][i]] or att
            n = min(alive, key=lambda n: (Vector(tracks[n]["loc"][i]) - gv).length)
            p = Vector(tracks[n]["loc"][i])
            c = p + Vector((-1.1, -0.9, 0.55))
            c.x = min(max(c.x, x0 + 0.3), x1 - 0.3)
            c.y = min(max(c.y, y0 + 0.3), y1 - 0.3)
            c.z = min(max(c.z, 0.4), room["ceiling"] - 0.3)
            s_t = p if s_t is None else s_t.lerp(p, 0.12)
            s_c = c if s_c is None else s_c.lerp(c, 0.06)
            tp.append(s_t.copy())
            cp.append(s_c.copy())
        fs = list(range(1, nfr + 1))
        for i in range(3):
            set_fcurve(tgt, "location", i, fs, [v[i] for v in tp])
            set_fcurve(cam, "location", i, fs, [v[i] for v in cp])
    cam_d.clip_start = 0.02

    # ---- render settings
    r = sc.render
    sc.render.engine = "CYCLES"
    cy_ = sc.cycles
    dev = "CPU"
    if not a.cpu:
        prefs = bpy.context.preferences.addons["cycles"].preferences
        for kind in ("OPTIX", "CUDA"):
            try:
                prefs.compute_device_type = kind
                prefs.get_devices()
                gpus = [d for d in prefs.devices if d.type == kind]
                if gpus:
                    for d in prefs.devices:
                        d.use = d.type == kind
                    dev = kind
                    break
            except TypeError:
                continue
    cy_.device = "GPU" if dev != "CPU" else "CPU"
    print(f"[render] cycles device: {dev}")
    if a.preview:
        r.resolution_x, r.resolution_y = 960, 540
        cy_.samples = a.samples or 16
    else:
        r.resolution_x, r.resolution_y = 1920, 1080
        cy_.samples = a.samples or 96
    r.resolution_percentage = 100
    cy_.use_denoising = True
    cy_.denoiser = "OPTIX" if dev == "OPTIX" else "OPENIMAGEDENOISE"
    cy_.use_adaptive_sampling = True
    cy_.max_bounces = 6
    cy_.transparent_max_bounces = 8
    cy_.caustics_reflective = cy_.caustics_refractive = False
    cy_.blur_glossy = 1.0
    r.use_motion_blur = True
    r.motion_blur_shutter = 0.5
    r.use_persistent_data = True
    r.film_transparent = False
    sc.view_settings.view_transform = "AgX"
    sc.view_settings.look = "AgX - Medium High Contrast"
    sc.view_settings.exposure = 0.0

    out = os.path.abspath(a.out)
    if a.still is not None:
        sc.frame_set(min(nfr, a.still + 1))
        r.image_settings.file_format = "PNG"
        r.filepath = out
        bpy.ops.render.render(write_still=True)
    else:
        r.image_settings.file_format = "FFMPEG"
        r.ffmpeg.format = "MPEG4"
        r.ffmpeg.codec = "H264"
        r.ffmpeg.constant_rate_factor = "HIGH" if a.preview else "PERC_LOSSLESS"
        r.ffmpeg.ffmpeg_preset = "GOOD"
        r.filepath = out
        bpy.ops.render.render(animation=True)
    print(f"[render] wrote {out}")


if __name__ == "__main__":
    main()
