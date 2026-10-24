#!/usr/bin/env python3
"""Synthetic 8 s drone-defense recording in the render_blender.py schema (no sim needed).

3 defenders (keeper + 2 blockers), 2 attackers weaving in from the tag-10 wall.
Blocker d1 rams attacker a0 at ~3.4 s (both go down and fall); a1 slips past and scores.

    python3 scripts/make_test_recording.py --out rec_test.json
"""
import argparse
import json
import math

FPS = 30
DUR = 8.0
GOAL = [5.4, 0.0, 1.22]
ROOM = {"wall_x": 0.0, "room_x": 6.0, "room_y": [-2.0, 3.7], "ceiling": 2.74, "tag10_height": 1.22}


def quat_from_euler(roll, pitch, yaw):
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    return [cr * cp * cy + sr * sp * sy, sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy]


def lerp(a, b, u):
    return [x + (y - x) * u for x, y in zip(a, b)]


def smooth(u):
    u = min(max(u, 0.0), 1.0)
    return u * u * (3 - 2 * u)


def attacker_path(t, y0, phase, speed=0.62, x0=0.35):
    x = x0 + speed * t
    y = y0 + 0.45 * math.sin(1.7 * t + phase) * min(1.0, t / 1.0)
    z = 1.15 + 0.15 * math.sin(1.1 * t + phase)
    return [x, y, z]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="rec_test.json")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    n = int(DUR * FPS)
    t_ram = 3.4
    t_score = None
    frames, events = [], []
    prev = {}
    fall = {}  # name -> (t0, pos0, vel0, spin)
    for i in range(n):
        t = i / FPS
        st = {}
        # attackers
        a0 = attacker_path(t, -0.4, 0.0)
        a1 = attacker_path(t, 1.3, 1.9, speed=0.66)
        a1 = [a1[0], a1[1] - 0.9 * smooth((t - 3.0) / 3.5), a1[2]]  # converge toward goal
        if a1[0] > GOAL[0]:
            a1[0] = GOAL[0]
        # defenders
        keeper = [5.0, 0.25 * math.sin(0.9 * t), 1.22 + 0.05 * math.sin(2.3 * t)]
        if t > 5.5:  # keeper slides toward a1 but is too late
            keeper = lerp(keeper, [5.0, a1[1] - 0.35, a1[2]], 0.6 * smooth((t - 5.5) / 1.0))
        d1_hold = [3.4, -0.6, 1.2]
        d1 = lerp(d1_hold, a0, smooth((t - 1.8) / (t_ram - 1.8))) if t < t_ram else None
        d2 = [3.6, 1.0 + 0.5 * math.sin(0.6 * t), 1.25]
        pos = {"a0": a0, "a1": a1, "d0": keeper, "d1": d1 if d1 else prev.get("d1"), "d2": d2}
        teams = {"a0": "attacker", "a1": "attacker", "d0": "defender", "d1": "defender", "d2": "defender"}
        roles = {"a0": "attack", "a1": "attack", "d0": "keeper", "d1": "block", "d2": "block"}
        if 1.8 <= t < t_ram:
            roles["d1"] = "intercept"
        if t >= t_ram and "a0" not in fall:
            mid = [(a + b) / 2 for a, b in zip(prev["a0"], prev["d1"])]
            events.append({"t": round(t, 4), "type": "ram", "who": ["d1", "a0"], "text": "d1 rams a0",
                           "pos": mid})
            events.append({"t": round(t, 4), "type": "down", "who": ["d1", "a0"], "text": "a0 and d1 down"})
            fall["a0"] = (t, prev["a0"], [0.5, 0.25, 0.4], 7.0)
            fall["d1"] = (t, prev["d1"], [-0.35, -0.2, 0.3], -6.0)
        if a1[0] >= GOAL[0] - 0.05 and t_score is None:
            t_score = t
            events.append({"t": round(t, 4), "type": "score", "who": ["a1"], "text": "a1 scores"})
        drones = {}
        for name in ["d0", "d1", "d2", "a0", "a1"]:
            alive = name not in fall
            if not alive:
                t0, p0, v0, spin = fall[name]
                dt = t - t0
                z = p0[2] + v0[2] * dt - 4.9 * dt * dt
                land = z <= 0.02
                if land:  # time of floor contact
                    a, b, c = -4.9, v0[2], p0[2] - 0.02
                    dt = (-b - math.sqrt(b * b - 4 * a * c)) / (2 * a)
                p = [p0[0] + v0[0] * dt, p0[1] + v0[1] * dt, max(0.02, p0[2] + v0[2] * dt - 4.9 * dt * dt)]
                q = quat_from_euler(min(1.0, dt) * spin * 0.4, min(1.0, dt) * spin * 0.15, spin * dt * 0.5)
                if land:
                    q = quat_from_euler(0.0, 0.0, spin * dt * 0.5)
                drones[name] = {"team": teams[name], "pos": p, "quat_wxyz": q, "alive": False, "role": "out"}
                pos[name] = p
                continue
            p = pos[name]
            pp = prev.get(name, p)
            vx, vy = (p[0] - pp[0]) * FPS, (p[1] - pp[1]) * FPS
            sp = math.hypot(vx, vy)
            yaw = math.atan2(vy, vx) if sp > 0.15 else (0.0 if teams[name] == "attacker" else math.pi)
            tilt = min(0.25, 0.25 * sp)
            q = quat_from_euler(0.0, tilt, yaw)
            if t_score is not None and name == "a1":
                roles[name] = "scored"
            drones[name] = {"team": teams[name], "pos": [round(c, 4) for c in p],
                            "quat_wxyz": [round(c, 5) for c in q], "alive": True, "role": roles[name]}
        frames.append({"t": round(t, 4), "drones": drones})
        prev = {k: v for k, v in pos.items() if v is not None}

    rec = {"fps": FPS, "room": ROOM, "goal": GOAL,
           "episodes": [{"seed": args.seed, "frames": frames, "events": events}]}
    with open(args.out, "w") as f:
        json.dump(rec, f)
    print(f"wrote {args.out}: {n} frames, events={[(e['type'], e['t']) for e in events]}")


if __name__ == "__main__":
    main()
