#!/usr/bin/env python3
"""
Camera feed with AprilTags in -> drone pose out: xyz (m) + quaternion (x, y, z, w) in the arena frame.

  python localize_tags.py --selftest                      # no hardware: checks the math on the map
  python localize_tags.py --source tello --show           # live Tello (laptop on TELLO-xxxx Wi-Fi)
  python localize_tags.py --source 0 --show               # webcam
  python localize_tags.py --source clip.mp4 --show        # recorded Tello video
  python localize_tags.py --source tello --udp 255.255.255.255:5005 --id def1   # also broadcast

One JSON line per accepted frame on stdout (and per UDP datagram with --udp):
  {"type":"pose","id":"def1","seq":42,"t":1729771234.12,"xyz":[2.51,0.03,1.10],
   "quat_xyzw":[0,0,1,0],"yaw_deg":180.0,"tags":[10,4],"reproj_px":0.6}
"""
import argparse
import json
import os
import socket
import sys
import time

import cv2
import numpy as np

from localization import (load_tag_map, check_map, load_camera, Localizer, TagDetector,
                          tag_object_points, quat_to_yaw_deg, R_BODY_CAM)

HERE = os.path.dirname(os.path.abspath(__file__))


def local_path(p):
    return p if os.path.exists(p) or not os.path.exists(os.path.join(HERE, p)) else os.path.join(HERE, p)


def render_detections(info, K, dist, p, yaw_deg, noise_px=0.5, rng=None):
    """Synthetic tag corners seen from a known pose (for --selftest)."""
    c, s = np.cos(np.radians(yaw_deg)), np.sin(np.radians(yaw_deg))
    R_wc = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]) @ R_BODY_CAM
    R_cw, t_cw = R_wc.T, -R_wc.T @ p
    rvec, _ = cv2.Rodrigues(R_cw)
    rng = rng or np.random.default_rng(0)
    out = []
    for tid, d in info.items():
        R = np.column_stack([np.cross(d["up"], d["normal"]), d["up"], d["normal"]])
        w = (R @ tag_object_points(d["size"]).T).T + d["center"]
        if ((R_cw @ w.T).T + t_cw)[:, 2].min() < 0.15 or np.dot(d["normal"], p - d["center"]) <= 0:
            continue
        img, _ = cv2.projectPoints(w, rvec, t_cw, K, dist)
        img = img.reshape(-1, 2) + rng.normal(0, noise_px, (4, 2))
        if (img >= 0).all() and (img[:, 0] < 960).all() and (img[:, 1] < 720).all():
            out.append((tid, img))
    return out


def selftest(a):
    corners, info, _ = load_tag_map(a.tag_map)
    K, dist, _ = load_camera(a.camera)
    rng = np.random.default_rng(0)
    worst = 0.0
    print(f"map {a.tag_map}: tags {sorted(info)}")
    for p, yaw in (([2.5, 0.0, 1.2], 180), ([2.0, 0.5, 1.0], -168), ([3.0, -0.6, 1.4], -170), ([1.6, 0.2, 1.1], 180)):
        p = np.array(p, float)
        loc = Localizer(corners, K, dist)
        seen = []
        for k in range(15):                                  # a short burst of frames
            dets = render_detections(info, K, dist, p, yaw, rng=rng)
            seen = [d[0] for d in dets]
            loc.update(dets, None, k / 30.0)
        if loc.xyz is None:
            print(f"  true {p.tolist()} yaw {yaw}: NO POSE (tags in view {seen})")
            worst = np.inf
            continue
        err = np.linalg.norm(loc.xyz - p) * 100
        yerr = abs((loc.yaw - yaw + 180) % 360 - 180)
        worst = max(worst, err)
        print(f"  true {p.tolist()} yaw {yaw:+4d}: est {np.round(loc.xyz, 3).tolist()} yaw {loc.yaw:+6.1f} | "
              f"err {err:.1f} cm, {yerr:.1f} deg | tags {loc.last['tags']}")
    print("SELFTEST", "PASS" if worst < 5.0 else "FAIL", f"(worst {worst:.1f} cm)")
    return worst < 5.0


def run(a):
    corners, info, _ = load_tag_map(a.tag_map)
    if not check_map(info):
        sys.exit(1)
    K, dist, (cw, ch) = load_camera(a.camera)
    loc = Localizer(corners, K, dist, a.cam_pitch_deg, a.max_reproj)
    det = TagDetector(detect_width=a.detect_width)
    sock = dest = None
    if a.udp:
        host, port = a.udp.rsplit(":", 1)
        sock, dest = socket.socket(socket.AF_INET, socket.SOCK_DGRAM), (host, int(port))
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)

    tello = None
    if a.source == "tello":
        from djitellopy import Tello
        tello = Tello(host=a.tello_ip); tello.connect(); tello.streamon()
        print(f"[tello] battery {tello.get_battery()}%", file=sys.stderr)
        reader = tello.get_frame_read()
    else:
        cap = cv2.VideoCapture(int(a.source) if a.source.isdigit() else a.source)
    seq, last = 0, None
    try:
        while True:
            if tello is not None:
                frame = reader.frame                       # RGB
                if frame is None or frame is last:
                    time.sleep(0.003); continue
                last = frame
                imu = None
                try:
                    imu = float(tello.get_yaw())           # resolves single-tag mirror poses
                except Exception:
                    pass
            else:
                ok, bgr = cap.read()
                if not ok:
                    break
                frame, imu = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), None
            if frame.shape[1] != cw:                       # intrinsics are for cw x ch
                frame = cv2.resize(frame, (cw, ch))
            t = time.time()
            dets = det(frame)
            if loc.update(dets, imu, t):
                msg = {"type": "pose", "id": a.id, "seq": seq, "t": round(t, 3),
                       "xyz": [round(float(v), 4) for v in loc.xyz],
                       "quat_xyzw": [round(float(v), 5) for v in loc.quat],
                       "yaw_deg": round(float(loc.yaw), 1), "tags": loc.last["tags"],
                       "reproj_px": round(loc.last["reproj_px"], 2)}
                print(json.dumps(msg), flush=True)
                if sock:
                    sock.sendto(json.dumps(msg).encode(), dest)
                seq += 1
            elif dets and a.verbose:
                print(f"# saw tags {[d[0] for d in dets]} but no pose", file=sys.stderr)
            if a.show:
                vis = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                for tid, c in dets:
                    col = (0, 255, 0) if tid in corners else (0, 200, 255)
                    cv2.polylines(vis, [c.astype(int).reshape(-1, 1, 2)], True, col, 2)
                    for i, pt in enumerate(c.astype(int)):       # corner order check: 0-3 = bl, br, tr, tl
                        cv2.putText(vis, str(i), tuple(pt), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
                    cv2.putText(vis, f"id {tid}", tuple(c.mean(axis=0).astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2)
                fresh = loc.xyz is not None and loc.age(t) < 0.5
                txt = (f"x {loc.xyz[0]:+.2f}  y {loc.xyz[1]:+.2f}  z {loc.xyz[2]:+.2f}  yaw {loc.yaw:+.0f}"
                       if fresh else "NO POSE")
                cv2.putText(vis, txt + "   q quits", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                cv2.imshow("localize_tags", vis)
                if (cv2.waitKey(1) & 0xFF) == ord("q"):
                    break
    except KeyboardInterrupt:
        pass
    finally:
        if tello is not None:
            for fn in (reader.stop, tello.streamoff, tello.end):
                try:
                    fn()
                except BaseException:
                    pass
        cv2.destroyAllWindows()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default="tello", help="'tello', webcam index, or video file")
    ap.add_argument("--tello-ip", default="192.168.10.1")
    ap.add_argument("--tag-map", default="tag_map.json")
    ap.add_argument("--camera", default="camera_tello.json")
    ap.add_argument("--cam-pitch-deg", type=float, default=0.0, help="camera tilted down by this much")
    ap.add_argument("--max-reproj", type=float, default=4.0, help="reject poses above this RMS pixel error")
    ap.add_argument("--detect-width", type=int, default=640, help="downscale for detection (0 = full res)")
    ap.add_argument("--id", default="def1", help="drone id in the pose messages")
    ap.add_argument("--udp", default="", help='also send each pose to host:port, e.g. "255.255.255.255:5005"')
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    a.tag_map, a.camera = local_path(a.tag_map), local_path(a.camera)
    if a.selftest:
        sys.exit(0 if selftest(a) else 1)
    run(a)
