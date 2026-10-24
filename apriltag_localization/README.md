# AprilTag localization (Tello)

Camera feed with AprilTags in → drone pose out: **xyz (m) + quaternion (x, y, z, w)** in the arena frame.
Vision only: the pose comes from the AprilTags the Tello camera sees. The Tello's IMU yaw is used
just to choose between the two mirror-image solutions a single tag can give.

## Quick start
```bash
pip install -r requirements.txt
python localize_tags.py --selftest            # no hardware; expect "SELFTEST PASS"
python localize_tags.py --source tello --show # laptop joined to the TELLO-xxxx Wi-Fi
```
Other sources: `--source 0` (webcam), `--source clip.mp4` (recorded video).
Add `--udp 255.255.255.255:5005 --id def1` to broadcast each pose on the network.

Output, one JSON line per accepted frame:
```json
{"type":"pose","id":"def1","seq":42,"t":1729771234.12,"xyz":[2.51,0.03,1.10],
 "quat_xyzw":[0,0,1,0],"yaw_deg":180.0,"tags":[10,4],"reproj_px":0.6}
```

## Arena frame
Origin = the floor point directly under the centre of tag 10. **x** = straight out from tag 10's
wall into the room, **y** = to your left with your back to that wall, **z** = up. Metres.
A drone hovering 2.5 m in front of tag 10 and facing it reads about `x 2.5, y 0, yaw 180`.

## Files
| File | What |
|---|---|
| `localization.py` | tag detection, multi-tag PnP, mirror-pose resolution, smoothing, IMU-yaw calibration |
| `localize_tags.py` | runner: camera → pose (stdout / UDP), `--show` overlay, `--selftest` |
| `tag_map.json` | tag positions in the arena frame (generated, see below) |
| `organizers_map.yaml` | the organizers' measured map (Action_Space `drone_deployment_infra/config/map.yaml`) |
| `convert_map.py` | `organizers_map.yaml` → `tag_map.json` |
| `camera_tello.json` | approximate Tello intrinsics at 960×720 |

## Tag map
`tag_map.json` was produced from the organizers' measured map (tags 10, 4, 6), with the organizers' tag sizes:
132.08 mm black square for tags 1–9 and 11, 571.5 mm for tag 10.
**Tag 10's height above the floor is assumed to be 1.20 m** (the organizers' default too). Measure it, then:
```bash
python convert_map.py --tag10-height <measured metres>
```
Tags that aren't in the map are detected but ignored. The organizers' map only has 10, 4 and 6 so far.

## How it works
1. `pupil_apriltags` detects tag36h11 corners (downscaled to 640 px wide for speed).
2. All known tags in view are solved together in one PnP problem (SQPnP + LM refine). More tags means a steadier pose.
3. With a single tag in view, IPPE gives two candidate poses. The drone is assumed close to level, and if
   both still fit, the IMU yaw (calibrated against multi-tag poses) or the previous position picks the right one.
   If it still can't tell, the frame is skipped.
4. Poses with reprojection error > 4 px are rejected. Single-frame jumps > 0.8 m are rejected. The rest are smoothed.

## Accuracy notes
- Self-test on the organizers' map: 0.2–2.5 cm position error, < 0.5° yaw, from synthetic corners with 0.5 px noise.
  Real error is larger: calibrate the camera (checkerboard, ~10 min) for the biggest gain.
- Keep the 57 cm tag 10 fully in frame (about 1.5 m or further back).
- `--show` numbers each tag's corners 0–3. They should run bottom-left → bottom-right → top-right → top-left.
  If poses look mirrored, check this first.
