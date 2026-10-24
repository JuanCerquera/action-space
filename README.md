# Action Space — site defense (Isaac Lab)

Scene only, with no learning yet: a ground plane, a green protected-site marker,
and defender and attacker Crazyflies hovering at their spawn positions.

The project uses **Isaac Lab 3.0**, including its Newton physics backend. The
headless setup runs on an SF Autoresearch H100 without Isaac Sim or an RTX
renderer. Project code uses Isaac Lab's scene, asset, state, and force APIs.

## SF Autoresearch setup

Use [SF Autoresearch](https://autoresearch.sfcompute.com/dashboard) and its `autor`
CLI. The separate `sf` CLI and compute-pool order workflow are not used here.

Install the CLI on Linux/macOS (Windows: use WSL), then authenticate:

```bash
curl -fsSL https://autoresearch.sfcompute.com/cli.sh | bash
export PATH="$HOME/.local/bin:$PATH"
autor login --device-code
autor whoami
autor billing
```

Create a mission to track setup costs, then a single H100 node using NVIDIA's
prebuilt Isaac Lab image without Isaac Sim. This image already includes Python
3.12, Isaac Lab, PyTorch, and its physics dependencies; no separate simulator
installation is needed. The digest pins the image used for this project:

```bash
autor api post /missions -d '{"name":"action-space-setup","title":"Validate Isaac Lab headless drone scene"}'
autor node create --name action-space --chip h100 \
  --image nvcr.io/nvidia/isaac-lab@sha256:d47169952b55effb46dba8f79761d381532b521561ecba38a3019adc7648884b \
  --mission action-space-setup --max-wait 10m --wait
```

An existing `action-space` node can be reused. Inspect it with
`autor node get action-space`; running a command wakes it if stopped. Compute
bills while the node runs, including its idle grace window. Check the current
rate returned by the platform; `autor node stop action-space` ends compute
billing and retains files under `/home/isaaclab`.

From this repository, upload the scene into that persistent home directory:

```bash
autor cp scripts/01_scene.py action-space:~/action-space/scripts/01_scene.py
```

Run the finite GPU smoke test. The timeout also bounds the first asset download
and kernel compilation. The CLI's default synchronous timeout is only 60 s, so
raise it explicitly. The extra 100 steps exercise the reset at step 2000:

```bash
autor run action-space --timeout 300 -- timeout 280 python -u \
  /home/isaaclab/action-space/scripts/01_scene.py \
  --device cuda:0 --viz none --steps 2100 --check-hover
```

Success means exit code 0, `[INFO] Scene ready`, and a `[INFO] Scene result` JSON
record showing 2100 steps, finite state, and maximum displacement at most 0.1 m
for each drone. The check measures displacement after every step, before any
periodic reset can hide it. First launch downloads the Crazyflie USD assets and
compiles physics kernels; allow more time than a warm run.

Validated on an Autoresearch H100 on October 23, 2026: 2100 steps (10.5 simulated
seconds), exit code 0, maximum drift 0.00700 m for the defender and 0.00706 m for
the attacker. The pinned image, script hash, and measured output are recorded in
[validation/autoresearch-smoke.json](validation/autoresearch-smoke.json).

For a repeat run on an existing node, this helper uploads the script, runs the
check, and stops the node on success or failure:

```bash
./scripts/smoke_autoresearch.sh action-space
```

Stop the node when finished, including after a failed run:

```bash
autor node stop action-space
autor node get action-space
autor usage
```

Only the node's home directory persists across stops. Keep code and future
checkpoints there. Files elsewhere in the image return to their original state
on wake; Isaac Lab's default asset cache is in `/tmp`, so assets may download
again after a stop.

## Scene controls

In the Isaac Lab image, or a compatible Isaac Lab 3.0 Python environment:

```bash
python scripts/01_scene.py --device cuda:0 --viz none --steps 2000 --check-hover
```

The default is 2000 steps at 0.005 s per step (10 simulated seconds). `--steps 0`
runs until interrupted; `--check-hover` requires a finite step count.
`--hover-tolerance` sets the allowed displacement in meters (default 0.1).

Tweak `SITE_RADIUS`, `DEFENDER_POS`, and `ATTACKER_POS` at the top of
[scripts/01_scene.py](scripts/01_scene.py). The site is a visual marker. Hover
uses a constant force balancing each drone's measured total weight, with a
reset every 2000 steps; this is a scene check, not an attitude controller or a
learned policy. No visualizer is selected in the cloud smoke test.

For a local viewport on a machine with graphics support, select Isaac Lab's
OpenGL visualizer:

```bash
python scripts/01_scene.py --viz newton_gl --steps 0
```

See the pinned release's [installation instructions](https://github.com/isaac-sim/IsaacLab/blob/961760bcb91e3af48e7d116292e57e8ecaefc31e/docs/source/isaaclab/installation.rst)
for source installation and platform requirements. This scene targets the 3.0
API, rather than the old `AppLauncher` / Isaac Sim workflow.

## Step 2 — scripted attacker

Rectangular arena (4×8 m), **3v3**: blue defenders hover on −Y, red attackers
spawn on +Y and fly toward the green defense volume (PD + gravity compensation).
Crazyflies stay at **real scale** (~10 cm), three per team (blue defend / red
attack). Viser may show collision boxes rather than the full mesh:

```bash
autor cp scripts/02_scripted_attacker.py action-space:~/action-space/scripts/02_scripted_attacker.py
autor run action-space --timeout 300 -- timeout 280 python -u \
  /home/isaaclab/action-space/scripts/02_scripted_attacker.py \
  --device cuda:0 --viz none --steps 2100 --check-approach --check-defender-hover
```

For the remote Viser viewer: `--viz viser --steps 0 --no-auto-reset`, then expose
port 8080. Use the Isaac Lab sidebar: **Reset Episode** restarts the approach,
**Pause Simulation** freezes physics. Slate rectangle = arena; green cylinder =
defense volume; small boxes = Crazyflie collision proxies in Newton/Viser.

## Next

1. **Tag / breach checks** — detect when the attacker enters the defense volume
   (and later: defender “tags” the attacker). Print / flag the event.
2. Gym environment and PPO.

