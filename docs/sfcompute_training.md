# Training the 3v2 defense policy on SF Compute (H100)

> **Status (2026-10-24, 3:45 AM): don't pay for a long run yet.** On the laptop the scripted block
> controller wins ~61% of the randomized game. PPO from scratch was at 0% after 300 iterations, and the
> residual policy, even started at exactly the block controller (58%), drifted down to ~34% within 200
> iterations. More GPU would only do the same thing faster. Fixes to try first, each a short laptop run:
> freeze the actor while the critic warms up (~50 iterations), lower the learning rate to 1e-4,
> shrink `residual_scale` to 0.2, and add a terminal bonus for holding the goal so the reward tracks
> win rate. Use the steps below once a 300-iteration laptop run ends above the block controller.

What gets trained: `Intercept-Defense-3v2-Residual-v0`. The scripted block controller
(`tello/controllers.py`) flies the defenders and PPO learns corrections on top of it, up to half of
full speed. On the laptop, PPO from scratch was still at 0% wins after 300 iterations, while the
residual policy starts near the controller's ~60% and only has to learn improvements.

Every round is randomized under the same rules (attacker tactics, speed and dodging skill, defender
lag, drift, sensor delay, random defender failures), so the policy needs a lot of experience. The
laptop (RTX 4050, 6 GB) runs 512 arenas at once; an H100 runs 4096, about 8x the data per iteration.

Only physics/logic runs here, no rendering: the H100 has no RT cores, so always pass `--headless`
and never `--enable_cameras` or `--video`.

## 0. On the laptop: push the code

The node needs the repo. From Anaconda Prompt:

```
cd C:\action_space
git add -A
git commit -m "3v2 defense env, block controller, AprilTag localizer, RealSense tracker"
git branch -M main
git remote add origin https://github.com/<you>/action_space.git
git push -u origin main
```

(`logs/` holds checkpoints, about 100 MB; leave it out of the push if you like by adding `logs/` to `.gitignore`.)

## 1. Get a node and log in

1. Open autoresearch.sfcompute.com, start an **h100-1** node (billed per minute).
2. SSH in with the command the dashboard shows, then start a tmux session so the run survives a dropped connection:

```bash
tmux new -s train
```

## 2. Install Isaac Sim 5.1 + Isaac Lab 2.3.0 (about 15 minutes, once)

Same versions as the laptop, so checkpoints move between the two.

```bash
# Python 3.11 environment
sudo apt-get update && sudo apt-get install -y python3.11 python3.11-venv git cmake build-essential
python3.11 -m venv ~/isaac && source ~/isaac/bin/activate
pip install --upgrade pip

# PyTorch 2.7.0 (CUDA 12.8) + Isaac Sim 5.1
pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com
export OMNI_KIT_ACCEPT_EULA=YES

# Isaac Lab v2.3.0 with rsl_rl
git clone --branch v2.3.0 https://github.com/isaac-sim/IsaacLab.git ~/IsaacLab
cd ~/IsaacLab && ./isaaclab.sh --install rsl_rl

# our project
git clone https://github.com/<you>/action_space.git ~/action_space
cd ~/action_space && pip install -e source/action_space

# check: GPU torch and the task is registered
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
python scripts/list_envs.py | grep Intercept
```

If `pip install isaacsim` fails, use NVIDIA's container instead (`docker run --gpus all -it
nvcr.io/nvidia/isaac-lab:2.3.0`), then clone our repo inside it and run `pip install -e source/action_space`.

## 3. Five-minute smoke test (do this before paying for a long run)

```bash
cd ~/action_space
python scripts/rsl_rl/train.py --task Intercept-Defense-3v2-Residual-v0 --headless --num_envs 4096 --max_iterations 20
```

Check in the output:

- `Computation: ... steps/s`: expect a few hundred thousand on an H100. Multiply by the full run
  (3000 iterations x 48 steps x 4096 arenas = 590M steps) to get the wall time.
- No `Traceback`, and `Metrics/win_rate` is printed.

If memory is the problem (it should not be on 80 GB), drop to `--num_envs 2048`.

## 4. Full run

First make the starting checkpoint. A fresh network adds random corrections that drag the block
controller from ~60% down to ~25% wins; zeroing its output makes training start as the pure block
controller. Use the `model_0.pt` the smoke test just wrote:

```bash
ls logs/rsl_rl/intercept_defense_3v2_residual/          # the smoke-test run folder, e.g. 2026-10-24_09-00-00
python scripts/make_residual_init.py logs/rsl_rl/intercept_defense_3v2_residual/<smoke run>/model_0.pt
```

Then train from it:

```bash
python scripts/rsl_rl/train.py --task Intercept-Defense-3v2-Residual-v0 --headless --num_envs 4096     --max_iterations 3000 --resume --load_run residual_init --checkpoint model_0.pt
```

Detach from tmux with `Ctrl+b` then `d`; reattach with `tmux attach -t train`.

What to watch (printed every iteration, also in TensorBoard):

| Metric | Good sign |
|---|---|
| `Metrics/win_rate` | climbing above the scripted block controller's ~60% |
| `Metrics/breach_rate` | falling |
| `Metrics/defenders_lost` | falling toward the random-failure floor (~0.3 per round) |
| `Metrics/collisions`, `Metrics/wall_hits` | near 0 |
| `Mean episode length` | rising toward 600 steps (a full 30 s round) |

TensorBoard from the laptop: `ssh -L 6006:localhost:6006 <node>` then on the node
`tensorboard --logdir ~/action_space/logs/rsl_rl --port 6006`, and open http://localhost:6006.

If win rate falls well below ~40% and stays there after ~300 iterations, stop the run (`Ctrl+c`)
instead of paying for the rest, and send the last printed block of metrics back.

## 5. Bring the policy back

Checkpoints land in `~/action_space/logs/rsl_rl/intercept_defense_3v2_residual/<date_time>/model_<iter>.pt`
every 100 iterations. From the laptop (Anaconda Prompt or PowerShell):

```
scp -r <node>:~/action_space/logs/rsl_rl/intercept_defense_3v2_residual C:\action_space\logs\rsl_rl\
```

**Then release the node in the dashboard** (it bills per minute even when idle).

## 6. Compare and watch on the laptop

```
cd C:\action_space
python scripts\eval_controllers.py --headless --task Intercept-Defense-3v2-v0 --controllers block --episodes 2000
python scripts\eval_controllers.py --headless --task Intercept-Defense-3v2-Residual-v0 --controllers ppo --episodes 2000
python scripts\play_arena.py --task Intercept-Defense-3v2-Residual-v0 --controller ppo
```

(Benchmark `block` on the plain task: on the residual task the env already adds the block controller.)

Fly whichever wins. On the real drones the residual policy's command is
`block_controller(obs) + 0.5 * policy(obs)` (clipped to full speed), so the scripted controller stays the
fallback: drop the policy term and it flies on its own.
