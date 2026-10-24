# Action Space — site defense (Isaac Lab)

Bite-sized build. **No learning yet.** Target runtime: [SF Compute](https://docs.sfcompute.com/preview/quick-start) GPU instance.

## SF Compute setup checklist

### 1. CLI + login (laptop)

The install script is **bash/macOS/Linux only**. On Windows, run it in **WSL** or **Git Bash** — not PowerShell (`curl` there is an alias and will error on `-fsSL`).

```bash
curl -fsSL https://cli.sfcompute.com | bash
sf login
sf billing balance    # top up on the dashboard if needed
sf availability       # pick a SKU with GPUs available now
```

### 2. Pool + buy time

```bash
sf pools create --name action-space
sf orders create --pool action-space --side buy   # interactive: nodes, duration, rate, SKU
sf orders list                                    # wait until status is filled
```

### 3. SSH key via cloud-init

```bash
cat >startup.sh <<'SCRIPT'
#!/bin/bash
mkdir -p /root/.ssh
cat >>/root/.ssh/authorized_keys <<"EOF"
# paste your public key below (or generate the file so $(cat ...) expands on your machine)
EOF
SCRIPT
```

Prefer expanding your local key when writing the script:

```bash
cat >startup.sh <<SCRIPT
#!/bin/bash
mkdir -p /root/.ssh
cat >>/root/.ssh/authorized_keys <<"EOF"
$(cat ~/.ssh/id_ed25519.pub 2>/dev/null)
$(cat ~/.ssh/id_rsa.pub 2>/dev/null)
EOF
SCRIPT
```

### 4. Create instance + SSH

```bash
sf images list
sf instances create --pool action-space \
  --image sfc:image:sfcompute:public:ubuntu-24.04-cuda-13.2 \
  --cloud-init ./startup.sh

sf instances list
sf instances ssh <instance-name>
```

On the box:

```bash
nvidia-smi   # confirm GPU is visible
```

### 5. Install Isaac Lab (on the instance)

Follow the current [Isaac Lab install docs](https://docs.nvidia.com/isaaclab/isaac-lab/installation) for your Lab version (pip Sim + Lab source is the usual path). Rough shape:

```bash
# Python 3.12 venv (version must match your Isaac Sim release)
uv venv --python 3.12 --seed ~/env_isaaclab
source ~/env_isaaclab/bin/activate

# Install Isaac Sim + CUDA PyTorch per the docs for your Lab version, then:
git clone https://github.com/isaac-sim/IsaacLab.git --branch <release-you-use>
cd IsaacLab
./isaaclab.sh -i
```

First Sim launch can take a long time while extensions download. Accept the Omniverse EULA when prompted (`OMNI_KIT_ACCEPT_EULA=yes` for non-interactive).

### 6. Clone this repo + run Step 1

```bash
# private repo — use a GitHub PAT or SSH deploy key
git clone https://github.com/JuanCerquera/action-space.git
cd action-space

# headless smoke test (no GUI on the VM)
~/IsaacLab/isaaclab.sh -p scripts/01_scene.py --headless
```

If that exits cleanly with `[INFO] Scene ready` / reset logs, the scene stack works. For a live viewport later, use remote desktop / streaming; for the hackathon, headless training is the main path.

### 7. Tear down when done

```bash
sf instances list
# stop/destroy per current sf CLI docs so you stop burning pool time
```

---

## Step 1 — scene only

One script: ground, green protected site, defender + attacker Crazyflies (hovering).

```bash
# SF Compute (headless)
./isaaclab.sh -p scripts/01_scene.py --headless

# Local workstation with a display (Windows)
.\isaaclab.bat -p C:\path\to\action-space\scripts\01_scene.py
```

Tweak layout at the top of `scripts/01_scene.py` (`SITE_RADIUS`, `DEFENDER_POS`, `ATTACKER_POS`).

## Next (not done yet)

2. Scripted attacker flies toward the site  
3. Tag / breach checks (print when it happens)  
4. Only then: gym env + PPO  
