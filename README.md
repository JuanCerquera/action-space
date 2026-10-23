# Action Space — site defense (Isaac Lab)

Bite-sized build. **No learning yet.**

## Step 1 — scene only

One script: ground, green protected site, defender + attacker Crazyflies (hovering).

On the workstation with Isaac Lab:

```powershell
.\isaaclab.bat -p C:\path\to\action-space\scripts\01_scene.py
```

You should see a green disk (site), one drone near it, one farther away. Both hover.

Tweak layout at the top of `scripts/01_scene.py` (`SITE_RADIUS`, `DEFENDER_POS`, `ATTACKER_POS`).

## Next (not done yet)

2. Scripted attacker flies toward the site  
3. Tag / breach checks (print when it happens)  
4. Only then: gym env + PPO  
