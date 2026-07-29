# Stage 3–6: Exact starter workflow

This pack begins the practical MuJoCo stage. It deliberately uses a simple planar MJCF model rather than automatically converting the URDF. The geometry and mass values are copied from the current URDF assumptions:
- hip spacing: 0.105 m
- thigh/shank pitch-axis distance: 0.100 m each
- foot: 0.145 × 0.085 × 0.006 m
- hip plate: 0.170 × 0.092 × 0.005 m
- electronics mass: 0.2583 kg
- six motor masses: 0.055 kg each

## Install into the WSL project

1. Download `biped_stage3_mujoco_starter.zip` in Windows.
2. In WSL, run these commands exactly (replace `<WINDOWS_USER>` once):

```bash
cd ~/biped_fddp_project
mkdir -p model config simulation results/logs results/figures
cp /mnt/c/Users/<WINDOWS_USER>/Downloads/biped_stage3_mujoco_starter.zip ~/
cd ~
unzip -o biped_stage3_mujoco_starter.zip -d biped_stage3_mujoco_starter
cp -r biped_stage3_mujoco_starter/model/. ~/biped_fddp_project/model/
cp -r biped_stage3_mujoco_starter/config/. ~/biped_fddp_project/config/
cp -r biped_stage3_mujoco_starter/simulation/. ~/biped_fddp_project/simulation/
```

3. Verify:

```bash
cd ~/biped_fddp_project
find model config simulation -maxdepth 2 -type f | sort
```

## Run in sequence

Activate your environment first:

```bash
conda activate fddp_biped
cd ~/biped_fddp_project
```

Check MuJoCo:

```bash
python -c "import mujoco; print(mujoco.__version__)"
```

Then run exactly:

```bash
python simulation/03_check_mujoco_model.py
python simulation/02_view_model.py
python simulation/04_passive_drop.py
python simulation/05_standing_position_control.py
python simulation/06_plot_standing_log.py
```

Expected outputs:
- `MODEL LOAD CHECK PASSED`
- `results/logs/passive_drop.csv`
- `results/logs/standing_position_control.csv`
- `results/figures/standing_torso_pitch.png`

## Do not change yet
- Do not touch FDDP, MPC, contact-phase logic or actuator delay yet.
- Do not tune gains before the XML model loads and the passive/standing logs are generated.
- `biped_planar.xml` is the **simulation model**. Keep your original `biped_v1.urdf` unchanged as the CAD/dynamics source.
