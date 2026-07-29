#!/usr/bin/env python3
"""Render the two-step FDDP C1 pitch-disturbance replay as a MuJoCo GIF."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import mujoco
import numpy as np
import pandas as pd
import yaml
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "figures" / "fddp_two_step_c1_pitch_disturbance.gif"


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    c1 = load(ROOT / "mpc" / "16_validate_fddp_two_step_c1_fir_qp.py", "gif_c1")
    common = load(ROOT / "mpc" / "07_run_reference_aligned_fir_qp_mpc_sweep.py", "gif_firqp")
    proxy = c1._load_proxy()
    t_ref, q_ref = c1.load_two_step_reference()
    fir = yaml.safe_load(c1.FIR_YAML.read_text(encoding="utf-8"))
    ar = yaml.safe_load(c1.AR_YAML.read_text(encoding="utf-8"))
    controller = common.FIRQP(
        np.asarray(fir["fir_coefficients_rad_per_rad"], dtype=float),
        np.asarray(ar["coefficients"], dtype=float),
        common.CANDIDATES[1],
    )
    baseline = pd.read_csv(c1.BASELINE_CSV)
    model = mujoco.MjModel.from_xml_path(str(ROOT / "model" / "biped_planar.xml"))
    data = mujoco.MjData(model)
    mapping = proxy.resolve_joint_and_actuator_mapping(model)
    torso = proxy.resolve_torso_body(model)
    pitch_joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "root_pitch_x")
    qadr = int(model.jnt_qposadr[pitch_joint])
    dadr = int(model.jnt_dofadr[pitch_joint])
    for address, value in zip(mapping["qpos_addresses"], q_ref[0]):
        data.qpos[address] = value
    mujoco.mj_forward(model, data)
    actuator = proxy.PositionActuatorProxy(6, float(model.opt.timestep))
    actuator.reset(q_ref[0])
    renderer = mujoco.Renderer(model, height=360, width=480)
    update_steps = max(1, int(round(common.TS / model.opt.timestep)))
    frames: list[Image.Image] = []
    alpha = 0.0
    applied = False
    step = 0
    next_frame = 0.0
    fps = 25.0
    duration = float(t_ref[-1])
    while data.time <= duration:
        now = min(float(data.time), duration)
        if not applied and now >= c1.DISTURBANCE_TIME_S:
            data.qpos[qadr] += c1.DISTURBANCE_PITCH_RAD
            data.qvel[dadr] += c1.DISTURBANCE_RATE_RAD_S
            mujoco.mj_forward(model, data)
            applied = True
        pitch_ref = float(np.interp(now, baseline["time_s"], baseline["root_pitch_rad"]))
        measured = float(proxy.torso_pitch_about_x(data, torso))
        if step % update_steps == 0:
            alpha = controller.update(measured - pitch_ref)
        command = proxy.interpolate_reference(now, t_ref, q_ref) + c1.G * alpha
        proxied = actuator.update(command)
        data.ctrl[:] = 0.0
        for aid, value in zip(mapping["actuator_ids"], proxied):
            data.ctrl[aid] = value
        mujoco.mj_step(model, data)
        step += 1
        if data.time + 1e-12 >= next_frame:
            renderer.update_scene(data)
            image = Image.fromarray(renderer.render())
            draw = ImageDraw.Draw(image)
            draw.text((10, 10), f"t={data.time:.2f}s  C1 alpha={np.rad2deg(alpha):+.2f} deg", fill="white")
            if applied:
                draw.text((10, 28), "pitch disturbance applied", fill="yellow")
            frames.append(image)
            next_frame += 1.0 / fps
    renderer.close()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(OUT, save_all=True, append_images=frames[1:], duration=int(1000 / fps), loop=0)
    print(OUT)


if __name__ == "__main__":
    main()
