#!/usr/bin/env python3
"""Render the gated FDDP two-step MuJoCo position-command replay as a GIF."""
from __future__ import annotations
from pathlib import Path
import importlib.util
import sys
import mujoco
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
XML=ROOT/"model"/"biped_planar.xml"
NPZ=ROOT/"transfer"/"trajectories"/"fddp_two_step_alternating_ref.npz"
OUT=ROOT/"results"/"figures"/"fddp_two_step_mujoco_position_command_replay.gif"

def load_proxy():
    path=ROOT/"simulation"/"03o_replay_fddp_com_shift_mujoco.py"
    spec=importlib.util.spec_from_file_location("animation_proxy",path)
    module=importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name]=module
    spec.loader.exec_module(module)
    return module

def main():
    with np.load(NPZ) as raw:
        t=np.asarray(raw["t_ref"],float); q=np.asarray(raw["q_ref"],float)
        if str(np.asarray(raw["tau_ref_hardware_usage"]).item())!="analysis_only":
            raise RuntimeError("tau_ref boundary invalid")
    proxy=load_proxy()
    model=mujoco.MjModel.from_xml_path(str(XML)); data=mujoco.MjData(model)
    mapping=proxy.resolve_joint_and_actuator_mapping(model)
    for address,value in zip(mapping["qpos_addresses"],q[0]): data.qpos[address]=value
    mujoco.mj_forward(model,data)
    actuator=proxy.PositionActuatorProxy(6,float(model.opt.timestep)); actuator.reset(q[0])
    renderer=mujoco.Renderer(model,height=360,width=480)
    frames=[]; next_sample=0.0; fps=25.0
    while data.time<=float(t[-1]):
        now=min(float(data.time),float(t[-1]))
        command=proxy.interpolate_reference(now,t,q)
        proxied=actuator.update(command)
        data.ctrl[:]=0.0
        for aid,value in zip(mapping["actuator_ids"],proxied): data.ctrl[aid]=value
        mujoco.mj_step(model,data)
        if data.time+1e-12>=next_sample:
            renderer.update_scene(data)
            frames.append(Image.fromarray(renderer.render()))
            next_sample+=1.0/fps
    renderer.close()
    OUT.parent.mkdir(parents=True,exist_ok=True)
    frames[0].save(OUT,save_all=True,append_images=frames[1:],duration=int(1000/fps),loop=0)
    print(OUT)
if __name__=="__main__":
    main()
