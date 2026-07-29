#!/usr/bin/env python3
"""MuJoCo position-command replay for the independent FDDP right single step.

This is simulation validation only.  q_ref drives position actuators through
a delay/lag/speed-limit proxy; tau_ref is never used as a control command.
"""
from __future__ import annotations
from pathlib import Path
import importlib.util
import sys
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mujoco
import yaml

ROOT=Path(__file__).resolve().parents[1]
XML_PATH=ROOT/"model"/"biped_planar.xml"
NPZ_PATH=ROOT/"transfer"/"trajectories"/"fddp_single_step_right_swing_ref.npz"
LOG_DIR=ROOT/"results"/"logs"
FIG_DIR=ROOT/"results"/"figures"
CSV_PATH=LOG_DIR/"fddp_single_step_right_swing_mujoco_replay.csv"
YAML_PATH=LOG_DIR/"fddp_single_step_right_swing_mujoco_replay.yaml"
FIG_PATH=FIG_DIR/"fddp_single_step_right_swing_mujoco_replay.png"
MAX_JOINT_RMSE_DEG=5.0
MAX_PITCH_DEG=15.0
MIN_HEIGHT_RATIO=0.65

def _load_proxy_module():
    path=Path(__file__).with_name("03o_replay_fddp_com_shift_mujoco.py")
    spec=importlib.util.spec_from_file_location("fddp_replay_proxy",path)
    module=importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name]=module
    spec.loader.exec_module(module)
    return module

def load_reference():
    with np.load(NPZ_PATH) as raw:
        required=("t_ref","q_ref","dq_ref","com_ref_world","phase")
        missing=[name for name in required if name not in raw.files]
        if missing: raise RuntimeError(f"{NPZ_PATH} missing {missing}")
        t=np.asarray(raw["t_ref"],float).reshape(-1)
        q=np.asarray(raw["q_ref"],float)
        dq=np.asarray(raw["dq_ref"],float)
        com=np.asarray(raw["com_ref_world"],float)
        phase=np.asarray(raw["phase"]).astype(str)
    if q.shape != (t.size,6) or dq.shape != (t.size,6) or com.shape != (t.size,3):
        raise RuntimeError("single-step reference shape mismatch")
    return t,q,dq,com,phase

def main():
    proxy=_load_proxy_module()
    # The 1.2-s swing reference needs a still actuator-aware but faster proxy
    # than the slow CoM-shift setting used by Stage 7C.
    proxy.COMMAND_DELAY_S=0.004
    proxy.FIRST_ORDER_TIME_CONSTANT_S=0.008
    proxy.MAX_COMMAND_SPEED_RAD_PER_S=20.0
    t_ref,q_ref,dq_ref,com_ref,phase=load_reference()
    model=mujoco.MjModel.from_xml_path(str(XML_PATH)); data=mujoco.MjData(model)
    mapping=proxy.resolve_joint_and_actuator_mapping(model)
    torso=proxy.resolve_torso_body(model)
    mujoco.mj_resetData(model,data)
    for address,value in zip(mapping["qpos_addresses"],q_ref[0]): data.qpos[address]=value
    mujoco.mj_forward(model,data)
    initial_height=float(data.xpos[torso][2]); initial_com_y=float(proxy.total_mass_com(model,data)[1])
    actuator=proxy.PositionActuatorProxy(6,float(model.opt.timestep)); actuator.reset(q_ref[0])
    duration=float(t_ref[-1]); records=[]
    while data.time <= duration:
        ref_t=min(float(data.time),duration)
        q_cmd=proxy.interpolate_reference(ref_t,t_ref,q_ref)
        q_proxy=actuator.update(q_cmd)
        data.ctrl[:]=0.0
        for actuator_id,value in zip(mapping["actuator_ids"],q_proxy): data.ctrl[actuator_id]=value
        mujoco.mj_step(model,data)
        measured=np.asarray([data.qpos[a] for a in mapping["qpos_addresses"]],float)
        pitch=proxy.torso_pitch_about_x(data,torso); height=float(data.xpos[torso][2])
        record={"time_s":float(data.time),"reference_time_s":ref_t,"torso_height_m":height,
          "torso_pitch_deg":float(np.rad2deg(pitch)),"com_y_change_m":float(proxy.total_mass_com(model,data)[1]-initial_com_y),
          "ncon":int(data.ncon)}
        for idx,name in enumerate(proxy.JOINT_ORDER):
            record[f"{name}_ref_rad"]=float(q_cmd[idx]); record[f"{name}_meas_rad"]=float(measured[idx])
        records.append(record)
    frame=pd.DataFrame(records)
    errors=np.column_stack([frame[f"{n}_meas_rad"]-frame[f"{n}_ref_rad"] for n in proxy.JOINT_ORDER])
    rmse=np.rad2deg(np.sqrt(np.mean(errors**2,axis=0)))
    metrics={"max_joint_rmse_deg":float(np.max(rmse)),"max_abs_torso_pitch_deg":float(np.max(np.abs(frame["torso_pitch_deg"]))),
      "min_torso_height_m":float(frame["torso_height_m"].min()),"initial_torso_height_m":initial_height,
      "min_global_contact_count":int(frame["ncon"].min()),"mujoco_achieved_com_y_shift_m":float(frame["com_y_change_m"].max()),
      "fddp_com_y_shift_m":float(com_ref[:,1].max()-com_ref[0,1])}
    failures=[]
    if metrics["max_joint_rmse_deg"]>MAX_JOINT_RMSE_DEG: failures.append("joint RMSE exceeds threshold")
    if metrics["max_abs_torso_pitch_deg"]>MAX_PITCH_DEG: failures.append("torso pitch exceeds threshold")
    if metrics["min_torso_height_m"]<MIN_HEIGHT_RATIO*initial_height: failures.append("torso height dropped below threshold")
    if metrics["min_global_contact_count"]<1: failures.append("no ground contact")
    LOG_DIR.mkdir(parents=True,exist_ok=True); FIG_DIR.mkdir(parents=True,exist_ok=True)
    frame.to_csv(CSV_PATH,index=False)
    fig,axes=plt.subplots(2,1,figsize=(10,7),sharex=True)
    axes[0].plot(frame["time_s"],frame["torso_pitch_deg"],label="torso pitch (deg)")
    axes[0].plot(frame["time_s"],frame["com_y_change_m"]*1000,label="MuJoCo CoM +Y (mm)")
    axes[0].grid(); axes[0].legend()
    axes[1].plot(frame["time_s"],frame["ncon"],label="global contacts")
    axes[1].set_xlabel("time (s)"); axes[1].grid(); axes[1].legend(); fig.tight_layout(); fig.savefig(FIG_PATH,dpi=180); plt.close(fig)
    summary={"stage":"single_step_right_swing_mujoco_position_command_replay","purpose":"MuJoCo position-command replay, not hardware walking",
      "inputs":{"reference_npz":str(NPZ_PATH),"mujoco_xml":str(XML_PATH),"tau_ref_usage":"analysis_only_never_sent_to_ctrl"},
      "metrics":metrics,"joint_rmse_deg":{n:float(v) for n,v in zip(proxy.JOINT_ORDER,rmse)},
      "integrity_passed":not failures,"failures":failures,"outputs":{"csv":str(CSV_PATH),"figure":str(FIG_PATH)}}
    YAML_PATH.write_text(yaml.safe_dump(summary,sort_keys=False),encoding="utf-8")
    print(yaml.safe_dump(summary,sort_keys=False))
    return 0 if not failures else 1
if __name__=="__main__":
    raise SystemExit(main())
