#!/usr/bin/env python3
"""Two-step FDDP C1 FIR-QP validation; simulation-only."""
from __future__ import annotations
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
NPZ_PATH = ROOT / "transfer" / "trajectories" / "fddp_two_step_alternating_ref.npz"
COMMAND_REFERENCE_ADVANCE_S = 0.0


def command_reference_time(now_s: float, t_ref: np.ndarray) -> float:
    return float(np.clip(
        now_s + COMMAND_REFERENCE_ADVANCE_S, t_ref[0], t_ref[-1]
    ))


def load_two_step_reference() -> tuple[np.ndarray, np.ndarray]:
    """Load only the gated two-step q reference; tau_ref remains analysis-only."""
    with np.load(NPZ_PATH) as raw:
        required = ("t_ref", "q_ref", "tau_ref_hardware_usage")
        missing = [key for key in required if key not in raw.files]
        if missing:
            raise RuntimeError(f"{NPZ_PATH} missing {missing}")
        usage = str(np.asarray(raw["tau_ref_hardware_usage"]).item())
        if usage != "analysis_only":
            raise RuntimeError("tau_ref boundary invalid")
        t = np.asarray(raw["t_ref"], dtype=float).reshape(-1)
        q = np.asarray(raw["q_ref"], dtype=float)
    if q.shape != (t.size, 6):
        raise RuntimeError("two-step q_ref shape mismatch")
    return t, q


import importlib.util
import sys
import mujoco
import pandas as pd

BASELINE_CSV = ROOT / "results" / "logs" / "fddp_two_step_c1_baseline.csv"


def _load_proxy():
    path = ROOT / "simulation" / "03o_replay_fddp_com_shift_mujoco.py"
    spec = importlib.util.spec_from_file_location("two_step_c1_proxy", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def run_baseline() -> Path:
    """Generate same-reference MuJoCo pitch trace for two-step C1 alignment."""
    t_ref, q_ref = load_two_step_reference()
    proxy = _load_proxy()
    xml = ROOT / "model" / "biped_planar.xml"
    model = mujoco.MjModel.from_xml_path(str(xml))
    data = mujoco.MjData(model)
    mapping = proxy.resolve_joint_and_actuator_mapping(model)
    torso = proxy.resolve_torso_body(model)
    for address, value in zip(mapping["qpos_addresses"], q_ref[0]):
        data.qpos[address] = value
    mujoco.mj_forward(model, data)
    actuator = proxy.PositionActuatorProxy(6, float(model.opt.timestep))
    actuator.reset(q_ref[0])
    rows = []
    duration = float(t_ref[-1])
    while data.time <= duration:
        now = min(float(data.time), duration)
        command_time = command_reference_time(now, t_ref)
        command = proxy.interpolate_reference(command_time, t_ref, q_ref)
        proxied = actuator.update(command)
        data.ctrl[:] = 0.0
        for aid, value in zip(mapping["actuator_ids"], proxied):
            data.ctrl[aid] = value
        mujoco.mj_step(model, data)
        rows.append({
            "time_s": float(data.time),
            "root_pitch_rad": float(proxy.torso_pitch_about_x(data, torso)),
            "root_pitch_rate_rad_s": float(data.qvel[2]),
        })
    BASELINE_CSV.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(BASELINE_CSV, index=False)
    return BASELINE_CSV


FIR_YAML = ROOT / "results" / "logs" / "fddp_two_step_c1_fir_identification.yaml"
G = np.array([0.27, 0.062, -0.228, 0.27, 0.062, -0.228], dtype=float)
PULSE_TIME_S = 0.40
PULSE_WIDTH_S = 0.010
PULSE_ALPHA_RAD = np.deg2rad(1.0)
FIR_SAMPLE_S = 0.010
FIR_LENGTH = 20


def run_pulse_identification() -> Path:
    """Identify two-step pitch FIR from a small actuator-space alpha pulse."""
    t_ref, q_ref = load_two_step_reference()
    proxy = _load_proxy()
    model = mujoco.MjModel.from_xml_path(str(ROOT / "model" / "biped_planar.xml"))
    data = mujoco.MjData(model)
    mapping = proxy.resolve_joint_and_actuator_mapping(model)
    torso = proxy.resolve_torso_body(model)
    for address, value in zip(mapping["qpos_addresses"], q_ref[0]):
        data.qpos[address] = value
    mujoco.mj_forward(model, data)
    actuator = proxy.PositionActuatorProxy(6, float(model.opt.timestep))
    actuator.reset(q_ref[0])
    samples = []
    duration = min(float(t_ref[-1]), PULSE_TIME_S + FIR_LENGTH * FIR_SAMPLE_S + 0.05)
    next_sample = PULSE_TIME_S
    while data.time <= duration:
        now = min(float(data.time), duration)
        alpha = PULSE_ALPHA_RAD if PULSE_TIME_S <= now < PULSE_TIME_S + PULSE_WIDTH_S else 0.0
        command_time = command_reference_time(now, t_ref)
        command = proxy.interpolate_reference(command_time, t_ref, q_ref) + G * alpha
        proxied = actuator.update(command)
        data.ctrl[:] = 0.0
        for aid, value in zip(mapping["actuator_ids"], proxied):
            data.ctrl[aid] = value
        mujoco.mj_step(model, data)
        if data.time + 1e-12 >= next_sample:
            samples.append(float(proxy.torso_pitch_about_x(data, torso)))
            next_sample += FIR_SAMPLE_S
    baseline = pd.read_csv(BASELINE_CSV)
    sample_times = PULSE_TIME_S + np.arange(len(samples)) * FIR_SAMPLE_S
    base = np.interp(sample_times, baseline["time_s"], baseline["root_pitch_rad"])
    h = (np.asarray(samples) - base) / PULSE_ALPHA_RAD
    import yaml
    FIR_YAML.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "stage": "two_step_fddp_c1_pulse_fir_identification",
        "reference_npz": str(NPZ_PATH),
        "tau_ref_usage": "analysis_only",
        "pulse": {"time_s": PULSE_TIME_S, "width_s": PULSE_WIDTH_S,
                  "alpha_rad": float(PULSE_ALPHA_RAD), "G": [float(v) for v in G]},
        "sample_time_s": FIR_SAMPLE_S,
        "fir_coefficients_rad_per_rad": [float(v) for v in h],
    }
    FIR_YAML.write_text(yaml.safe_dump(report, sort_keys=False), encoding="utf-8")
    return FIR_YAML


DISTURBANCE_CSV = ROOT / "results" / "logs" / "fddp_two_step_c1_c0_pitch_disturbance.csv"
DISTURBANCE_TIME_S = 0.40
DISTURBANCE_PITCH_RAD = np.deg2rad(2.0)
DISTURBANCE_RATE_RAD_S = 0.25


def run_disturbed_c0() -> Path:
    """Generate same-reference C0 pitch disturbance trace for AR fitting."""
    t_ref, q_ref = load_two_step_reference()
    proxy = _load_proxy()
    model = mujoco.MjModel.from_xml_path(str(ROOT / "model" / "biped_planar.xml"))
    data = mujoco.MjData(model)
    mapping = proxy.resolve_joint_and_actuator_mapping(model)
    torso = proxy.resolve_torso_body(model)
    pitch_joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "root_pitch_x")
    if pitch_joint < 0:
        raise RuntimeError("root_pitch_x not found")
    qadr = int(model.jnt_qposadr[pitch_joint])
    dadr = int(model.jnt_dofadr[pitch_joint])
    for address, value in zip(mapping["qpos_addresses"], q_ref[0]):
        data.qpos[address] = value
    mujoco.mj_forward(model, data)
    actuator = proxy.PositionActuatorProxy(6, float(model.opt.timestep))
    actuator.reset(q_ref[0])
    applied = False
    rows = []
    duration = float(t_ref[-1])
    while data.time <= duration:
        now = min(float(data.time), duration)
        if not applied and now >= DISTURBANCE_TIME_S:
            data.qpos[qadr] += DISTURBANCE_PITCH_RAD
            data.qvel[dadr] += DISTURBANCE_RATE_RAD_S
            mujoco.mj_forward(model, data)
            applied = True
        command_time = command_reference_time(now, t_ref)
        command = proxy.interpolate_reference(command_time, t_ref, q_ref)
        proxied = actuator.update(command)
        data.ctrl[:] = 0.0
        for aid, value in zip(mapping["actuator_ids"], proxied):
            data.ctrl[aid] = value
        mujoco.mj_step(model, data)
        rows.append({"time_s": float(data.time),
                     "measured_pitch_rad": float(proxy.torso_pitch_about_x(data, torso))})
    DISTURBANCE_CSV.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(DISTURBANCE_CSV, index=False)
    return DISTURBANCE_CSV


AR_YAML = ROOT / "results" / "logs" / "fddp_two_step_c1_ar_fit.yaml"
AR_ORDER = 3


def fit_ar_model() -> Path:
    """Fit and stabilize AR(3) to two-step C0 disturbance relative to baseline."""
    base = pd.read_csv(BASELINE_CSV)
    disturbed = pd.read_csv(DISTURBANCE_CSV)
    end = min(float(base["time_s"].max()), float(disturbed["time_s"].max()))
    times = np.arange(DISTURBANCE_TIME_S, end + 1e-12, FIR_SAMPLE_S)
    y = np.interp(times, disturbed["time_s"], disturbed["measured_pitch_rad"])
    ref = np.interp(times, base["time_s"], base["root_pitch_rad"])
    error = y - ref
    X = np.asarray([error[k-AR_ORDER:k][::-1] for k in range(AR_ORDER, len(error))])
    target = error[AR_ORDER:]
    coeff = np.linalg.solve(X.T @ X + 1e-7 * np.eye(AR_ORDER), X.T @ target)
    companion = np.zeros((AR_ORDER, AR_ORDER)); companion[0] = coeff
    companion[1:, :-1] = np.eye(AR_ORDER - 1)
    radius = float(np.max(np.abs(np.linalg.eigvals(companion))))
    stabilized = False
    if radius >= 0.985:
        coeff *= 0.95 / radius
        companion[0] = coeff
        radius = float(np.max(np.abs(np.linalg.eigvals(companion))))
        stabilized = True
    predicted = np.zeros_like(error); predicted[:AR_ORDER] = error[:AR_ORDER]
    for k in range(AR_ORDER, len(error)):
        predicted[k] = float(np.dot(coeff, predicted[k-AR_ORDER:k][::-1]))
    import yaml
    report = {"stage": "two_step_fddp_c1_ar_fit", "order": AR_ORDER,
              "sample_time_s": FIR_SAMPLE_S, "coefficients": [float(v) for v in coeff],
              "spectral_radius": radius, "stabilized": stabilized,
              "rmse_deg": float(np.rad2deg(np.sqrt(np.mean((predicted-error)**2))))}
    AR_YAML.parent.mkdir(parents=True, exist_ok=True)
    AR_YAML.write_text(yaml.safe_dump(report, sort_keys=False), encoding="utf-8")
    return AR_YAML


C1_CSV = ROOT / "results" / "logs" / "fddp_two_step_c1_fir_qp.csv"
C1_YAML = ROOT / "results" / "logs" / "fddp_two_step_c1_fir_qp.yaml"
C1_FIG = ROOT / "results" / "figures" / "fddp_two_step_c0_c1_pitch.png"


def _load_firqp_module():
    path = ROOT / "mpc" / "07_run_reference_aligned_fir_qp_mpc_sweep.py"
    spec = importlib.util.spec_from_file_location("two_step_c1_firqp", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def run_c1_fir_qp() -> dict:
    """Run balanced FIR-QP C1 on the same two-step reference and disturbance."""
    import yaml, matplotlib.pyplot as plt
    fir = yaml.safe_load(FIR_YAML.read_text(encoding="utf-8"))
    ar = yaml.safe_load(AR_YAML.read_text(encoding="utf-8"))
    h = np.asarray(fir["fir_coefficients_rad_per_rad"], dtype=float)
    a = np.asarray(ar["coefficients"], dtype=float)
    common = _load_firqp_module()
    controller = common.FIRQP(h, a, common.CANDIDATES[1])
    t_ref, q_ref = load_two_step_reference()
    baseline = pd.read_csv(BASELINE_CSV)
    proxy = _load_proxy()
    model = mujoco.MjModel.from_xml_path(str(ROOT / "model" / "biped_planar.xml"))
    data = mujoco.MjData(model)
    mapping = proxy.resolve_joint_and_actuator_mapping(model)
    torso = proxy.resolve_torso_body(model)
    pitch_joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "root_pitch_x")
    qadr = int(model.jnt_qposadr[pitch_joint]); dadr = int(model.jnt_dofadr[pitch_joint])
    for address, value in zip(mapping["qpos_addresses"], q_ref[0]): data.qpos[address] = value
    mujoco.mj_forward(model, data)
    actuator = proxy.PositionActuatorProxy(6, float(model.opt.timestep)); actuator.reset(q_ref[0])
    update_steps = max(1, int(round(common.TS / model.opt.timestep)))
    applied = False; alpha = 0.0; rows = []; duration = float(t_ref[-1])
    step = 0
    while data.time <= duration:
        now = min(float(data.time), duration)
        if not applied and now >= DISTURBANCE_TIME_S:
            data.qpos[qadr] += DISTURBANCE_PITCH_RAD; data.qvel[dadr] += DISTURBANCE_RATE_RAD_S
            mujoco.mj_forward(model, data); applied = True
        pref = float(np.interp(now, baseline["time_s"], baseline["root_pitch_rad"]))
        measured = float(proxy.torso_pitch_about_x(data, torso))
        if step % update_steps == 0:
            alpha = controller.update(measured - pref)
        command_time = command_reference_time(now, t_ref)
        command = proxy.interpolate_reference(command_time, t_ref, q_ref) + G * alpha
        proxied = actuator.update(command); data.ctrl[:] = 0.0
        for aid, value in zip(mapping["actuator_ids"], proxied): data.ctrl[aid] = value
        mujoco.mj_step(model, data); step += 1
        rows.append({"time_s": float(data.time), "pitch_ref_rad": pref,
                     "pitch_measured_rad": float(proxy.torso_pitch_about_x(data, torso)),
                     "pitch_error_deg": float(np.rad2deg(proxy.torso_pitch_about_x(data, torso)-pref)),
                     "alpha_rad": float(alpha), "ncon": int(data.ncon),
                     "torso_height_m": float(data.xpos[torso][2])})
    frame = pd.DataFrame(rows); C1_CSV.parent.mkdir(parents=True, exist_ok=True); C1_FIG.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(C1_CSV, index=False)
    err = frame["pitch_error_deg"].to_numpy(); time_s = frame["time_s"].to_numpy(); mask = time_s >= DISTURBANCE_TIME_S
    metrics = {"peak_abs_error_deg": float(np.max(np.abs(err[mask]))),
               "iae_deg_s": float(np.trapz(np.abs(err[mask]), time_s[mask])),
               "max_abs_alpha_deg": float(np.rad2deg(np.abs(frame["alpha_rad"]).max())),
               "min_contacts": int(frame["ncon"].min()),
               "min_torso_height_m": float(frame["torso_height_m"].min())}
    report = {"stage": "two_step_fddp_c1_fir_qp", "candidate": common.CANDIDATES[1],
              "reference_npz": str(NPZ_PATH), "tau_ref_usage": "analysis_only",
              "metrics": metrics, "outputs": {"csv": str(C1_CSV), "figure": str(C1_FIG)}}
    C1_YAML.write_text(yaml.safe_dump(report, sort_keys=False), encoding="utf-8")
    fig, ax = plt.subplots(figsize=(9,4)); ax.plot(time_s, err, label="C1 pitch error")
    ax.axvline(DISTURBANCE_TIME_S, linestyle="--", color="k", label="disturbance")
    ax.set_xlabel("time (s)"); ax.set_ylabel("pitch error (deg)"); ax.grid(); ax.legend(); fig.tight_layout()
    fig.savefig(C1_FIG, dpi=180); plt.close(fig)
    return report


C2_CSV = ROOT / "results" / "logs" / "fddp_two_step_c2_pitch_disturbance.csv"
C2_YAML = ROOT / "results" / "logs" / "fddp_two_step_c2_pitch_disturbance.yaml"
C2_FIG = ROOT / "results" / "figures" / "fddp_two_step_c0_c1_c2_pitch.png"


def _load_phase_sync_module():
    path = ROOT / "mpc" / "08_run_contact_phase_synchronised_fir_qp_mpc.py"
    spec = importlib.util.spec_from_file_location("two_step_c2_phase_sync", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def run_c2_disturbance() -> dict:
    """Run natural-event phase-retimed C2 under the same pitch disturbance."""
    import yaml, matplotlib.pyplot as plt
    phase_mod = _load_phase_sync_module()
    c2_input = _load_external(
        ROOT / "mpc" / "14_validate_fddp_two_step_natural_contact_c2.py",
        "two_step_c2_input",
    )
    reference = c2_input.load_reference()
    scheduled = c2_input.build_reference_events(reference)
    baseline = pd.read_csv(BASELINE_CSV)
    proxy = _load_proxy()
    model = mujoco.MjModel.from_xml_path(str(ROOT / "model" / "biped_planar.xml"))
    data = mujoco.MjData(model)
    mapping = proxy.resolve_joint_and_actuator_mapping(model); torso = proxy.resolve_torso_body(model)
    pitch_joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "root_pitch_x")
    qadr = int(model.jnt_qposadr[pitch_joint]); dadr = int(model.jnt_dofadr[pitch_joint])
    for address, value in zip(mapping["qpos_addresses"], reference["q_ref"][0]): data.qpos[address] = value
    mujoco.mj_forward(model, data)
    observer = phase_mod.ContactObserver(model, data, float(model.opt.timestep))
    sync = phase_mod.ContactPhaseSynchronizer(scheduled)
    actuator = proxy.PositionActuatorProxy(6, float(model.opt.timestep)); actuator.reset(reference["q_ref"][0])
    applied = False; rows = []; duration = float(reference["t_ref"][-1])
    while data.time <= duration:
        wall = min(float(data.time), duration)
        if not applied and wall >= DISTURBANCE_TIME_S:
            data.qpos[qadr] += DISTURBANCE_PITCH_RAD; data.qvel[dadr] += DISTURBANCE_RATE_RAD_S
            mujoco.mj_forward(model, data); applied = True
        _, _, _, observed = observer.update(model, data, wall)
        sync.update(wall, observed)
        phase_t = sync.reference_time(wall, duration)
        pref = float(np.interp(phase_t, baseline["time_s"], baseline["root_pitch_rad"]))
        command = proxy.interpolate_reference(phase_t, reference["t_ref"], reference["q_ref"])
        proxied = actuator.update(command); data.ctrl[:] = 0.0
        for aid, value in zip(mapping["actuator_ids"], proxied): data.ctrl[aid] = value
        mujoco.mj_step(model, data)
        measured = float(proxy.torso_pitch_about_x(data, torso))
        rows.append({"time_s": float(data.time), "pitch_ref_rad": pref,
                     "pitch_measured_rad": measured,
                     "pitch_error_deg": float(np.rad2deg(measured-pref)),
                     "phase_offset_s": float(sync.offset_s), "ncon": int(data.ncon),
                     "torso_height_m": float(data.xpos[torso][2])})
    frame = pd.DataFrame(rows); C2_CSV.parent.mkdir(parents=True, exist_ok=True); C2_FIG.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(C2_CSV, index=False)
    time_s = frame["time_s"].to_numpy(); err = frame["pitch_error_deg"].to_numpy(); mask = time_s >= DISTURBANCE_TIME_S
    metrics = {"peak_abs_error_deg": float(np.max(np.abs(err[mask]))),
               "iae_deg_s": float(np.trapezoid(np.abs(err[mask]), time_s[mask])),
               "max_abs_phase_offset_ms": float(1000*np.max(np.abs(frame["phase_offset_s"]))),
               "matched_contact_events": int(len(sync.matched)),
               "min_contacts": int(frame["ncon"].min())}
    report = {"stage": "two_step_fddp_c2_same_disturbance", "reference_npz": str(NPZ_PATH),
              "tau_ref_usage": "analysis_only", "metrics": metrics,
              "outputs": {"csv": str(C2_CSV), "figure": str(C2_FIG)}}
    C2_YAML.write_text(yaml.safe_dump(report, sort_keys=False), encoding="utf-8")
    return report



def _load_external(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module
