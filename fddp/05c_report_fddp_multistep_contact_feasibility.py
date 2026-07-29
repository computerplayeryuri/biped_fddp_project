"""Audit every active FDDP contact before promoting the ten-step nominal."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pinocchio as pin
import yaml

from contact_feasibility import (
    assert_friction_contract,
    contact_frame_to_sole_offset,
    contact_wrench_at_sole,
    evaluate_planar_support_feasibility,
    promote_candidate_reference,
    sole_tangential_speed,
)


ROOT = Path(__file__).resolve().parents[1]
SOLVER_PATH = ROOT / "fddp" / "05b_solve_fddp_multistep_continuation.py"
CONFIG_PATH = ROOT / "config" / "fddp_multistep_experiment.yaml"
GEOMETRY_PATH = ROOT / "config" / "robot_geometry.yaml"
MJCF_PATH = ROOT / "model" / "biped_planar.xml"
CANDIDATE_PATH = (
    ROOT / "transfer" / "trajectories"
    / "fddp_multistep_10step_planning_candidate.npz"
)
CANDIDATE_SUMMARY_PATH = CANDIDATE_PATH.with_name(
    "fddp_multistep_10step_planning_candidate_summary.yaml"
)
FINAL_PATH = (
    ROOT / "transfer" / "trajectories" / "fddp_multistep_10step_ref.npz"
)
FINAL_SUMMARY_PATH = FINAL_PATH.with_suffix(".yaml")
CSV_PATH = ROOT / "results" / "logs" / "fddp_multistep_contact_feasibility.csv"
YAML_PATH = ROOT / "results" / "logs" / "fddp_multistep_contact_feasibility.yaml"
FIGURE_PATH = ROOT / "results" / "figures" / "fddp_multistep_contact_feasibility.png"


def _load_solver_module():
    name = "_fddp_contact_report_multistep_solver"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, SOLVER_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load solver module: {SOLVER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _scalar_text(value) -> str:
    return str(np.asarray(value).reshape(()).item())


def _load_inputs() -> tuple[dict, dict, dict]:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    geometry = yaml.safe_load(GEOMETRY_PATH.read_text(encoding="utf-8"))
    planning = yaml.safe_load(CANDIDATE_SUMMARY_PATH.read_text(encoding="utf-8"))
    if planning.get("planning_acceptance", {}).get("passed") is not True:
        raise RuntimeError("planning candidate summary is not accepted")
    return config, geometry, planning


def _mjcf_contract() -> dict:
    root = ET.parse(MJCF_PATH).getroot()
    foot_values = {}
    sole_offsets = {}
    half_sizes = {}
    for side in ("left", "right"):
        geom = root.find(f".//geom[@name='{side}_foot_geom']")
        if geom is None:
            raise RuntimeError(f"MJCF is missing {side}_foot_geom")
        friction = np.fromstring(geom.attrib["friction"], sep=" ")
        position = np.fromstring(geom.attrib["pos"], sep=" ")
        size = np.fromstring(geom.attrib["size"], sep=" ")
        if friction.size != 3 or position.size != 3 or size.size != 3:
            raise RuntimeError(f"MJCF {side} foot geometry is malformed")
        foot_values[side] = float(friction[0])
        sole_offsets[side] = np.array([position[0], position[1], position[2] - size[2]])
        half_sizes[side] = size[:2].copy()
    if not np.allclose(sole_offsets["left"], sole_offsets["right"], atol=0.0):
        raise RuntimeError("left/right MJCF sole offsets differ")
    if not np.allclose(half_sizes["left"], half_sizes["right"], atol=0.0):
        raise RuntimeError("left/right MJCF foot half sizes differ")
    return {
        "friction_coefficients": foot_values,
        "sole_offset_local_m": sole_offsets["left"],
        "foot_half_sizes_xy_m": half_sizes["left"],
    }


def _foot_polygon(geometry: dict, mjcf: dict) -> np.ndarray:
    length = float(geometry["geometry"]["foot_length_m"])
    width = float(geometry["geometry"]["foot_width_m"])
    expected = np.array([width / 2.0, length / 2.0])
    if not np.allclose(expected, mjcf["foot_half_sizes_xy_m"], atol=1.0e-12):
        raise RuntimeError("robot_geometry.yaml and MJCF foot dimensions differ")
    hx, hy = expected
    return np.array([[-hx, -hy], [hx, -hy], [hx, hy], [-hx, hy]])


def extract_rows(
    problem, model, metadata, xs, us, *, sole_offset_from_ankle_local
) -> list[dict]:
    problem.calc([x.copy() for x in xs], [u.copy() for u in us])
    pin_data = model.createData()
    rows = []
    contact_to_side = {"left_foot_6d": "left", "right_foot_6d": "right"}
    for node, differential_data in enumerate(problem.runningDatas):
        x = xs[node]
        q = x[: model.nq]
        v = x[model.nq :]
        pin.forwardKinematics(model, pin_data, q, v)
        pin.updateFramePlacements(model, pin_data)
        contacts = differential_data.differential.multibody.contacts.contacts.todict()
        expected = {
            side
            for side, active in (
                ("left", metadata["phase_by_node"][node].left_contact_active),
                ("right", metadata["phase_by_node"][node].right_contact_active),
            )
            if active
        }
        actual = {contact_to_side[name] for name in contacts}
        if actual != expected:
            raise RuntimeError(
                f"node {node} contact-data set {actual} disagrees with schedule {expected}"
            )
        for name, contact_data in contacts.items():
            side = contact_to_side[name]
            frame_id = int(contact_data.frame)
            rotation = np.asarray(pin_data.oMf[frame_id].rotation, dtype=float)
            contact_offset = np.asarray(
                model.frames[frame_id].placement.translation, dtype=float
            )
            contact_to_sole = contact_frame_to_sole_offset(
                sole_offset_from_ankle_local=sole_offset_from_ankle_local,
                contact_frame_offset_from_ankle_local=contact_offset,
            )
            wrench_sole = contact_wrench_at_sole(
                np.asarray(contact_data.f.vector, dtype=float),
                frame_rotation_world=rotation,
                sole_offset_local=contact_to_sole,
            )
            speed = sole_tangential_speed(
                np.asarray(contact_data.v.vector, dtype=float),
                frame_rotation_world=rotation,
                sole_offset_local=contact_to_sole,
            )
            rows.append(
                {
                    "node": node,
                    "time_s": node * 0.010,
                    "phase": metadata["phase_by_node"][node].name,
                    "cycle_id": int(metadata["cycle_id_by_node"][node]),
                    "side": side,
                    "sole_position_world_m": np.asarray(
                        pin_data.oMf[frame_id].translation, dtype=float
                    ).copy(),
                    "wrench": wrench_sole,
                    "tangential_speed_m_s": speed,
                }
            )
    return rows


def _csv_rows(raw_rows: list[dict], report_rows: list[dict]) -> list[dict]:
    """Repeat one node-level gate result for every raw contact diagnostic."""

    report_by_node = {int(row["node"]): row for row in report_rows}
    if len(report_by_node) != len(report_rows):
        raise RuntimeError("planar report contains duplicate node results")
    output = []
    for raw in raw_rows:
        node = int(raw["node"])
        if node not in report_by_node:
            raise RuntimeError(f"raw contact node {node} has no planar result")
        wrench = np.asarray(raw["wrench"], dtype=float)
        evaluated = report_by_node[node]
        output.append(
            {
                "node": node,
                "time_s": raw["time_s"],
                "phase": raw["phase"],
                "cycle_id": raw["cycle_id"],
                "side": raw["side"],
                "sole_x_world_m": raw["sole_position_world_m"][0],
                "sole_y_world_m": raw["sole_position_world_m"][1],
                "sole_z_world_m": raw["sole_position_world_m"][2],
                "fx_n": wrench[0],
                "fy_n": wrench[1],
                "fz_n": wrench[2],
                "tx_nm": wrench[3],
                "ty_nm": wrench[4],
                "tz_nm": wrench[5],
                **{
                    key: value
                    for key, value in evaluated.items()
                    if key
                    not in {
                        "node",
                        "time_s",
                        "phase",
                        "cycle_id",
                        "active_sides",
                        "support_interval_y_m",
                        "inner_support_interval_y_m",
                        "cone_values",
                    }
                },
            }
        )
    return output

def _plot(rows: list[dict], thresholds: dict, accepted: bool) -> None:
    """Plot node-level planar gate metrics, not non-unique per-foot loads."""

    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(4, 1, figsize=(12, 11), sharex=True)
    fields = (
        (
            "resultant_fz_n",
            "Combined normal force [N]",
            thresholds["minimum_normal_force_n"],
        ),
        (
            "friction_utilisation",
            "Sagittal friction utilisation [-]",
            thresholds["maximum_friction_utilisation"],
        ),
        (
            "cop_margin_m",
            "Sagittal CoP margin [m]",
            thresholds["minimum_cop_margin_m"],
        ),
        (
            "maximum_tangential_speed_m_s",
            "Maximum sole tangential speed [m/s]",
            thresholds["maximum_tangential_speed_m_s"],
        ),
    )
    for axis, (field, ylabel, threshold) in zip(axes, fields):
        axis.plot(
            [row["time_s"] for row in rows],
            [row[field] for row in rows],
            ".",
            ms=2.0,
            color="tab:blue",
            label="combined planar support",
        )
        axis.axhline(threshold, color="black", ls="--", lw=1.0, label="gate")
        axis.set_ylabel(ylabel)
        axis.grid(True, alpha=0.25)
    axes[0].legend(ncol=2, loc="best")
    axes[-1].set_xlabel("Time [s]")
    fig.suptitle(
        "FDDP ten-step planar support feasibility: "
        + ("PASS" if accepted else "FAIL")
    )
    fig.tight_layout()
    fig.savefig(FIGURE_PATH, dpi=180)
    plt.close(fig)

def _safe(value):
    if isinstance(value, dict):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def main() -> int:
    config, geometry, planning = _load_inputs()
    contact_config = dict(config["contact"])
    mjcf = _mjcf_contract()
    planning_mu = float(contact_config["friction_coefficient"])
    friction_contract_passed = True
    friction_error = None
    try:
        for replay_mu in mjcf["friction_coefficients"].values():
            assert_friction_contract(planning_mu, replay_mu)
    except ValueError as error:
        friction_contract_passed = False
        friction_error = str(error)
    polygon = _foot_polygon(geometry, mjcf)

    solver = _load_solver_module()
    model = pin.buildModelFromUrdf(str(solver.URDF_PATH), pin.JointModelFreeFlyer())
    x0 = solver.load_x0(solver.X0_PATH, model.nq + model.nv)
    problem, _state, metadata = solver.build_multistep_problem(model, x0, step_count=10)
    with np.load(CANDIDATE_PATH, allow_pickle=False) as archive:
        if _scalar_text(archive["acceptance_state"]) != "planning_accepted_contact_pending":
            raise RuntimeError("candidate acceptance state is invalid")
        if int(np.asarray(archive["step_count"]).reshape(()).item()) != 10:
            raise RuntimeError("candidate is not ten steps")
        xs = np.asarray(archive["state_ref"], dtype=float)
        us = np.asarray(archive["tau_ref"], dtype=float)
    if xs.shape != (1001, model.nq + model.nv) or us.shape != (1000, 6):
        raise RuntimeError("candidate state/control arrays violate the ten-step schema")
    raw_rows = extract_rows(
        problem,
        model,
        metadata,
        xs,
        us,
        sole_offset_from_ankle_local=mjcf["sole_offset_local_m"],
    )
    contact_report = evaluate_planar_support_feasibility(
        raw_rows,
        contact_config,
        foot_half_length_m=float(polygon[2, 1]),
    )
    contact_report["accepted"] = bool(
        contact_report["accepted"] and friction_contract_passed
    )
    node_rows = contact_report.pop("rows")
    rows = _csv_rows(raw_rows, node_rows)
    violation_counts = Counter()
    for row in node_rows:
        for name in (
            "unilateral_violation",
            "normal_force_violation",
            "friction_violation",
            "cop_violation",
            "slip_violation",
        ):
            violation_counts[name] += int(bool(row[name]))
    contact_report["rows"] = node_rows

    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CSV_PATH.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    _plot(node_rows, contact_config, contact_report["accepted"])

    candidate_sha = _sha256(CANDIDATE_PATH)
    promotion = None
    if contact_report["accepted"]:
        promotion = promote_candidate_reference(
            CANDIDATE_PATH,
            FINAL_PATH,
            expected_sha256=candidate_sha,
            contact_report=contact_report,
        )
        FINAL_SUMMARY_PATH.write_text(
            yaml.safe_dump(
                _safe(
                    {
                        "schema_version": "fddp-multistep-final-summary-v1",
                        "acceptance_state": "planning_and_contact_accepted",
                        "planning_acceptance": planning["planning_acceptance"],
                        "contact_acceptance": contact_report,
                        "promotion": promotion,
                        "claim_boundary": "simulation-only nominal; replay pending",
                        "tau_ref_hardware_usage": "analysis_only",
                    }
                ),
                sort_keys=False,
            ),
            encoding="utf-8",
        )

    summary = {
        "schema_version": "fddp-contact-feasibility-v2-planar-resultant",
        "passed": contact_report["accepted"],
        "planning_candidate_sha256": candidate_sha,
        "planning_candidate_acceptance": planning["acceptance_state"],
        "friction_contract_passed": friction_contract_passed,
        "friction_contract_error": friction_error,
        "planning_friction_coefficient": planning_mu,
        "mjcf_friction_coefficients": mjcf["friction_coefficients"],
        "sole_offset_from_ankle_local_m": mjcf["sole_offset_local_m"],
        "contact_frame_offsets_from_ankle_local_m": {
            "left": model.frames[metadata["left_frame_id"]].placement.translation,
            "right": model.frames[metadata["right_frame_id"]].placement.translation,
        },
        "contact_to_sole_offsets_local_m": {
            side: contact_frame_to_sole_offset(
                sole_offset_from_ankle_local=mjcf["sole_offset_local_m"],
                contact_frame_offset_from_ankle_local=model.frames[frame_id].placement.translation,
            )
            for side, frame_id in (
                ("left", metadata["left_frame_id"]),
                ("right", metadata["right_frame_id"]),
            )
        },
        "foot_polygon_xy_m": polygon,
        "thresholds": contact_config,
        "metrics": contact_report,
        "violation_counts": dict(violation_counts),
        "promotion": promotion,
        "final_reference_created": FINAL_PATH.exists() if promotion else False,
        "outputs": {"csv": str(CSV_PATH), "yaml": str(YAML_PATH), "figure": str(FIGURE_PATH)},
        "claim_boundary": (
            "simulation-only FDDP contact feasibility passed; MuJoCo replay pending"
            if contact_report["accepted"]
            else "planning candidate retained; contact feasibility failed; no final nominal"
        ),
        "tau_ref_hardware_usage": "analysis_only",
    }
    YAML_PATH.write_text(
        yaml.safe_dump(_safe(summary), sort_keys=False), encoding="utf-8"
    )
    print(yaml.safe_dump(_safe(summary), sort_keys=False))
    return 0 if summary["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
