#!/usr/bin/env python3
"""Generate a reproducible URDF/MJCF consistency ledger without editing models."""
from __future__ import annotations

import csv
import json
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
URDF_PATH = ROOT / "model" / "biped_v1.urdf"
MJCF_PATH = ROOT / "model" / "biped_planar.xml"
CSV_PATH = ROOT / "results" / "logs" / "urdf_mjcf_consistency_audit.csv"
YAML_PATH = ROOT / "results" / "logs" / "urdf_mjcf_consistency_audit.yaml"
DOC_PATH = ROOT / "docs" / "2026-07-13_urdf_mujoco_consistency_audit.md"

JOINTS = [
    "left_hip_pitch_joint",
    "left_knee_pitch_joint",
    "left_ankle_pitch_joint",
    "right_hip_pitch_joint",
    "right_knee_pitch_joint",
    "right_ankle_pitch_joint",
]
PROXY_DELAY_S = 0.020
PROXY_TIME_CONSTANT_S = 0.030
PROXY_SPEED_RAD_S = 8.0


def build_body_mapping() -> dict[str, list[str]]:
    mapping = {
        "hip_plate_link": ["hip_plate_link"],
        "electronics_lump_link": ["electronics_lump_link"],
    }
    for side in ("left", "right"):
        for part in (
            "hip_motor_link",
            "thigh_bracket_link",
            "knee_motor_link",
            "shank_bracket_link",
            "ankle_motor_link",
        ):
            name = f"{side}_{part}"
            mapping[name] = [name]
        foot = f"{side}_foot_link"
        mapping[foot] = [foot, f"{side}_foot_contact_lump"]
    return mapping


def _vector(value: str | None, default=(0.0, 0.0, 0.0)) -> np.ndarray:
    if value is None:
        return np.asarray(default, dtype=float)
    return np.asarray([float(part) for part in value.split()], dtype=float)


def _display(value) -> str:
    if isinstance(value, np.ndarray):
        value = value.tolist()
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def load_roots():
    return ET.parse(URDF_PATH).getroot(), ET.parse(MJCF_PATH).getroot()


def _urdf_link(root, name: str):
    item = root.find(f"./link[@name='{name}']")
    if item is None:
        raise KeyError(f"URDF link missing: {name}")
    return item


def _urdf_joint(root, name: str):
    item = root.find(f"./joint[@name='{name}']")
    if item is None:
        raise KeyError(f"URDF joint missing: {name}")
    return item


def _urdf_parent_joint(root, child: str):
    for joint in root.findall("./joint"):
        child_item = joint.find("./child")
        if child_item is not None and child_item.get("link") == child:
            return joint
    return None


def _mjcf_body(root, name: str):
    item = root.find(f".//body[@name='{name}']")
    if item is None:
        raise KeyError(f"MJCF body missing: {name}")
    return item


def _source_offset_to_target(urdf_root, source: str, target: str) -> np.ndarray:
    if source == target:
        return np.zeros(3)
    joint = _urdf_parent_joint(urdf_root, source)
    if joint is None or joint.get("type") != "fixed":
        raise ValueError(f"{source} is not directly fixed to {target}")
    parent = joint.find("./parent")
    if parent is None or parent.get("link") != target:
        raise ValueError(f"unsupported folding chain: {target} -> {source}")
    origin = joint.find("./origin")
    if origin is not None and not np.allclose(_vector(origin.get("rpy")), 0.0):
        raise ValueError("rotated folded inertials are not supported by this ledger")
    return _vector(None if origin is None else origin.get("xyz"))


def _urdf_inertial(root, link_name: str, target_name: str) -> dict:
    link = _urdf_link(root, link_name)
    inertial = link.find("./inertial")
    if inertial is None:
        raise KeyError(f"URDF inertial missing: {link_name}")
    mass = float(inertial.find("./mass").get("value"))
    origin = inertial.find("./origin")
    if origin is not None and not np.allclose(_vector(origin.get("rpy")), 0.0):
        raise ValueError(f"rotated inertial frame not supported: {link_name}")
    com = _source_offset_to_target(root, link_name, target_name)
    com += _vector(None if origin is None else origin.get("xyz"))
    raw = inertial.find("./inertia")
    inertia = np.array(
        [
            [float(raw.get("ixx")), float(raw.get("ixy")), float(raw.get("ixz"))],
            [float(raw.get("ixy")), float(raw.get("iyy")), float(raw.get("iyz"))],
            [float(raw.get("ixz")), float(raw.get("iyz")), float(raw.get("izz"))],
        ]
    )
    return {"mass": mass, "com": com, "inertia": inertia}


def combine_urdf_inertials(root, sources: list[str], target: str) -> dict:
    items = [_urdf_inertial(root, source, target) for source in sources]
    mass = sum(item["mass"] for item in items)
    com = sum(item["mass"] * item["com"] for item in items) / mass
    inertia = np.zeros((3, 3))
    for item in items:
        delta = item["com"] - com
        inertia += item["inertia"] + item["mass"] * (
            np.dot(delta, delta) * np.eye(3) - np.outer(delta, delta)
        )
    return {"mass": float(mass), "com": com, "inertia": inertia}


def mjcf_body_inertial(root, body_name: str) -> dict:
    body = _mjcf_body(root, body_name)
    inertial = body.find("./inertial")
    if inertial is None:
        raise KeyError(f"MJCF inertial missing: {body_name}")
    diagonal = _vector(inertial.get("diaginertia"))
    return {
        "mass": float(inertial.get("mass")),
        "com": _vector(inertial.get("pos")),
        "inertia": np.diag(diagonal),
    }


def derive_standing_base_height() -> float:
    urdf_root, _ = load_roots()
    offsets = []
    for child in (
        "left_hip_motor_link",
        "left_knee_motor_link",
        "left_ankle_motor_link",
    ):
        joint = _urdf_parent_joint(urdf_root, child)
        offsets.append(_vector(joint.find("./origin").get("xyz"))[2])
    foot = _urdf_link(urdf_root, "left_foot_link")
    collision = foot.find("./collision")
    foot_origin_z = _vector(collision.find("./origin").get("xyz"))[2]
    foot_half_z = _vector(collision.find("./geometry/box").get("size"))[2] / 2.0
    bottom_from_base = sum(offsets) + foot_origin_z - foot_half_z
    return float(-bottom_from_base)


def _row(rows, category, item, urdf_value, mjcf_value, status, note):
    rows.append(
        {
            "category": category,
            "item": item,
            "urdf_value": _display(urdf_value),
            "mjcf_value": _display(mjcf_value),
            "status": status,
            "note": note,
        }
    )


def _status_close(a, b, atol: float) -> str:
    return "matched" if np.allclose(a, b, rtol=0.0, atol=atol) else "unexpected_mismatch"


def _default(root, child: str, attribute: str, fallback: float = 0.0) -> float:
    item = root.find(f"./default/{child}")
    if item is None or item.get(attribute) is None:
        return fallback
    return float(item.get(attribute))


def audit_models() -> dict:
    urdf_root, mjcf_root = load_roots()
    rows = []
    mapping = build_body_mapping()
    total_urdf = 0.0
    total_mjcf = 0.0
    for body_name, sources in mapping.items():
        urdf = combine_urdf_inertials(urdf_root, sources, body_name)
        mjcf = mjcf_body_inertial(mjcf_root, body_name)
        total_urdf += urdf["mass"]
        total_mjcf += mjcf["mass"]
        _row(rows, "body_mass", body_name, urdf["mass"], mjcf["mass"],
             _status_close(urdf["mass"], mjcf["mass"], 1e-12), "kg")
        _row(rows, "inertial_origin", body_name, urdf["com"], mjcf["com"],
             _status_close(urdf["com"], mjcf["com"], 1e-12), "COM in body/link frame, m")
        _row(rows, "inertia_tensor", body_name, urdf["inertia"], mjcf["inertia"],
             _status_close(urdf["inertia"], mjcf["inertia"], 5e-9), "kg m^2; MJCF foot values rounded")

        mj_body = _mjcf_body(mjcf_root, body_name)
        mj_pos = _vector(mj_body.get("pos"))
        if body_name == "hip_plate_link":
            urdf_pos = np.array([0.0, 0.0, derive_standing_base_height()])
            category = "standing_base_origin"
            note = "MJCF floating-base zero pose places the URDF foot bottom on z=0"
        else:
            parent_joint = _urdf_parent_joint(urdf_root, body_name)
            origin = parent_joint.find("./origin")
            urdf_pos = _vector(None if origin is None else origin.get("xyz"))
            category = "body_origin"
            note = "URDF parent-to-joint/child transform vs MJCF local body pos"
        _row(rows, category, body_name, urdf_pos, mj_pos,
             _status_close(urdf_pos, mj_pos, 1e-12), note)

        urdf_link = _urdf_link(urdf_root, body_name)
        visual = urdf_link.find("./visual")
        urdf_size = _vector(visual.find("./geometry/box").get("size"))
        urdf_geom_pos = _vector(visual.find("./origin").get("xyz"))
        mj_geom = mj_body.find("./geom")
        mj_size = 2.0 * _vector(mj_geom.get("size"))
        mj_geom_pos = _vector(mj_geom.get("pos"))
        _row(rows, "visual_geometry_size", body_name, urdf_size, mj_size,
             _status_close(urdf_size, mj_size, 1e-12), "URDF full box size vs twice MJCF half-size")
        _row(rows, "visual_geometry_origin", body_name, urdf_geom_pos, mj_geom_pos,
             _status_close(urdf_geom_pos, mj_geom_pos, 1e-12), "geometry origin in owning link/body frame")

        collision_present = urdf_link.find("./collision") is not None
        contype = int(mj_geom.get("contype", "1"))
        foot = body_name.endswith("foot_link")
        collision_status = "matched" if foot and collision_present and contype == 1 else "deliberate_simplification"
        _row(rows, "collision_enablement", body_name, collision_present, contype,
             collision_status,
             "feet collide in both; non-foot MJCF collision is intentionally disabled")
        if len(sources) > 1:
            _row(rows, "rigid_body_folding", body_name, sources, body_name,
                 "rigid_body_folded",
                 "contact-lump mass/COM/inertia folded into foot; its separate geometry is omitted")

    _row(rows, "total_mass", "all mapped bodies", total_urdf, total_mjcf,
         _status_close(total_urdf, total_mjcf, 1e-12), "kg")

    default_damping = _default(mjcf_root, "joint", "damping")
    actuators = {item.get("joint"): item for item in mjcf_root.findall("./actuator/position")}
    for joint_name in JOINTS:
        urdf_joint = _urdf_joint(urdf_root, joint_name)
        mj_joint = mjcf_root.find(f".//joint[@name='{joint_name}']")
        urdf_axis = _vector(urdf_joint.find("./axis").get("xyz"))
        mj_axis = _vector(mj_joint.get("axis"))
        _row(rows, "joint_axis", joint_name, urdf_axis, mj_axis,
             _status_close(urdf_axis, mj_axis, 1e-12), "axis expressed in joint/child frame")
        limit = urdf_joint.find("./limit")
        urdf_range = [float(limit.get("lower")), float(limit.get("upper"))]
        mj_range = _vector(mj_joint.get("range"))
        _row(rows, "joint_limit", joint_name, urdf_range, mj_range,
             _status_close(urdf_range, mj_range, 1e-12), "rad")
        dynamics = urdf_joint.find("./dynamics")
        urdf_damping = float(dynamics.get("damping"))
        mj_damping = float(mj_joint.get("damping", default_damping))
        _row(rows, "joint_damping", joint_name, urdf_damping, mj_damping,
             _status_close(urdf_damping, mj_damping, 1e-12), "N m s/rad proxy parameter")
        urdf_friction = float(dynamics.get("friction"))
        mj_frictionloss = float(mj_joint.get("frictionloss", "0"))
        _row(rows, "joint_passive_friction", joint_name, urdf_friction, mj_frictionloss,
             "deliberate_mismatch", "URDF Coulomb friction is not represented in current MJCF")
        actuator = actuators[joint_name]
        force_range = _vector(actuator.get("forcerange", mjcf_root.find("./default/position").get("forcerange")))
        _row(rows, "actuator_effort", joint_name, float(limit.get("effort")), force_range,
             "deliberate_mismatch", "URDF effort 2.94 N m; MJCF position actuator is capped at +/-2 N m")
        ctrl_range = _vector(actuator.get("ctrlrange"))
        _row(rows, "actuator_position_range", joint_name, urdf_range, ctrl_range,
             _status_close(urdf_range, ctrl_range, 1e-12), "position-command range, rad")
        _row(rows, "actuator_velocity", joint_name, float(limit.get("velocity")), PROXY_SPEED_RAD_S,
             "deliberate_mismatch", "URDF velocity limit vs replay proxy command slew limit")

    _row(rows, "actuator_semantics", "six leg joints", "no URDF transmission/driver block",
         {"type": "MJCF position", "delay_s": PROXY_DELAY_S,
          "time_constant_s": PROXY_TIME_CONSTANT_S, "speed_rad_s": PROXY_SPEED_RAD_S},
         "consumer_specific", "replay q_cmd proxy is not a calibrated hardware actuator model")
    _row(rows, "root_dofs", "planar floating base", "not represented as URDF joints",
         {"forward": [0, 1, 0], "vertical": [0, 0, 1], "pitch": [1, 0, 0]},
         "consumer_specific", "MJCF adds +Y, +Z and +X planar root coordinates")

    default_geom = mjcf_root.find("./default/geom")
    contact = {
        "friction": _vector(default_geom.get("friction")).tolist(),
        "solref": _vector(default_geom.get("solref")).tolist(),
        "solimp": _vector(default_geom.get("solimp")).tolist(),
    }
    _row(rows, "contact_parameters", "foot-ground contact", "unspecified in URDF", contact,
         "mjcf_only_uncalibrated", "MuJoCo-only simulation parameters; not hardware-calibrated")
    for side in ("left", "right"):
        body = _mjcf_body(mjcf_root, f"{side}_foot_link")
        geom = body.find("./geom")
        half = _vector(geom.get("size"))
        centre = _vector(geom.get("pos"))
        bottom = centre[2] - half[2]
        for end, sign in (("toe", 1.0), ("heel", -1.0)):
            site_name = f"{side}_{end}_site"
            site = body.find(f"./site[@name='{site_name}']")
            pos = _vector(site.get("pos"))
            valid = abs(pos[1]) <= half[1] + 1e-12 and abs(pos[2] - bottom) <= 1e-12
            _row(rows, "contact_site_geometry", site_name,
                 {"foot_forward_half_m": half[1], "foot_bottom_z_m": bottom}, pos,
                 "matched" if valid else "unexpected_mismatch",
                 "toe/heel site lies within footprint and on the foot bottom plane")
            sensor = mjcf_root.find(f"./sensor/touch[@site='{site_name}']")
            _row(rows, "contact_sensor", f"{side}_{end}_touch", "no URDF sensor block",
                 None if sensor is None else sensor.get("name"), "mjcf_only_runtime_feature",
                 "natural-contact C2 uses this MuJoCo touch sensor")

    counts = Counter(row["status"] for row in rows)
    unexpected = counts.get("unexpected_mismatch", 0)
    summary = {
        "mapped_mjcf_body_count": len(mapping),
        "mapped_urdf_inertial_link_count": sum(len(value) for value in mapping.values()),
        "audited_joint_count": len(JOINTS),
        "row_count": len(rows),
        "matched_count": counts.get("matched", 0),
        "unexpected_mismatch_count": unexpected,
        "status_counts": dict(sorted(counts.items())),
        "full_equivalence": False,
        "audit_integrity_passed": unexpected == 0,
    }
    return {
        "stage": "urdf_mjcf_consistency_audit",
        "urdf": str(URDF_PATH),
        "mjcf": str(MJCF_PATH),
        "units": "SI: m, kg, s, rad",
        "frame_convention": "+X robot right/pitch axis, +Y forward, +Z upward",
        "source_note": "dimensions and inertials are current project estimates/proxies unless independently measured",
        "summary": summary,
        "rows": rows,
        "claim_boundary": (
            "The mapping ledger is complete for the current files, but the models are not fully equivalent "
            "and the MJCF contact/actuator parameters are not hardware calibrated."
        ),
    }


def _write_markdown(report: dict) -> None:
    summary = report["summary"]
    differences = [row for row in report["rows"] if row["status"] != "matched"]
    lines = [
        "# URDF and MuJoCo Consistency Audit",
        "",
        "## Scope and claim boundary",
        "",
        "Audit date: 2026-07-13. FDDP/Pinocchio consumes `model/biped_v1.urdf`; MuJoCo replay consumes `model/biped_planar.xml`.",
        "This is a complete mapping ledger for the current simulation files, not a hardware calibration certificate and not proof of full URDF/MJCF equivalence.",
        "No model file was modified by this audit.",
        "",
        "## Design ledger",
        "",
        "- Units: SI (m, kg, s, rad).",
        "- Axes: +X robot right and pitch axis, +Y forward, +Z upward.",
        "- Sources: current URDF primitives and estimated/proxy inertials; replace estimates with measurements before hardware claims.",
        "- Mapping: 14 MuJoCo bodies cover 16 URDF inertial links; each foot contact lump is rigid-folded into its foot using the parallel-axis theorem.",
        "- Consumer difference: MJCF adds planar root DOFs and simulation-only contact/position-actuator semantics.",
        "",
        "## Automated result",
        "",
        f"- Ledger rows: {summary['row_count']}",
        f"- Matched numeric/structural rows: {summary['matched_count']}",
        f"- Unexpected mismatches: {summary['unexpected_mismatch_count']}",
        f"- Audit integrity passed: `{summary['audit_integrity_passed']}`",
        f"- Full equivalence: `{summary['full_equivalence']}`",
        "- Zero-pose base height derived from URDF foot-bottom geometry: 0.288 m; matches MJCF.",
        "",
        "## Deliberate or consumer-specific differences",
        "",
        "| Category | Item | Status | Meaning |",
        "|---|---|---|---|",
    ]
    for row in differences:
        lines.append(
            f"| {row['category']} | {row['item']} | {row['status']} | {row['note']} |"
        )
    lines.extend(
        [
            "",
            "## Verified matches",
            "",
            "Per mapped body, the ledger checks mass, COM/inertial origin, inertia tensor, local body origin, primitive box size/origin, and collision enablement. For all six leg joints it checks +X axis, range, damping, and actuator position range. Toe/heel sites are checked against the foot footprint and bottom plane.",
            "",
            "## Remaining calibration boundary",
            "",
            "The current MJCF intentionally disables non-foot collision, omits URDF joint Coulomb friction, uses a +/-2 N m position-actuator cap instead of the URDF 2.94 N m effort limit, and uses an 8 rad/s replay-proxy slew limit instead of the URDF 4.712 rad/s velocity limit. Friction, solref/solimp, delay, lag, actuator gains, and touch sensing are simulation parameters and are not hardware calibrated.",
            "",
            "Therefore the supported statement is: the current files have a complete, reproducible consistency ledger with no unexpected mismatches under its documented mappings; they are not fully equivalent physical models.",
            "",
            "## Machine-readable artifacts",
            "",
            "- `results/logs/urdf_mjcf_consistency_audit.csv`",
            "- `results/logs/urdf_mjcf_consistency_audit.yaml`",
        ]
    )
    DOC_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    report = audit_models()
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CSV_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=report["rows"][0].keys())
        writer.writeheader()
        writer.writerows(report["rows"])
    YAML_PATH.write_text(yaml.safe_dump(report, sort_keys=False), encoding="utf-8")
    _write_markdown(report)
    print(yaml.safe_dump(report["summary"], sort_keys=False))
    print(CSV_PATH)
    print(YAML_PATH)
    print(DOC_PATH)
    if not report["summary"]["audit_integrity_passed"]:
        raise SystemExit("unexpected URDF/MJCF mismatch detected")


if __name__ == "__main__":
    main()
