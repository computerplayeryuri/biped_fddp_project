"""Pure contact-feasibility metrics and gated nominal-reference promotion.

Pinocchio spatial-force vectors are interpreted as ``[Fx, Fy, Fz, Tx, Ty,
Tz]``.  The project convention is x right, y forward, and z upward.
"""

from __future__ import annotations

import hashlib
import os
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np


def _as_polygon(polygon_xy: np.ndarray) -> np.ndarray:
    polygon = np.asarray(polygon_xy, dtype=float)
    if polygon.ndim != 2 or polygon.shape[1] != 2 or polygon.shape[0] < 3:
        raise ValueError("polygon must have shape (N, 2), N >= 3")
    if not np.all(np.isfinite(polygon)):
        raise ValueError("polygon must contain only finite coordinates")
    edges = np.roll(polygon, -1, axis=0) - polygon
    if np.any(np.linalg.norm(edges, axis=1) <= np.finfo(float).eps):
        raise ValueError("polygon contains a zero-length edge")
    return polygon


def point_polygon_signed_margin(point_xy, polygon_xy) -> float:
    """Return the minimum signed distance to a convex polygon edge.

    Positive values are inside, zero lies on an edge, and negative values are
    outside.  Clockwise and counter-clockwise vertex ordering are supported.
    """

    point = np.asarray(point_xy, dtype=float).reshape(-1)
    if point.size != 2 or not np.all(np.isfinite(point)):
        raise ValueError("point_xy must contain two finite coordinates")
    polygon = _as_polygon(polygon_xy)
    signed_area_twice = float(
        np.sum(
            polygon[:, 0] * np.roll(polygon[:, 1], -1)
            - np.roll(polygon[:, 0], -1) * polygon[:, 1]
        )
    )
    if abs(signed_area_twice) <= np.finfo(float).eps:
        raise ValueError("polygon area must be non-zero")
    orientation = 1.0 if signed_area_twice > 0.0 else -1.0
    edges = np.roll(polygon, -1, axis=0) - polygon
    offsets = point - polygon
    cross = edges[:, 0] * offsets[:, 1] - edges[:, 1] * offsets[:, 0]
    distances = orientation * cross / np.linalg.norm(edges, axis=1)
    return float(np.min(distances))


def wrench_metrics(wrench, mu: float, polygon: np.ndarray) -> dict:
    """Compute normal-force, Coulomb-friction, and CoP metrics for one foot."""

    vector = np.asarray(wrench, dtype=float).reshape(-1)
    if vector.size != 6 or not np.all(np.isfinite(vector)):
        raise ValueError("wrench must contain six finite values")
    if not np.isfinite(mu) or mu <= 0.0:
        raise ValueError("mu must be finite and positive")
    polygon = _as_polygon(polygon)
    fx, fy, fz, tx, ty, _tz = vector
    tangential_force = float(np.hypot(fx, fy))
    normal_force_invalid = bool(fz <= 0.0)
    unilateral_violation = bool(fz < 0.0)
    if normal_force_invalid:
        friction_utilisation = float("inf")
        cop = np.full(2, np.nan, dtype=float)
        cop_margin = float("nan")
    else:
        friction_utilisation = tangential_force / (float(mu) * float(fz))
        cop = np.array([-ty / fz, tx / fz], dtype=float)
        cop_margin = point_polygon_signed_margin(cop, polygon)
    return {
        "normal_force_n": float(fz),
        "tangential_force_n": tangential_force,
        "friction_utilisation": float(friction_utilisation),
        "cop_xy_m": cop,
        "cop_margin_m": float(cop_margin),
        "normal_force_invalid": normal_force_invalid,
        "unilateral_violation": unilateral_violation,
    }


def planar_support_resultant(
    contacts: Sequence[Mapping],
    *,
    friction_coefficient: float,
    maximum_friction_utilisation: float,
    foot_half_length_m: float,
    minimum_cop_margin_m: float,
) -> dict:
    """Evaluate one planar support resultant at the active sole centres.

    This is the feasibility contract for the declared planar model: forward
    force Fy, vertical force Fz, and pitch moment Mx are gated against the
    union of active sagittal foot intervals.  Fx, Ty, and Tz are preserved as
    diagnostics only because the model has no lateral root translation or
    yaw/roll dynamics.  In double support this resultant is identifiable
    whereas an individual left/right load split is not.
    """

    if not contacts:
        raise ValueError("at least one active contact is required")
    mu = float(friction_coefficient)
    maximum_utilisation = float(maximum_friction_utilisation)
    half_length = float(foot_half_length_m)
    required_margin = float(minimum_cop_margin_m)
    values = (mu, maximum_utilisation, half_length, required_margin)
    if not all(np.isfinite(value) for value in values):
        raise ValueError("planar support parameters must be finite")
    if mu <= 0.0 or maximum_utilisation <= 0.0:
        raise ValueError("friction parameters must be positive")
    if half_length <= 0.0 or required_margin < 0.0:
        raise ValueError("support dimensions are invalid")
    if required_margin >= half_length:
        raise ValueError("CoP margin must be smaller than foot half length")

    fy = fz = mx_world = 0.0
    fx = ty = tz = 0.0
    support_min = float("inf")
    support_max = float("-inf")
    sides = set()
    for contact in contacts:
        side = str(contact.get("side", ""))
        if side not in {"left", "right"}:
            raise ValueError(f"unsupported contact side: {side}")
        if side in sides:
            raise ValueError(f"duplicate active contact side: {side}")
        sides.add(side)
        position = np.asarray(
            contact["sole_position_world_m"], dtype=float
        ).reshape(-1)
        wrench = np.asarray(contact["wrench"], dtype=float).reshape(-1)
        if position.size != 3 or not np.all(np.isfinite(position)):
            raise ValueError("sole_position_world_m must contain three finite values")
        if wrench.size != 6 or not np.all(np.isfinite(wrench)):
            raise ValueError("wrench must contain six finite values")
        contact_fx, contact_fy, contact_fz, contact_tx, contact_ty, contact_tz = wrench
        fx += float(contact_fx)
        fy += float(contact_fy)
        fz += float(contact_fz)
        mx_world += float(
            contact_tx + position[1] * contact_fz - position[2] * contact_fy
        )
        ty += float(contact_ty)
        tz += float(contact_tz)
        support_min = min(support_min, float(position[1] - half_length))
        support_max = max(support_max, float(position[1] + half_length))

    inner_min = support_min + required_margin
    inner_max = support_max - required_margin
    effective_mu = mu * maximum_utilisation
    cone_values = np.array(
        [
            fz,
            effective_mu * fz - fy,
            effective_mu * fz + fy,
            mx_world - inner_min * fz,
            inner_max * fz - mx_world,
        ],
        dtype=float,
    )
    normal_force_invalid = bool(fz <= 0.0)
    if normal_force_invalid:
        friction_utilisation = float("inf")
        cop_y = float("nan")
        cop_margin = float("nan")
    else:
        friction_utilisation = abs(fy) / (mu * fz)
        cop_y = mx_world / fz
        cop_margin = min(cop_y - support_min, support_max - cop_y)

    # Numerical tolerance is only for algebraic equality at floating-point
    # precision; it is scaled by the evaluated quantity, not a relaxed gate.
    cone_tolerance = 128.0 * np.finfo(float).eps * max(
        1.0, float(np.max(np.abs(cone_values)))
    )
    utilisation_tolerance = 128.0 * np.finfo(float).eps * max(
        1.0, maximum_utilisation
    )
    coordinate_tolerance = 128.0 * np.finfo(float).eps * max(
        1.0, abs(support_min), abs(support_max), abs(required_margin)
    )
    unilateral_violation = normal_force_invalid
    friction_violation = bool(
        normal_force_invalid
        or friction_utilisation > maximum_utilisation + utilisation_tolerance
    )
    cop_violation = bool(
        normal_force_invalid
        or not np.isfinite(cop_margin)
        or cop_margin < required_margin - coordinate_tolerance
    )
    return {
        "evaluation_mode": "planar_sagittal_resultant",
        "out_of_plane_policy": "diagnostic_only",
        "active_sides": tuple(sorted(sides)),
        "per_foot_load_split_nonidentifiable": len(sides) > 1,
        "resultant_fy_n": float(fy),
        "resultant_fz_n": float(fz),
        "resultant_mx_world_nm": float(mx_world),
        "out_of_plane_resultant_fx_n": float(fx),
        "out_of_plane_resultant_ty_nm": float(ty),
        "out_of_plane_resultant_tz_nm": float(tz),
        "support_interval_y_m": np.array([support_min, support_max], dtype=float),
        "inner_support_interval_y_m": np.array([inner_min, inner_max], dtype=float),
        "effective_friction_coefficient": float(effective_mu),
        "friction_utilisation": float(friction_utilisation),
        "cop_y_m": float(cop_y),
        "cop_margin_m": float(cop_margin),
        "normal_force_invalid": normal_force_invalid,
        "unilateral_violation": unilateral_violation,
        "friction_violation": friction_violation,
        "cop_violation": cop_violation,
        "cone_values": cone_values,
        "numerical_cone_tolerance": float(cone_tolerance),
        "accepted": bool(
            not unilateral_violation
            and not friction_violation
            and not cop_violation
            and np.all(cone_values >= -cone_tolerance)
        ),
    }

def evaluate_planar_support_feasibility(
    rows: Sequence[Mapping],
    config: Mapping,
    *,
    foot_half_length_m: float,
) -> dict:
    """Gate active-contact rows once per node using a planar resultant."""

    required = {
        "friction_coefficient",
        "minimum_normal_force_n",
        "maximum_friction_utilisation",
        "minimum_cop_margin_m",
        "maximum_tangential_speed_m_s",
    }
    missing = sorted(required.difference(config))
    if missing:
        raise KeyError(f"missing contact configuration keys: {missing}")
    if not rows:
        return {
            "accepted": False,
            "evaluation_mode": "planar_sagittal_resultant",
            "out_of_plane_policy": "diagnostic_only",
            "support_node_count": 0,
            "active_contact_row_count": 0,
            "violation_count": 0,
            "rows": [],
        }

    grouped = defaultdict(list)
    for source in rows:
        grouped[int(source["node"])].append(source)

    minimum_normal = float(config["minimum_normal_force_n"])
    maximum_speed = float(config["maximum_tangential_speed_m_s"])
    evaluated = []
    for node in sorted(grouped):
        source_rows = grouped[node]
        first = source_rows[0]
        contacts = [
            {
                "side": row["side"],
                "sole_position_world_m": row["sole_position_world_m"],
                "wrench": row["wrench"],
            }
            for row in source_rows
        ]
        result = planar_support_resultant(
            contacts,
            friction_coefficient=float(config["friction_coefficient"]),
            maximum_friction_utilisation=float(
                config["maximum_friction_utilisation"]
            ),
            foot_half_length_m=foot_half_length_m,
            minimum_cop_margin_m=float(config["minimum_cop_margin_m"]),
        )
        speed = max(float(row["tangential_speed_m_s"]) for row in source_rows)
        normal_force_violation = bool(
            result["resultant_fz_n"] < minimum_normal
        )
        slip_violation = classify_slip(speed, maximum=maximum_speed)
        accepted = bool(
            result["accepted"] and not normal_force_violation and not slip_violation
        )
        evaluated.append(
            {
                "node": node,
                "time_s": float(first.get("time_s", float("nan"))),
                "phase": str(first.get("phase", "")),
                "cycle_id": int(first.get("cycle_id", -1)),
                "active_contact_count": len(source_rows),
                "maximum_tangential_speed_m_s": speed,
                "normal_force_violation": normal_force_violation,
                "slip_violation": slip_violation,
                "accepted": accepted,
                **result,
            }
        )

    finite_utilisations = [
        row["friction_utilisation"]
        for row in evaluated
        if np.isfinite(row["friction_utilisation"])
    ]
    finite_margins = [
        row["cop_margin_m"] for row in evaluated if np.isfinite(row["cop_margin_m"])
    ]
    violation_count = sum(not row["accepted"] for row in evaluated)
    return {
        "accepted": bool(evaluated) and violation_count == 0,
        "evaluation_mode": "planar_sagittal_resultant",
        "out_of_plane_policy": "diagnostic_only",
        "support_node_count": len(evaluated),
        "active_contact_row_count": len(rows),
        "violation_count": int(violation_count),
        "minimum_normal_force_n": min(
            (row["resultant_fz_n"] for row in evaluated), default=float("nan")
        ),
        "maximum_friction_utilisation": max(
            finite_utilisations, default=float("nan")
        ),
        "minimum_cop_margin_m": min(finite_margins, default=float("nan")),
        "maximum_tangential_speed_m_s": max(
            (row["maximum_tangential_speed_m_s"] for row in evaluated),
            default=float("nan"),
        ),
        "rows": evaluated,
    }

def contact_wrench_at_sole(
    wrench_lwa, *, frame_rotation_world, sole_offset_local
) -> np.ndarray:
    """Express an LWA wrench at the foot sole center in foot-local axes."""

    wrench = np.asarray(wrench_lwa, dtype=float).reshape(-1)
    rotation = np.asarray(frame_rotation_world, dtype=float)
    offset = np.asarray(sole_offset_local, dtype=float).reshape(-1)
    if wrench.size != 6 or not np.all(np.isfinite(wrench)):
        raise ValueError("wrench_lwa must contain six finite values")
    if rotation.shape != (3, 3) or not np.all(np.isfinite(rotation)):
        raise ValueError("frame_rotation_world must be a finite 3x3 matrix")
    if offset.size != 3 or not np.all(np.isfinite(offset)):
        raise ValueError("sole_offset_local must contain three finite values")
    force_local = rotation.T @ wrench[:3]
    moment_origin_local = rotation.T @ wrench[3:]
    moment_sole_local = moment_origin_local - np.cross(offset, force_local)
    return np.concatenate((force_local, moment_sole_local))


def contact_frame_to_sole_offset(
    *, sole_offset_from_ankle_local, contact_frame_offset_from_ankle_local
) -> np.ndarray:
    """Return the sole translation measured from the actual contact frame."""

    sole = np.asarray(sole_offset_from_ankle_local, dtype=float).reshape(-1)
    contact = np.asarray(
        contact_frame_offset_from_ankle_local, dtype=float
    ).reshape(-1)
    if sole.size != 3 or contact.size != 3:
        raise ValueError("sole and contact-frame offsets must contain three values")
    if not np.all(np.isfinite(sole)) or not np.all(np.isfinite(contact)):
        raise ValueError("sole and contact-frame offsets must be finite")
    return sole - contact

def sole_tangential_speed(
    velocity_lwa, *, frame_rotation_world, sole_offset_local
) -> float:
    """Compute local sole-center tangential speed from an LWA spatial motion."""

    velocity = np.asarray(velocity_lwa, dtype=float).reshape(-1)
    rotation = np.asarray(frame_rotation_world, dtype=float)
    offset = np.asarray(sole_offset_local, dtype=float).reshape(-1)
    if velocity.size != 6 or not np.all(np.isfinite(velocity)):
        raise ValueError("velocity_lwa must contain six finite values")
    if rotation.shape != (3, 3) or not np.all(np.isfinite(rotation)):
        raise ValueError("frame_rotation_world must be a finite 3x3 matrix")
    if offset.size != 3 or not np.all(np.isfinite(offset)):
        raise ValueError("sole_offset_local must contain three finite values")
    linear_local = rotation.T @ velocity[:3]
    angular_local = rotation.T @ velocity[3:]
    sole_linear_local = linear_local + np.cross(angular_local, offset)
    return float(np.linalg.norm(sole_linear_local[:2]))

def classify_slip(tangential_speed_m_s: float, *, maximum: float = 1.0e-3) -> bool:
    """Return True only when tangential speed exceeds the inclusive limit."""

    speed = float(tangential_speed_m_s)
    limit = float(maximum)
    if not np.isfinite(speed) or speed < 0.0:
        raise ValueError("tangential speed must be finite and non-negative")
    if not np.isfinite(limit) or limit < 0.0:
        raise ValueError("maximum tangential speed must be finite and non-negative")
    return bool(speed > limit)


def assert_friction_contract(planning_mu: float, replay_mu: float) -> None:
    """Reject any planning/replay friction mismatch, including silent rounding."""

    planning = float(planning_mu)
    replay = float(replay_mu)
    if not np.isfinite(planning) or not np.isfinite(replay) or planning <= 0.0:
        raise ValueError("friction coefficients must be finite and positive")
    if planning != replay:
        raise ValueError(
            f"friction contract mismatch: planning mu={planning}, replay mu={replay}"
        )


def evaluate_contact_feasibility(
    rows: Sequence[Mapping], config: Mapping, *, polygon: np.ndarray
) -> dict:
    """Evaluate per-active-contact rows against inclusive feasibility gates."""

    required = {
        "friction_coefficient",
        "minimum_normal_force_n",
        "maximum_friction_utilisation",
        "minimum_cop_margin_m",
        "maximum_tangential_speed_m_s",
    }
    missing = sorted(required.difference(config))
    if missing:
        raise KeyError(f"missing contact configuration keys: {missing}")
    mu = float(config["friction_coefficient"])
    minimum_normal = float(config["minimum_normal_force_n"])
    maximum_utilisation = float(config["maximum_friction_utilisation"])
    minimum_cop_margin = float(config["minimum_cop_margin_m"])
    maximum_speed = float(config["maximum_tangential_speed_m_s"])
    enriched = []
    by_node = defaultdict(list)
    for source in rows:
        node = int(source["node"])
        side = str(source["side"])
        if side not in {"left", "right"}:
            raise ValueError(f"unsupported contact side: {side}")
        speed = float(source["tangential_speed_m_s"])
        metrics = wrench_metrics(source["wrench"], mu=mu, polygon=polygon)
        row = {
            "node": node,
            "side": side,
            "normal_force_n": metrics["normal_force_n"],
            "tangential_force_n": metrics["tangential_force_n"],
            "friction_utilisation": metrics["friction_utilisation"],
            "cop_x_m": float(metrics["cop_xy_m"][0]),
            "cop_y_m": float(metrics["cop_xy_m"][1]),
            "cop_margin_m": metrics["cop_margin_m"],
            "tangential_speed_m_s": speed,
            "unilateral_violation": bool(
                metrics["normal_force_n"] < minimum_normal
            ),
            "friction_violation": bool(
                not np.isfinite(metrics["friction_utilisation"])
                or metrics["friction_utilisation"] > maximum_utilisation
            ),
            "cop_violation": bool(
                not np.isfinite(metrics["cop_margin_m"])
                or metrics["cop_margin_m"] < minimum_cop_margin
            ),
            "slip_violation": classify_slip(speed, maximum=maximum_speed),
        }
        row["accepted"] = not any(
            row[name]
            for name in (
                "unilateral_violation",
                "friction_violation",
                "cop_violation",
                "slip_violation",
            )
        )
        enriched.append(row)
        by_node[node].append(row)

    for node_rows in by_node.values():
        total = sum(max(0.0, row["normal_force_n"]) for row in node_rows)
        for row in node_rows:
            row["normal_load_fraction"] = (
                max(0.0, row["normal_force_n"]) / total
                if total > 0.0
                else float("nan")
            )

    violations = sum(not row["accepted"] for row in enriched)
    finite_utils = [
        row["friction_utilisation"]
        for row in enriched
        if np.isfinite(row["friction_utilisation"])
    ]
    finite_margins = [
        row["cop_margin_m"]
        for row in enriched
        if np.isfinite(row["cop_margin_m"])
    ]
    return {
        "accepted": bool(enriched) and violations == 0,
        "active_contact_row_count": len(enriched),
        "violation_count": int(violations),
        "minimum_normal_force_n": min(
            (row["normal_force_n"] for row in enriched), default=float("nan")
        ),
        "maximum_friction_utilisation": max(
            finite_utils, default=float("nan")
        ),
        "minimum_cop_margin_m": min(finite_margins, default=float("nan")),
        "maximum_tangential_speed_m_s": max(
            (row["tangential_speed_m_s"] for row in enriched),
            default=float("nan"),
        ),
        "rows": enriched,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _scalar_text(value) -> str:
    array = np.asarray(value)
    if array.size != 1:
        raise ValueError("metadata field must be scalar")
    return str(array.reshape(()).item())


def promote_candidate_reference(
    candidate_path,
    final_path,
    *,
    expected_sha256: str,
    contact_report: Mapping,
) -> dict:
    """Atomically promote a validated planning candidate after contact PASS."""

    candidate = Path(candidate_path)
    final = Path(final_path)
    if not bool(contact_report.get("accepted", False)):
        raise RuntimeError("contact feasibility did not pass; final reference withheld")
    actual_sha = _sha256(candidate)
    if actual_sha != str(expected_sha256):
        raise RuntimeError(
            f"candidate SHA-256 mismatch: expected {expected_sha256}, got {actual_sha}"
        )
    with np.load(candidate, allow_pickle=False) as archive:
        payload = {name: archive[name].copy() for name in archive.files}
    if _scalar_text(payload.get("schema_version")) != "fddp-multistep-v1":
        raise RuntimeError("candidate schema is not fddp-multistep-v1")
    if (
        _scalar_text(payload.get("acceptance_state"))
        != "planning_accepted_contact_pending"
    ):
        raise RuntimeError("candidate planning acceptance state is invalid")
    if int(np.asarray(payload.get("step_count")).reshape(()).item()) != 10:
        raise RuntimeError("candidate is not the required ten-step reference")
    if _scalar_text(payload.get("tau_ref_hardware_usage")) != "analysis_only":
        raise RuntimeError("candidate violates the tau_ref hardware-use boundary")
    payload["acceptance_state"] = np.array("planning_and_contact_accepted")
    final.parent.mkdir(parents=True, exist_ok=True)
    temporary_name = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{final.stem}.", suffix=".npz", dir=final.parent, delete=False
        ) as stream:
            temporary_name = stream.name
        np.savez(temporary_name, **payload)
        os.replace(temporary_name, final)
        temporary_name = None
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)
    return {
        "source_path": str(candidate),
        "final_path": str(final),
        "source_sha256": actual_sha,
        "final_sha256": _sha256(final),
        "acceptance_state": "planning_and_contact_accepted",
    }
