# URDF and MuJoCo Consistency Audit

## Scope and claim boundary

Audit date: 2026-07-13. FDDP/Pinocchio consumes `model/biped_v1.urdf`; MuJoCo replay consumes `model/biped_planar.xml`.
This is a complete mapping ledger for the current simulation files, not a hardware calibration certificate and not proof of full URDF/MJCF equivalence.
No model file was modified by this audit.

## Design ledger

- Units: SI (m, kg, s, rad).
- Axes: +X robot right and pitch axis, +Y forward, +Z upward.
- Sources: current URDF primitives and estimated/proxy inertials; replace estimates with measurements before hardware claims.
- Mapping: 14 MuJoCo bodies cover 16 URDF inertial links; each foot contact lump is rigid-folded into its foot using the parallel-axis theorem.
- Consumer difference: MJCF adds planar root DOFs and simulation-only contact/position-actuator semantics.

## Automated result

- Ledger rows: 154
- Matched numeric/structural rows: 115
- Unexpected mismatches: 0
- Audit integrity passed: `True`
- Full equivalence: `False`
- Zero-pose base height derived from URDF foot-bottom geometry: 0.288 m; matches MJCF.

## Deliberate or consumer-specific differences

| Category | Item | Status | Meaning |
|---|---|---|---|
| collision_enablement | hip_plate_link | deliberate_simplification | feet collide in both; non-foot MJCF collision is intentionally disabled |
| collision_enablement | electronics_lump_link | deliberate_simplification | feet collide in both; non-foot MJCF collision is intentionally disabled |
| collision_enablement | left_hip_motor_link | deliberate_simplification | feet collide in both; non-foot MJCF collision is intentionally disabled |
| collision_enablement | left_thigh_bracket_link | deliberate_simplification | feet collide in both; non-foot MJCF collision is intentionally disabled |
| collision_enablement | left_knee_motor_link | deliberate_simplification | feet collide in both; non-foot MJCF collision is intentionally disabled |
| collision_enablement | left_shank_bracket_link | deliberate_simplification | feet collide in both; non-foot MJCF collision is intentionally disabled |
| collision_enablement | left_ankle_motor_link | deliberate_simplification | feet collide in both; non-foot MJCF collision is intentionally disabled |
| rigid_body_folding | left_foot_link | rigid_body_folded | contact-lump mass/COM/inertia folded into foot; its separate geometry is omitted |
| collision_enablement | right_hip_motor_link | deliberate_simplification | feet collide in both; non-foot MJCF collision is intentionally disabled |
| collision_enablement | right_thigh_bracket_link | deliberate_simplification | feet collide in both; non-foot MJCF collision is intentionally disabled |
| collision_enablement | right_knee_motor_link | deliberate_simplification | feet collide in both; non-foot MJCF collision is intentionally disabled |
| collision_enablement | right_shank_bracket_link | deliberate_simplification | feet collide in both; non-foot MJCF collision is intentionally disabled |
| collision_enablement | right_ankle_motor_link | deliberate_simplification | feet collide in both; non-foot MJCF collision is intentionally disabled |
| rigid_body_folding | right_foot_link | rigid_body_folded | contact-lump mass/COM/inertia folded into foot; its separate geometry is omitted |
| joint_passive_friction | left_hip_pitch_joint | deliberate_mismatch | URDF Coulomb friction is not represented in current MJCF |
| actuator_effort | left_hip_pitch_joint | deliberate_mismatch | URDF effort 2.94 N m; MJCF position actuator is capped at +/-2 N m |
| actuator_velocity | left_hip_pitch_joint | deliberate_mismatch | URDF velocity limit vs replay proxy command slew limit |
| joint_passive_friction | left_knee_pitch_joint | deliberate_mismatch | URDF Coulomb friction is not represented in current MJCF |
| actuator_effort | left_knee_pitch_joint | deliberate_mismatch | URDF effort 2.94 N m; MJCF position actuator is capped at +/-2 N m |
| actuator_velocity | left_knee_pitch_joint | deliberate_mismatch | URDF velocity limit vs replay proxy command slew limit |
| joint_passive_friction | left_ankle_pitch_joint | deliberate_mismatch | URDF Coulomb friction is not represented in current MJCF |
| actuator_effort | left_ankle_pitch_joint | deliberate_mismatch | URDF effort 2.94 N m; MJCF position actuator is capped at +/-2 N m |
| actuator_velocity | left_ankle_pitch_joint | deliberate_mismatch | URDF velocity limit vs replay proxy command slew limit |
| joint_passive_friction | right_hip_pitch_joint | deliberate_mismatch | URDF Coulomb friction is not represented in current MJCF |
| actuator_effort | right_hip_pitch_joint | deliberate_mismatch | URDF effort 2.94 N m; MJCF position actuator is capped at +/-2 N m |
| actuator_velocity | right_hip_pitch_joint | deliberate_mismatch | URDF velocity limit vs replay proxy command slew limit |
| joint_passive_friction | right_knee_pitch_joint | deliberate_mismatch | URDF Coulomb friction is not represented in current MJCF |
| actuator_effort | right_knee_pitch_joint | deliberate_mismatch | URDF effort 2.94 N m; MJCF position actuator is capped at +/-2 N m |
| actuator_velocity | right_knee_pitch_joint | deliberate_mismatch | URDF velocity limit vs replay proxy command slew limit |
| joint_passive_friction | right_ankle_pitch_joint | deliberate_mismatch | URDF Coulomb friction is not represented in current MJCF |
| actuator_effort | right_ankle_pitch_joint | deliberate_mismatch | URDF effort 2.94 N m; MJCF position actuator is capped at +/-2 N m |
| actuator_velocity | right_ankle_pitch_joint | deliberate_mismatch | URDF velocity limit vs replay proxy command slew limit |
| actuator_semantics | six leg joints | consumer_specific | replay q_cmd proxy is not a calibrated hardware actuator model |
| root_dofs | planar floating base | consumer_specific | MJCF adds +Y, +Z and +X planar root coordinates |
| contact_parameters | foot-ground contact | mjcf_only_uncalibrated | MuJoCo-only simulation parameters; not hardware-calibrated |
| contact_sensor | left_toe_touch | mjcf_only_runtime_feature | natural-contact C2 uses this MuJoCo touch sensor |
| contact_sensor | left_heel_touch | mjcf_only_runtime_feature | natural-contact C2 uses this MuJoCo touch sensor |
| contact_sensor | right_toe_touch | mjcf_only_runtime_feature | natural-contact C2 uses this MuJoCo touch sensor |
| contact_sensor | right_heel_touch | mjcf_only_runtime_feature | natural-contact C2 uses this MuJoCo touch sensor |

## Verified matches

Per mapped body, the ledger checks mass, COM/inertial origin, inertia tensor, local body origin, primitive box size/origin, and collision enablement. For all six leg joints it checks +X axis, range, damping, and actuator position range. Toe/heel sites are checked against the foot footprint and bottom plane.

## Remaining calibration boundary

The current MJCF intentionally disables non-foot collision, omits URDF joint Coulomb friction, uses a +/-2 N m position-actuator cap instead of the URDF 2.94 N m effort limit, and uses an 8 rad/s replay-proxy slew limit instead of the URDF 4.712 rad/s velocity limit. Friction, solref/solimp, delay, lag, actuator gains, and touch sensing are simulation parameters and are not hardware calibrated.

Therefore the supported statement is: the current files have a complete, reproducible consistency ledger with no unexpected mismatches under its documented mappings; they are not fully equivalent physical models.

## Machine-readable artifacts

- `results/logs/urdf_mjcf_consistency_audit.csv`
- `results/logs/urdf_mjcf_consistency_audit.yaml`
