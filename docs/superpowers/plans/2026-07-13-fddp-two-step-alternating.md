# FDDP Two-Step Alternating Implementation Plan

Goal: Build and validate a gated two-step FDDP DS-left-SS-DS-right-SS-DS simulation experiment and an independent MuJoCo position-command replay.

Architecture: fddp/04c_solve_fddp_two_step_alternating.py owns fixed phase construction, contacts, costs, diagnostics, and gated export. simulation/03r_replay_fddp_two_step_alternating_mujoco.py consumes only the exported reference and delegates position actuator behaviour to the existing replay proxy.

Tech Stack: Python 3.10, Pinocchio, Crocoddyl FDDP, MuJoCo, NumPy, Pandas, Matplotlib, PyYAML, unittest.

## Global Constraints
- Do not modify Stage 7C fddp/03n_solve_damped_6d_full_continuation.py, Stage 8 scripted gait, or replay simulation/03o_replay_fddp_com_shift_mujoco.py.
- Start from static x0, not a scripted gait warm start.
- World convention is +Y forward, +Z vertical, +X pitch.
- Hardware is out of scope. tau_ref is analysis only; replay uses position commands only.
- Every nominal export is conditional on acceptance gates.

## Task 1: Two-step phase and reference contract
Files: create tests/test_fddp_two_step_alternating.py and fddp/04c_solve_fddp_two_step_alternating.py.
Write tests for a 200-node [20,70,20,70,20] schedule with phase names double_support_start, right_stance_left_swing, double_support_mid, left_stance_right_swing, double_support_landing. Verify active contact side in each SS, 10 mm +Y advance for each swing, and clearance rejection below 10 mm. Run unittest and observe module-missing RED. Implement only constants, PhaseSpec, schedule, quintic progress, validation, and swing-pose helpers. Re-run targeted tests GREEN.

## Task 2: FDDP problem and gated export
Add failing tests that inspect action models at nodes 20 and 110 for right-only then left-only 6D contact, and require an SE(3) swing residual for the uncontacted foot. Implement build_two_step_problem(model,x0), static seed, diagnostics, and acceptance. Require two landing pose diagnostics, two clearances, joint margin, gap, support error, and CoM progress. Export only fddp_two_step_alternating_ref.npz and matching logs/figure on pass. Run targeted tests GREEN, then solver. If any gate fails, record diagnostics only and stop before replay.

## Task 3: Independent replay with per-foot contact evidence
Create tests/test_fddp_two_step_alternating_replay.py and simulation/03r_replay_fddp_two_step_alternating_mujoco.py. Its load_reference() returns t_ref,q_ref,dq_ref,com_ref,phase. The test requires five phase labels and tau_ref usage boundary, first RED then GREEN. Replay writes independent CSV/YAML/PNG and logs joint RMSE, pitch, height, global contacts, and per-foot toe/heel contacts. Summary says MuJoCo position-command replay, not hardware walking.

## Task 4: Regression and evidence record
Run all single-step and two-step targeted tests, then the full unittest suite. Record interpreter, commands, schedule, gates, replay metrics, artifact paths, and simulation-only boundary in a new two-step experiment document. Do not choose natural-contact C2 integration until solver and replay results are available.

## Self-Review
The plan separates phase contract, solver/export, replay, and evidence. It does not modify protected Stage 7C, Stage 8, or 03o files; it does not use scripted warm start or permit hardware/torque-command claims.
