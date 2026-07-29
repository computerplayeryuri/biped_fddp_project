# Valid Known-Obstacle Robustness Design

Date: 2026-07-19
Status: approved and implemented
Workspace: `/home/zhijian/biped_fddp_project`

## 1. Objective

Close the remaining 6 mm robustness failure without weakening any acceptance threshold and without creating a second planner unnecessarily. The scenario must represent a real future obstacle, must not intersect either foot at initialisation, must be encountered during the ten-step replay, and must retain the simulation-only position-command claim boundary.

## 2. Evidence That Changes the Earlier Plan

The current `build_step_xml()` places two narrow boxes at sagittal centre `y=0.068 m`, each with half-length `0.012 m`. The nominal initial foot centre is `y=0.0 m` and the foot half-length is `0.0725 m`, so the obstacle rear edge is `0.056 m` while the initial toe reaches `0.0725 m`. At time zero, each obstacle overlaps its foot by 6 mm. MuJoCo reports 16 initial contacts, including eight obstacle-foot contacts with `dist=-0.006 m`.

This is an invalid initial condition, not a valid step-up landing. It also does not match the FDDP model, which assumes a flat 6D sole contact. Therefore the earlier proposal to add a 6 mm landing-height target to FDDP would solve the wrong problem.

A read-only position scan moved the obstacle centre to `y=0.085...0.125 m`, beyond the initial toe. With the existing nominal FDDP reference and existing C2 controller, every tested placement produced event match ratio `1.0`, valid event order, zero unintended flight, and peak pitch error between approximately `4.48 deg` and `6.91 deg`.

## 3. Selected Approach

Repair the existing terrain injector and keep the current FDDP reference unchanged.

The obstacle centre will be derived from geometry rather than retained as the invalid magic value:

`obstacle_center_y = initial_foot_center_y + foot_half_length_y + initial_clearance + obstacle_half_length_y`

Use:

- initial foot centre: `0.0 m`;
- foot half-length: `0.0725 m`, taken from the existing model contract;
- obstacle half-length: `0.012 m`, preserving the existing narrow bump;
- initial clearance: `0.005 m`;
- resulting centre: `0.0895 m`.

This makes the rear obstacle edge `0.0775 m`, leaving 5 mm clearance beyond the initial toe. The ten-step gait still encounters the obstacle because each foot advances and its toe crosses this interval during the replay.

The scenario will be described as a known sagittal toe obstacle or bump, not as a full raised landing platform. A true step-up platform remains outside this change because it requires a compatible terrain contact model, terrain-specific initial state, and new FDDP/IK reference generation.

## 4. Components and Data Flow

### 4.1 Existing terrain injector

Modify only `mpc/13_run_scripted_robustness_matrix.py` to calculate obstacle placement from explicit geometry constants. Preserve the existing XML construction, friction assignment, obstacle size, height variants, and MuJoCo model-loading path.

### 4.2 Initial-condition integrity check

Add a reusable geometry check that reports the initial toe edge, obstacle rear edge, and clearance. Reject configurations with negative or insufficient clearance before running a campaign.

The physical integration test must also initialise the existing ten-step reference in the 6 mm model and verify that no foot-obstacle contact or penetration exists at time zero.

### 4.3 Encounter evidence

Extend the existing multi-step runner in `mpc/18_validate_fddp_multistep_c2_final.py` to count contacts between either foot geometry and either obstacle geometry. Keep floor contacts separate.

Report at least:

- `step_obstacle_contact_detected`;
- `step_obstacle_contact_sample_count`;
- first obstacle-contact time;
- maximum obstacle penetration depth.

For any scenario with `step_height_m > 0`, absence of obstacle contact is a failed trial. This prevents a false pass caused by moving the obstacle out of the robot path.

### 4.4 Statistical runner

Reuse `mpc/21_run_fddp_multistep_robustness_matrix.py`. Add one failure category, `step_obstacle_not_encountered`, without changing the existing pitch, height, flight, event, saturation, or command gates. Failed trials remain preserved.

## 5. Testing Strategy

Follow red-green-refactor and use real model geometry:

1. Unit test that the generated 2/4/6 mm obstacle rear edge has at least 5 mm clearance beyond the initial toe.
2. Unit test that invalid clearance is rejected.
3. MuJoCo integration test that the nominal initial pose has no foot-obstacle contact and no obstacle penetration.
4. Replay integration test that the 6 mm obstacle is encountered after time zero.
5. Three-seed Task9 smoke matrix; acceptance target is 32/32 scenarios and 96/96 trials.
6. Only after the smoke matrix passes, execute 30 seeds per scenario and apply the existing Wilson and aggregate gates.

No acceptance threshold will be relaxed. The formal matrix must still preserve every failure and its machine-readable category.

## 6. Outputs

Regenerate the existing Task9 evidence paths rather than inventing parallel report formats:

- `results/logs/fddp_multistep_robustness_trials.csv`;
- `results/logs/fddp_multistep_robustness_aggregate.csv`;
- `results/logs/fddp_multistep_robustness_summary.yaml`;
- `results/figures/fddp_multistep_robustness_matrix.png`.

The summary must record the obstacle geometry, initial clearance, encounter evidence, seed count, and `simulation_only_position_command` claim boundary.

## 7. Non-Goals and Escalation Boundary

This change does not add online terrain perception, inverse kinematics, foothold adaptation, torque control, or real-hardware walking. It also does not claim full-platform step-up capability.

If the corrected, non-intersecting 6 mm obstacle fails the three-seed gate, stop and return to a separate design choice between terrain-specific FDDP contact poses and an online bounded foot/CoM adaptation layer. Do not lower the 6 mm height or remove the scenario silently.

## 8. Acceptance

The design is complete when:

- the initial 6 mm model contains no obstacle-foot contact or penetration;
- the obstacle is encountered later in every step-height trial;
- 2, 4, and 6 mm trials satisfy all unchanged controller and plant gates;
- the 32-scenario three-seed matrix passes before the formal 30-seed campaign begins;
- all new and existing relevant tests pass;
- results continue to state simulation validation, not hardware validation.
