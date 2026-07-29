# 2026-07-13 FDDP Two-Step Alternating Experiment

## Scope and claim boundary
This record covers simulation-only FDDP two-step alternating gait evidence.
It does not claim real-hardware autonomous walking. Hardware command semantics remain position-only: q_cmd = q_ref + delta_q. tau_ref is planning and analysis output only.

## Interpreter
/home/zhijian/miniforge3/envs/fddp_biped/bin/python

## Commands
- python fddp/04c_solve_fddp_two_step_alternating.py
- python simulation/03r_replay_fddp_two_step_alternating_mujoco.py
- python mpc/14_validate_fddp_two_step_natural_contact_c2.py
- python mpc/15_run_fddp_two_step_robustness_matrix.py
- python simulation/03s_render_fddp_two_step_animation.py
- python mpc/16_validate_fddp_two_step_c1_fir_qp.py
- python simulation/03t_render_fddp_two_step_c1_disturbance_animation.py
- python mpc/17_run_fddp_two_step_c1_parameter_ablation.py
- python model/02_audit_urdf_mjcf_consistency.py
- python -m unittest discover -s tests -v

## FDDP schedule and result
Schedule: DS(20) -> right-stance/left-swing SS(70) -> DS(20) -> left-stance/right-swing SS(70) -> DS(20), DT=0.010 s.
Solver: 166 iterations, stop=2.8386e-8, max shooting gap=0.
Support translation error=1.573e-5 m; first/second landing translation errors=1.320e-5 m and 4.598e-8 m.
Left/right swing clearance=0.0199992 m / 0.0199985 m.
Joint margin=0.0119921 rad. CoM forward displacement=0.0081180 m.
All two-step FDDP gates passed and gated NPZ was exported.

## Replay and C2
Position-command replay: max joint RMSE=2.7067 deg, max pitch=10.5953 deg, min global contacts=2, integrity passed.
Natural toe/heel C2: 4 scheduled and 4 observed events; 3 matched and 1 skipped; max phase offset=47.375 ms.
This demonstrates natural-event-driven phase retiming, not perfect event matching.

## Robustness
70 randomized trials: 7 scenarios x 5 seeds x C0/C2.
Nominal, mu=0.50 friction, +3 deg slope, 4 mm step, 12 ms command delay, and 20 ms sensor delay were 5/5 in both modes.
+4 N push was 0/5 in both C0 and C2; this negative result is retained.
For a 5/5 condition the Wilson 95 percent lower bound is 0.5655, so evidence is small-sample simulation robustness only.

## Artifacts
- transfer/trajectories/fddp_two_step_alternating_ref.npz
- transfer/trajectories/fddp_two_step_alternating_ref_summary.yaml
- results/logs/fddp_two_step_alternating_diagnostics.yaml
- results/figures/fddp_two_step_alternating_com_clearance.png
- results/logs/fddp_two_step_alternating_mujoco_replay.yaml
- results/figures/fddp_two_step_alternating_mujoco_replay.png
- results/logs/fddp_two_step_natural_contact_c2.yaml
- results/figures/fddp_two_step_natural_contact_c2.png
- results/logs/fddp_two_step_robustness_trials.csv
- results/logs/fddp_two_step_robustness_aggregate.csv
- results/logs/fddp_two_step_robustness_summary.yaml
- results/figures/fddp_two_step_robustness_success_rates.png
- results/figures/fddp_two_step_mujoco_position_command_replay.gif
- results/figures/fddp_two_step_c1_pitch_disturbance.gif
- results/logs/fddp_two_step_c1_parameter_ablation.csv
- results/logs/fddp_two_step_c1_parameter_ablation_traces.csv
- results/logs/fddp_two_step_c1_parameter_ablation.yaml
- results/figures/fddp_two_step_c1_parameter_ablation.png
- results/logs/urdf_mjcf_consistency_audit.csv
- results/logs/urdf_mjcf_consistency_audit.yaml
- docs/2026-07-13_urdf_mujoco_consistency_audit.md

## Verification
Full unittest discovery with -s tests: 51/51 passed before the disturbance-animation regression test was added. A fresh final count is recorded below.


## Two-step C1 FIR-QP update
Two-step-specific inputs were generated before C1 execution: same-reference baseline CSV, 1 degree/10 ms pulse FIR YAML, C0 root-pitch disturbance CSV, and stabilized AR(3) YAML.
Balanced FIR-QP candidate: q=1500, r=20, slew=25.
C0 disturbance: peak=2.3788 deg, IAE=0.4304 deg s.
C1 disturbance: peak=2.4420 deg, IAE=0.3785 deg s, max alpha=2.6830 deg, min contacts=2.
C1 IAE improved about 12.1 percent; peak increased about 2.7 percent, within the existing 5 percent peak-increase tolerance.
This was initially a two-step simulation C0/C1 comparison. The common-disturbance C0/C1/C2 protocol reported below was subsequently run, so C2 remains separately interpreted using those later results.
Additional artifacts:
- results/logs/fddp_two_step_c1_baseline.csv
- results/logs/fddp_two_step_c1_fir_identification.yaml
- results/logs/fddp_two_step_c1_c0_pitch_disturbance.csv
- results/logs/fddp_two_step_c1_ar_fit.yaml
- results/logs/fddp_two_step_c1_fir_qp.csv
- results/logs/fddp_two_step_c1_fir_qp.yaml
- results/figures/fddp_two_step_c0_c1_pitch.png


## Same-disturbance C0/C1/C2 comparison update
Common root-pitch disturbance protocol: +2 deg and +0.25 rad/s at 0.4 s, same two-step FDDP q_ref and position proxy.
C0: peak=2.3788 deg, IAE=0.4304 deg s.
C1 balanced FIR-QP: peak=2.4420 deg, IAE=0.3785 deg s.
C2 natural-event phase retiming: peak=2.7091 deg, IAE=0.4834 deg s, matched events=3, max phase offset=49.0 ms, min contacts=2.
Interpretation: C1 reduced IAE for this pitch pulse. C2 demonstrated natural contact-event retiming but did not improve this pitch-pulse metric; it must not be claimed as superior recovery under this protocol.
Additional artifacts:
- results/logs/fddp_two_step_c2_pitch_disturbance.csv
- results/logs/fddp_two_step_c2_pitch_disturbance.yaml
- results/figures/fddp_two_step_c0_c1_c2_pitch.png

## Disturbance animation update
The C1 position-command replay under the same +2 deg, +0.25 rad/s pitch disturbance was rendered as a dedicated MuJoCo GIF.
Integrity metadata: 51 frames, 480 x 360 pixels, 40 ms per frame (approximately 2.04 s).
This animation visualises simulation-only C1 pitch-disturbance recovery; it is not hardware walking evidence.
Artifact:
- results/figures/fddp_two_step_c1_pitch_disturbance.gif

## Final verification
Targeted disturbance-animation regression test: 1/1 passed. Fresh full unittest discovery with -s tests: 52/52 passed.

## C1 key-parameter ablation update
One-factor MuJoCo ablation used the same two-step FDDP q_ref, FIR/AR models, position-command proxy, and +2 deg/+0.25 rad/s pitch disturbance.
Candidates: balanced plus q low/high, r low/high, and slew-penalty low/high (7 total). All 7 passed the declared two-step contact, torso-height, command, IAE-improvement, and peak-increase gates.
Balanced: IAE=0.3785 deg s, peak=2.4420 deg, IAE improvement=12.06 percent.
Lowest observed IAE: slew_penalty_low (q=1500, r=20, slew=12), IAE=0.3688 deg s and IAE improvement=14.31 percent.
This bounded one-factor sweep is sensitivity evidence, not a claim of global optimality.

## Complete URDF/MJCF mapping audit update
The reproducible ledger maps 14 MuJoCo bodies to all 16 URDF inertial links, rigid-folding each foot contact lump into its foot with the parallel-axis theorem, and audits all 6 leg joints.
Result: 154 ledger rows, 115 numeric/structural matches, 0 unexpected mismatches, audit integrity passed.
Retained differences: 18 deliberate actuator/passive-friction mismatches, 12 non-foot collision simplifications, 2 consumer-specific definitions, 4 MuJoCo-only touch sensors, and 1 MuJoCo-only uncalibrated contact-parameter row.
Therefore full_equivalence=false. Contact, actuator force/speed, delay/lag, and touch parameters remain simulation semantics rather than hardware calibration evidence.

## Latest final verification
Targeted ablation tests: 4/4 passed. Targeted consistency-audit tests: 4/4 passed. Fresh full unittest discovery with -s tests: 60/60 passed with clean exit 0.

## D-drive result archive policy
All existing and future result artifacts are mirrored after final verification to:
D:\毕业论文\机器人控制流程图\FDDP仿真结果归档

Included source trees: results/, transfer/trajectories/, and docs/.
Initial full sync archived and SHA-256 verified 252 files (92,531,652 bytes).
The archive root contains SHA256_MANIFEST.csv; the repeatable command is tools/sync_thesis_results_to_d_drive.ps1.
The workspace remains the source of truth, and the archive policy does not broaden simulation or hardware claims.
