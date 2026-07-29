# Thesis-Grade Multi-Step FDDP Simulation Closure Design

Date: 2026-07-14  
Status: approved architecture approach; implementation plan pending user review  
Workspace: `/home/zhijian/biped_fddp_project`

## 1. Objective

Complete a thesis-grade, simulation-only multi-step biped control loop using an incremental gated-evidence architecture. The final system must generate at least ten alternating FDDP steps, replay them through the MuJoCo position-command path, run a combined C2-final controller, survive the declared robustness envelope with statistical evidence, fail safely under injected software faults, and produce reproducible thesis artifacts.

The supported hardware-facing command remains:

`q_cmd = q_ref + delta_q`

`tau_ref` remains FDDP planning and analysis output only. Real-hardware autonomous walking and hardware torque control are outside this design.

## 2. Current Baseline

The project already has:

- converged double-support FDDP CoM shift evidence;
- left- and right-swing single-step FDDP feasibility evidence;
- a converged two-step DS-left-SS-DS-right-SS-DS FDDP trajectory;
- MuJoCo position-command replay and walking animation;
- two-step-specific C1 FIR-QP pitch recovery;
- natural toe/heel phase-only C2 retiming;
- a 70-trial C0/C2 robustness matrix;
- model consistency, parameter ablation, timing, safety, and regression-test evidence.

Known gaps that this design closes:

- no ten-step FDDP gait or long-horizon continuity evidence;
- two-step C1 and natural-contact phase retiming are not integrated into one final C2 path;
- current two-step robustness excludes C1 and combined C2-final;
- +4 N push recovery currently fails 0/5;
- no complete FDDP contact-force, friction-cone, CoP, or slip report;
- insufficient fault injection and deterministic fallback evidence;
- no high-sample statistical matrix or repeated-run reproducibility gate.

## 3. Selected Architecture

Use an incremental gated-evidence ladder:

1. solve and validate one periodic two-step FDDP cycle;
2. measure cycle-boundary state, control, contact, and pose continuity;
3. use the accepted cycle as a FDDP-generated continuation seed for 4, 6, and 10 steps;
4. export only when every solver, dynamics, contact, landing, clearance, and joint gate passes;
5. replay the ten-step reference in MuJoCo using position commands;
6. expose one shared controller interface for C0, C1, phase-only, and C2-final;
7. add bounded recovery authority without changing the position-control boundary;
8. execute fault, timing, ablation, and batch-robustness campaigns;
9. archive all accepted and failed evidence with explicit claim boundaries.

The rejected alternatives are a monolithic unseeded ten-step solve and online receding-horizon FDDP. The former is unnecessarily difficult to diagnose; the latter exceeds the intended MSc real-time scope.

## 4. Protected Existing Evidence

The following historical evidence must not be overwritten or re-labelled:

- `fddp/03n_solve_damped_6d_full_continuation.py`;
- `simulation/03o_replay_fddp_com_shift_mujoco.py`;
- `mpc/12_validate_scripted_alternating_step_locomotion.py`;
- `mpc/13_run_scripted_robustness_matrix.py`;
- existing left-, right-, and two-step trajectory artifacts.

New work uses independent versioned modules and output prefixes. Stable helpers may be imported from existing files, but historical scripts and results remain reproducible.

## 5. Planning Components

### 5.1 Periodic cycle builder

Create a new periodic-cycle module that starts from the accepted static double-support state and constructs a left/right two-step cycle. It must report:

- initial and terminal floating-base state difference;
- six-joint position and velocity boundary difference;
- left/right foot pose boundary difference;
- CoM position and velocity boundary difference;
- contact-mode compatibility at the cycle boundary;
- control discontinuity at the boundary.

The cycle is accepted only when it can be continued without an artificial state reset. After removing the planned forward translation, cycle-boundary tolerances are: joint-position difference <= 1e-3 rad, joint-velocity difference <= 1e-2 rad/s, base height and pitch difference <= 1e-3, CoM-velocity difference <= 1e-3 m/s, relative foot-placement difference <= 1e-3 m and 0.2 deg, and control discontinuity <= 0.10 in the exported analysis-control units.

### 5.2 Multi-step continuation

Build 4-, 6-, and 10-step problems in sequence. Each larger problem uses only an accepted FDDP-generated shorter trajectory as its seed. Stage 8 scripted trajectories are forbidden as warm starts.

Each planning segment must satisfy:

- `solver.stop <= 1e-7`;
- maximum shooting gap `<= 1e-7`;
- active-support translation error `<= 5e-4 m`;
- active-support rotation error `<= 0.5 deg`;
- landing translation error `<= 1e-3 m`;
- landing rotation error `<= 1.0 deg`;
- swing clearance `>= 0.010 m`;
- joint-limit margin `>= 0.010 rad`.

The final ten-step nominal must last at least 10 s and achieve at least 0.040 m forward displacement.

### 5.3 Gated export

Failed solves always retain diagnostics and iteration histories. They must not export a nominal trajectory, update a success manifest, or overwrite the last accepted reference.

The accepted nominal export contains schema-versioned:

- `t_ref`, `q_ref`, `dq_ref`, `state_ref`, and analysis-only `tau_ref`;
- left/right contact flags and phase names;
- foot-pose and CoM references;
- cycle identifiers and segment provenance;
- solver and acceptance metadata;
- `tau_ref_hardware_usage=analysis_only`.

## 6. Contact Feasibility

Extract the active-contact wrench at every FDDP node and compute:

- normal force;
- tangential force magnitude;
- friction utilisation `|F_t| / (mu F_n)`;
- unilateral-contact violation;
- centre of pressure;
- minimum CoP margin to the foot support polygon;
- predicted slip margin;
- left/right support-load distribution during double support.

Acceptance requires:

- normal force >= -1e-5 N, treating only smaller negative values as numerical tolerance;
- maximum friction utilisation `<= 0.90`;
- minimum CoP polygon margin `>= 0.002 m`;
- active-support tangential speed <= 1e-3 m/s, with any larger value classified as slip;
- identical documented friction assumptions between each planning report and its replay scenario.

Contact-feasibility failure withholds nominal export even when pose and shooting-gap gates pass.

## 7. MuJoCo Replay

Create an independent multi-step replay module that consumes only the new gated multi-step NPZ. It uses the existing position-actuator-proxy semantics and never sends `tau_ref` to MuJoCo controls.

Nominal ten-step replay acceptance requires:

- no fall;
- maximum absolute torso pitch `<= 15 deg`;
- maximum joint RMSE `<= 4 deg`;
- minimum torso height `>= 65%` of initial height;
- no sustained penetration, unintended flight, or invalid contact ordering;
- scheduled-to-observed per-leg event match ratio `>= 0.90`;
- all liftoff/touchdown event kinds appear in the declared order.

Global contact count is retained as a diagnostic but is not accepted as a substitute for per-leg event evidence.

## 8. Controller Architecture

Define one shared, testable controller interface with four modes:

- C0: fixed-time reference-only position replay;
- C1: fixed-time FIR-QP pitch correction;
- phase-only: natural-contact phase retiming without pitch correction, used only for ablation;
- C2-final: C1 correction and natural-contact phase retiming operating simultaneously.

C2-final must load the new multi-step FDDP reference. Loading an older CoM-shift reference is an integrity failure for final multi-step experiments.

Every update returns:

- `q_cmd`, `q_ref`, and `delta_q`;
- wall time and reference time;
- pitch error and correction `alpha`;
- phase offset and matched/skipped event counters;
- limiter, saturation, deadline, and fallback status;
- a machine-readable controller and schema version.

Performance gates under the common pitch-disturbance protocol are:

- C1 IAE improvement over C0 `>= 10%`;
- C1 peak-error increase over C0 `<= 5%`;
- C2-final IAE <= 1.02 * C1 IAE and C2-final peak error <= 1.05 * C1 peak error;
- C2-final reduces early/late-contact timing error by at least 30% relative to fixed-phase C1;
- `|alpha| <= 5 deg`;
- every output remains within joint-position and rate limits.

## 9. Recovery Authority

The existing scalar FIR-QP correction may be insufficient for +4 N push recovery. Add recovery authority incrementally and keep it position-command based:

1. retune the bounded C1 correction using multi-step identification;
2. test phase-aware gain scheduling;
3. add bounded CoM or foothold adjustment only if the first two are insufficient;
4. prevent any recovery term from bypassing joint, rate, contact, or landing gates.

Any adaptive foothold offset must remain within +/-0.015 m of the nominal foothold in the forward direction and must be represented in both the reference metadata and contact-feasibility report.

If +4 N recovery requires changing mechanical geometry, actuator torque authority, or unknown hardware parameters, the work stops for user direction. The threshold is not silently reduced.

## 10. Safety and Fault Handling

Introduce a deterministic runtime state machine with normal, degraded, fallback, safe-hold, and failed states.

Required injected faults:

- stale pitch measurement;
- NaN or Inf measurement;
- toe/heel sensor stuck on or stuck off;
- missing or out-of-order contact event;
- QP non-convergence or invalid solution;
- controller deadline miss;
- command and rate saturation;
- increased actuator delay or lag;
- missing, truncated, or schema-invalid trajectory files.

Required behaviour:

- no uncaught controller exception;
- no NaN or Inf command;
- no command outside position or rate limits;
- invalid online correction falls back deterministically to C0 or safe hold;
- file corruption prevents controller start;
- every transition records time, reason, previous state, next state, and recovery outcome.

## 11. Robustness and Statistics

Run at least 30 deterministic seeds per scenario for C0, C1, phase-only, and C2-final where applicable.

The matrix includes:

- nominal;
- forward and backward 2 N and 4 N pushes;
- friction `mu=0.40`, `0.50`, and `0.80`;
- slopes `-5`, `-3`, `+3`, and `+5 deg`;
- steps `2`, `4`, and `6 mm`;
- command delays `4`, `12`, and `20 ms`;
- sensor delays `0`, `20`, and `40 ms`;
- actuator gain uncertainty of +/-20% and actuator time constants from 20 to 50 ms;
- initial joint noise with standard deviation 0.002 rad and initial pitch noise with standard deviation 0.5 deg;
- combined case A: mu=0.50, +3 deg slope, and 12 ms command delay;
- combined case B: 4 mm step, 20 ms sensor delay, and +2 N push;
- combined case C: mu=0.40, -3 deg slope, and +4 N push.

Acceptance requires:

- nominal C2-final: `30/30`;
- every critical scenario success rate >= 90%; critical means each maximum single-factor case plus combined cases A, B, and C;
- forward and backward 4 N push recovery `>= 90%`;
- C2-final aggregate success rate across the declared matrix >= 95%;
- Wilson 95% confidence interval for every scenario and aggregate;
- no deleted or suppressed failed seeds;
- a machine-readable failure category for every failed trial.

Passing means satisfying explicit height, pitch, contact, event, command, and controller-integrity gates. It does not mean hardware validation.

## 12. Timing

On the configured `/home/zhijian/miniforge3/envs/fddp_biped/bin/python` environment, benchmark at least 100,000 controller updates per final controller mode.

Acceptance requires:

- controller period `20 ms`;
- p99 controller computation `< 5 ms`;
- maximum controller computation `< 20 ms`;
- zero unexplained deadline misses;
- controller computation time reported separately from MuJoCo physics and rendering time.

## 13. Software Quality and Reproducibility

Completion requires:

- targeted and fresh full tests all pass;
- meaningful branch coverage of new core modules `>= 85%`;
- unit, integration, fault-injection, schema, replay, and batch-aggregation tests;
- dynamic loaders register modules in `sys.modules` before execution;
- explicit and versioned NPZ, CSV, and YAML schemas;
- deterministic random seeds stored in trial artifacts;
- two clean repeated pipeline runs produce identical pass/fail decisions and metrics within declared numeric tolerances;
- no silent fallback, silent export, or success status that contradicts diagnostics;
- no unresolved P0 or P1 defect.

Defect severity:

- P0: crash, result corruption, unsafe command, or invalid success export;
- P1: incorrect acceptance logic, invalid contact schedule, non-reproducible evidence, unexplained model mismatch, or claim/result contradiction.

## 14. Artifact and Animation Contract

The completed experiment produces independent, schema-versioned:

- gated multi-step NPZ and summary YAML;
- solver convergence and feasibility CSV/YAML/PNG;
- contact-force, friction, CoP, and slip CSV/YAML/PNG;
- MuJoCo nominal replay CSV/YAML/PNG;
- C0/C1/phase-only/C2-final ablation CSV/YAML/PNG;
- robustness trial and aggregate tables with confidence intervals;
- timing and fault-injection reports;
- nominal walking GIF or MP4;
- successful 4 N recovery GIF or MP4;
- representative failure GIF or MP4;
- experiment record with commands, interpreter, configs, seeds, thresholds, paths, and claim boundaries.

All accepted and failed results are mirrored after final verification to:

`D:\毕业论文\机器人控制流程图\FDDP仿真结果归档`

The archive sync must regenerate and verify `SHA256_MANIFEST.csv`.

## 15. Testing Order

Every production change follows RED to GREEN:

1. pure schedule, schema, gate, or state-machine tests;
2. targeted module tests;
3. small MuJoCo smoke tests;
4. solver or controller experiment;
5. artifact validation;
6. full test suite;
7. repeated-run reproducibility check;
8. D-drive archive and SHA-256 verification.

Long batch runs begin only after their small deterministic smoke cases pass.

## 16. Error and Stop Policy

Failures remain evidence. A failed solver, replay, controller, or scenario must retain diagnostics and must not be rewritten as success.

Stop and request user direction when:

- the declared 4 N recovery threshold requires more actuator authority than the current model allows;
- a proposed fix requires mechanical, electrical, or hardware-interface changes;
- required hardware calibration data are unavailable and would change any acceptance decision beyond its declared tolerance;
- meeting a threshold requires changing an approved claim boundary;
- two safe software approaches produce opposite pass/fail decisions or conflicting thesis claims.

## 17. Completion Claim

Completion supports the statement:

"Within the declared simulation model, controller limits, scenario matrix, and statistical sample size, the project provides a reproducible thesis-grade multi-step FDDP position-control pipeline with integrated pitch correction, natural-contact phase synchronisation, contact-feasibility evidence, bounded recovery, fault handling, and archived validation artifacts."

Completion does not support real-hardware autonomous walking, hardware torque control, universal stability, or robustness outside the tested envelope.
