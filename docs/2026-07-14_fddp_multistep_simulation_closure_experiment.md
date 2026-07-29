# 2026-07-14 FDDP Multi-Step Simulation Closure Experiment

## Status

- Overall status: **IN PROGRESS**
- Scope: software and MuJoCo simulation only
- Hardware status: explicitly out of scope until all software/simulation gates pass
- Supported command boundary: `q_cmd = q_ref + delta_q`
- `tau_ref` boundary: FDDP planning/analysis output only; never a hardware or MuJoCo torque command

## Environment

- Workspace: `/home/zhijian/biped_fddp_project`
- Interpreter: `/home/zhijian/miniforge3/envs/fddp_biped/bin/python`
- Platform: WSL Ubuntu-22.04 launched from Windows PowerShell
- Version control: no `.git` repository; checkpoints use tests, artifacts, records, and SHA-256 archival

## Approved Design and Plan

- Design: `docs/superpowers/specs/2026-07-14-thesis-grade-multistep-fddp-simulation-design.md`
- Implementation plan: `docs/superpowers/plans/2026-07-14-thesis-grade-multistep-fddp-simulation.md`
- Execution route: incremental gated evidence (periodic cycle -> 4/6/10-step FDDP -> contact feasibility -> MuJoCo replay -> integrated C2-final -> recovery/fault/statistics/timing)

### Plan self-review correction

The first plan draft incorrectly combined a ten-step schedule with both 1000 nodes and merged cycle-boundary DS phases. The reviewed contract preserves five complete 200-node periodic cycles:

`[DS20, right-SS70, DS20, left-SS70, DS20] x 5`

The terminal DS of one cycle and the initial DS of the next remain consecutive, producing exactly 1000 nodes and 10.0 s. This is a deliberate cycle-boundary settling interval, not an accidental duplicate.

## Baseline Verification

Command:

```bash
/home/zhijian/miniforge3/envs/fddp_biped/bin/python -m unittest discover -s tests -p 'test_*.py' -v
```

Result before new production code: **60 tests passed, 0 failures, 0 errors**. The outer command wrapper timed out after the unittest process had already printed its complete `Ran 60 tests ... OK` result; this is recorded as a wrapper timeout, not a test failure.

## Task 1 - Multi-Step Pure Core

Status: **PASSED**

Created:

- `fddp/multistep_core.py`
- `config/fddp_multistep_experiment.yaml`
- `tests/test_fddp_multistep_core.py`

### RED evidence

Command:

```bash
/home/zhijian/miniforge3/envs/fddp_biped/bin/python -m unittest tests/test_fddp_multistep_core.py -v
```

Observed result: **7 tests errored only because `fddp/multistep_core.py` did not exist**. The dynamic loader itself was valid and registered the module in `sys.modules` before `exec_module()`.

### GREEN evidence

Command:

```bash
/home/zhijian/miniforge3/envs/fddp_biped/bin/python -m unittest tests/test_fddp_multistep_core.py tests/test_fddp_two_step_alternating.py -v
```

Result: **13 tests passed, 0 failures, 0 errors**.

Verified contracts:

| Contract | Result |
|---|---:|
| Ten-step phase blocks | 25 phases / 5 complete cycles |
| Horizon | 1000 nodes |
| Duration | 10.0 s |
| Planned cycle advance | 0.050 m |
| Schema | `fddp-multistep-v1` |
| Config schema | `fddp-multistep-config-v1` |
| Hardware usage of `tau_ref` | `analysis_only` |
| Command interface | `q_cmd=q_ref+delta_q` |
| Threshold boundary semantics | inclusive and tested |
| Invalid/missing/non-finite metrics | rejected |

## Task 2 - Periodic Two-Step FDDP Cycle

Status: **PASSED**

Created:

- `fddp/05a_solve_fddp_periodic_cycle.py`
- `tests/test_fddp_periodic_cycle.py`
- `results/logs/fddp_periodic_cycle_diagnostics.yaml`
- `results/logs/fddp_periodic_cycle_iterations.csv`
- `transfer/trajectories/fddp_periodic_cycle_ref.npz`

### TDD and root-cause sequence

1. Boundary, gate, failed-export, problem, seed, diagnostics, and one-iteration smoke tests were written before their production APIs.
2. The first full solve used a morphed boundary-closed seed and a single `1e8` terminal weight. It failed with `solver.stop=1.70e8`, gap `5.30`, and clearance `0.19 mm`; nominal export was correctly withheld.
3. Diagnostic measurement showed the accepted two-step seed had gap 0 but terminal-state error 0.1608, while the morphed seed had terminal error 0 but gap 5.66.
4. A runtime-only terminal-weight sweep showed that FDDP homotopy `1 -> 10 -> 100 -> 1000` preserved dynamics feasibility and closed the state boundary, but exposed an unmodelled control-boundary discontinuity.
5. The start/end DS quasi-static controls are both zero. A symmetric node-0/node-199 control cost with homotopy `1 -> 10 -> 30` reduced the discontinuity from about 3.20 to 0.033 while preserving first-order convergence.
6. Regression tests were changed to require an unchanged dynamics-feasible FDDP seed, the exact seven-stage homotopy, and symmetric boundary-control costs. The targeted suite then passed `22/22`.

### Formal solve command

```bash
/home/zhijian/miniforge3/envs/fddp_biped/bin/python fddp/05a_solve_fddp_periodic_cycle.py
```

Result: **all periodic and planning gates passed; accepted NPZ exported**.

| Gate | Threshold | Result | Status |
|---|---:|---:|---:|
| Final solver stop | <= 1e-7 | 7.04e-8 | PASS |
| Shooting gap | <= 1e-7 | 0 | PASS |
| Joint-position boundary | <= 1e-3 rad | 1.93e-5 rad | PASS |
| Joint-velocity boundary | <= 1e-2 rad/s | 8.40e-6 rad/s | PASS |
| Base-height boundary | <= 1e-3 m | 7.29e-7 m | PASS |
| Base-pitch boundary | <= 1e-3 rad | 3.64e-5 rad | PASS |
| CoM-velocity boundary | <= 1e-3 m/s | 1.62e-4 m/s | PASS |
| Relative-foot translation | <= 1e-3 m | 9.82e-7 m | PASS |
| Relative-foot rotation | <= 0.2 deg | 0.000265 deg | PASS |
| Control discontinuity | <= 0.10 | 0.0330 | PASS |
| Active-support translation | <= 5e-4 m | 1.80e-5 m | PASS |
| Active-support rotation | <= 0.5 deg | 0.000265 deg | PASS |
| Landing translation | <= 1e-3 m | 1.73e-5 m | PASS |
| Landing rotation | <= 1.0 deg | 0.000265 deg | PASS |
| Swing clearance | >= 0.010 m | 0.019998 m | PASS |
| Joint-limit margin | >= 0.010 rad | 0.011988 rad | PASS |

Homotopy: **7/7 stages converged**, 970 logged iteration rows. Accepted NPZ schema is `fddp-periodic-cycle-v1`; `tau_ref_hardware_usage=analysis_only`; SHA-256 is `2dde3fda0af927138f6f4b5321e99c2e99ef5583ac5132ba13340fda24016228`.

## Task 3 - 4/6/10-Step FDDP Continuation

Status: **HISTORICAL PLANNING PASS; SUPERSEDED BY THE CONTACT-AWARE 2026-07-15 RUN BELOW**

Created:

- `fddp/05b_solve_fddp_multistep_continuation.py`
- `tests/test_fddp_multistep_continuation.py`
- `results/logs/fddp_multistep_planning_diagnostics.yaml`
- `results/logs/fddp_multistep_continuation_iterations.csv`
- `transfer/trajectories/fddp_multistep_4step_accepted_seed.npz`
- `transfer/trajectories/fddp_multistep_6step_accepted_seed.npz`
- `transfer/trajectories/fddp_multistep_10step_planning_candidate.npz`
- `transfer/trajectories/fddp_multistep_10step_planning_candidate_summary.yaml`

### Gate-order correction

The implementation plan was corrected before production work: Task 3 writes only a `planning_accepted_contact_pending` candidate. The final `fddp_multistep_10step_ref.npz` must not exist until Task 4 contact feasibility passes and atomically promotes the candidate. This prevents a planning-only pass from being mislabelled as a complete nominal.

### TDD and root-cause sequence

1. Continuation ladder, provenance rejection, exact segment gates, repeated contact schedule, seed construction, candidate export, multi-landing diagnostics, segment smoke, ladder failure preservation, and solver configuration were tested before their production implementations.
2. The first seed used full forward dynamics rollout to force gap 0. It became numerically unstable after the second cycle, reaching `1e43/NaN` around state 301. A regression test was added for finite states and bounded cycle-boundary gaps.
3. The replacement uses multiple-shooting stitching of accepted FDDP state blocks. It remains finite and localises the initial maximum gap `5.64e-4` at the cycle boundary; FDDP then closes it to zero.
4. The first formal ladder accepted 4-step but rejected 6-step at `solver.stop=5.69e-7`, while every physical gate passed. Iteration history showed termination at the default regularisation limit `1e8`.
5. A single-variable diagnostic raised only `reg_max` to `1e12`; 6-step reached `solver.stop=6.11e-8` at `preg=1e9` without changing physical metrics or thresholds. A regression test fixed this solver configuration.
6. The final ladder accepted 4, 6, and 10 steps. Task 3 plus periodic/core regression passed `26/26`.

### Formal planning results

| Segment | Iterations | solver.stop | Gap | Support translation | Landing translation | Clearance | Joint margin | Forward | Duration |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4-step | 377 | 1.80e-8 | 0 | 4.39e-5 m | 3.29e-5 m | 19.998 mm | 11.987 mrad | 20.08 mm | 4.0 s |
| 6-step | 110 | 6.11e-8 | 0 | 4.96e-5 m | 3.99e-5 m | 19.997 mm | 11.958 mrad | 30.08 mm | 6.0 s |
| 10-step | 89 | 1.64e-8 | 0 | 6.01e-5 m | 4.58e-5 m | 19.991 mm | 11.879 mrad | 50.07 mm | 10.0 s |

All segment rotation, landing, clearance, joint, duration, and forward gates passed. Combined iteration log contains 579 rows: 378 for 4-step, 111 for 6-step, and 90 for 10-step.

Candidate contract:

- schema: `fddp-multistep-v1`;
- state/control shapes: `(1001,25)` / `(1000,6)`;
- acceptance state: `planning_accepted_contact_pending`;
- `tau_ref_hardware_usage=analysis_only`;
- all core numeric arrays finite;
- SHA-256: `c7e06b2bc454e478aa70fdad190e172271b94589cdd7e5cf9616bca0dc30208e`;
- final reference exists: **false**.

## Task 4 - Contact Feasibility RED Baseline

Status: **FAILED AS EXPECTED; FINAL NOMINAL WITHHELD**

The contact report was implemented with TDD for Pinocchio force ordering, CoP signs, convex-polygon margins, inclusive limits, DS normal-load fractions, planning/replay friction equality, ankle-to-sole wrench shifting, sole velocity, and atomic promotion. Unit tests passed `10/10`.

Command:

```bash
/home/zhijian/miniforge3/envs/fddp_biped/bin/python fddp/05c_report_fddp_multistep_contact_feasibility.py
```

The report reconstructs the exact 10-step Crocoddyl problem, evaluates all 1,300 active-contact rows, shifts each `LOCAL_WORLD_ALIGNED` wrench from the ankle frame to the MJCF sole plane at `z=-0.043 m`, and uses the verified `0.085 m x 0.145 m` support polygon. Planning and MuJoCo friction coefficients both equal `0.8`.

| Metric | Gate | RED result | Status |
|---|---:|---:|---:|
| Minimum normal force | >= -1e-5 N | -525,053.48 N | FAIL |
| Maximum friction utilisation | <= 0.90 | 2.545 | FAIL |
| Minimum CoP margin | >= 0.002 m | -1.0316 m | FAIL |
| Maximum sole tangential speed | <= 0.001 m/s | 0 | PASS |

There were 1,253 rejected active-contact rows: 286 unilateral violations, 291 friction violations, 1,253 CoP violations, and zero kinematic-slip violations. The DS boundaries show large equal-and-opposite normal forces, revealing unregulated internal-force indeterminacy in the dual 6D contact model. This is a physical-feasibility failure even though Task 3 state, gap, landing, and solver-stop gates passed.

Preserved evidence:

- `results/logs/fddp_multistep_contact_feasibility.csv`
- `results/logs/fddp_multistep_contact_feasibility.yaml`
- `results/figures/fddp_multistep_contact_feasibility.png`
- candidate SHA-256 remains `c7e06b2bc454e478aa70fdad190e172271b94589cdd7e5cf9616bca0dc30208e`
- `fddp_multistep_10step_ref.npz` exists: **false**

Next repair: introduce sole-centred contact frames plus explicit contact-force regularisation and `WrenchCone` barrier costs, then re-run FDDP continuation and this unchanged report gate.

## 2026-07-15 Contact-Aware Source Repair and Formal Promotion

Status: **PLANNING AND CONTACT FEASIBILITY PASSED; FINAL NOMINAL PROMOTED**

The RED baseline above is retained as negative evidence. It was not repaired by lowering any threshold. The source formulation was changed and the complete periodic/continuation/contact pipeline was rerun.

### Root cause and source repair

1. The old accepted periodic source used ankle contact frames and had no force-aware planar support cost. Copying that cycle into 4/6/10-step problems copied its contact infeasibility.
2. `fddp/05a_solve_fddp_periodic_cycle.py` now adds idempotent sole-centred operational frames at `[0,0,-0.043] m`, enables contact-force Jacobians, and adds one combined planar resultant support-cone residual at every running node.
3. The optimiser uses an inner CoP margin while the independent acceptance gate remains unchanged at `2.0 mm`. A `2.0 mm` internal probe left five violations with a minimum external margin of `1.985 mm`; a `2.1 mm` internal margin produced zero violations and a `2.079 mm` external margin.
4. The formal periodic cycle passed all seven homotopy stages with `solver.stop=9.90065e-8`, gap `0`, 200/200 support nodes, zero contact violations, minimum normal force `9.9127 N`, maximum friction utilisation `0.3203`, minimum CoP margin `2.0787 mm`, zero slip, `19.998 mm` swing clearance, `11.988 mrad` joint margin, and `10.088 mm` forward CoM displacement.

### Continuation seam and conditioning repair

1. Repeating the new periodic artifact exposed a `0.0323` multiple-shooting seam gap because the approximate periodic terminal state was followed by the next cycle's local state. Every internal seam is now explicitly replaced by the translated declared periodic start state.
2. The repaired source seam gap is `2.063e-4` and all following dynamics rollouts are near machine precision; regressions cover both the first seam and every extra seam in the ten-step seed.
3. `reg_max` was raised to `1e14` after the 10-step solve reached the old regularisation ceiling while all physical gates passed. No acceptance threshold was changed.
4. A contact-weight continuation was selected from runtime probes: `1e3` for 4-step, `1e4` for 6-step, and `1e5` for 10-step. The multi-step optimiser uses a `2.7 mm` internal CoP margin; the external independent gate remains `2.0 mm`.
5. `compute_multistep_diagnostics()` now reuses the accepted `05a` contact evaluator instead of duplicating wrench mathematics. `evaluate_segment()` requires support-node coverage, zero violations, unilateral force, friction, CoP, and slip gates for every 4/6/10-step segment.

### RED -> GREEN tests

The new tests first demonstrated that a friction utilisation of `0.9001` was incorrectly accepted and that 4-step diagnostics lacked contact fields. After the production change:

- the segment gate rejects `0.9001` and keeps `0.90` inclusive;
- 4-step diagnostics cover 400/400 running nodes and preserve cycle IDs `{0,1}`;
- one-iteration failure paths retain contact diagnostics and export no candidate;
- the complete continuation file passes `15/15`;
- the post-artifact two-step/periodic/continuation/contact/promotion regression passes `52/52`.

Command:

```bash
/home/zhijian/miniforge3/envs/fddp_biped/bin/python -m unittest \
  tests/test_fddp_two_step_alternating.py \
  tests/test_fddp_periodic_cycle.py \
  tests/test_fddp_multistep_continuation.py \
  tests/test_fddp_contact_feasibility.py \
  tests/test_fddp_multistep_contact_report.py -v
```

### Formal 4/6/10-step result

No runtime monkeypatch was used. The formal runner loaded production `05b` and called `solve_ladder(targets=(4,6,10), max_iterations=500)` with the official log and trajectory directories.

| Segment | Iterations | solver.stop | Gap | Support error | Landing error | Clearance | Joint margin | Forward | Contact nodes | Violations | Friction utilisation | CoP margin |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4-step | 156 | 8.628e-8 | 0 | 46.94 um | 27.89 um | 19.998 mm | 11.983 mrad | 20.088 mm | 400/400 | 0 | 0.3229 | 2.335 mm |
| 6-step | 96 | 4.062e-8 | 0 | 50.88 um | 30.87 um | 19.997 mm | 11.946 mrad | 30.088 mm | 600/600 | 0 | 0.3929 | 2.544 mm |
| 10-step | 66 | 4.410e-8 | 0 | 55.27 um | 35.73 um | 19.993 mm | 11.790 mrad | 50.087 mm | 1000/1000 | 0 | 0.5807 | 2.391 mm |

All segments have minimum normal force at least `9.9125 N` and maximum tangential speed `0 m/s`. All planning and contact gates passed without threshold relaxation.

### Independent `05c` promotion

Command:

```bash
/home/zhijian/miniforge3/envs/fddp_biped/bin/python \
  fddp/05c_report_fddp_multistep_contact_feasibility.py
```

The independent report reconstructed the exact ten-step problem and re-evaluated the promoted candidate:

| Metric | Gate | Result | Status |
|---|---:|---:|---:|
| Support nodes | 1000 | 1000 | PASS |
| Active contact rows | diagnostic | 1300 | RECORDED |
| Contact violations | 0 | 0 | PASS |
| Minimum normal force | >= -1e-5 N | 9.9126 N | PASS |
| Maximum friction utilisation | <= 0.90 | 0.5807 | PASS |
| Minimum CoP margin | >= 2.0 mm | 2.3913 mm | PASS |
| Maximum tangential speed | <= 1 mm/s | 0 | PASS |
| Planning/MJCF friction coefficient | identical | 0.8 / 0.8 | PASS |

The candidate SHA-256 is `86e34b91b9e62fe6fe0f9568df650900715c2992d5f353a32534a83421adb96a`. Atomic promotion created `transfer/trajectories/fddp_multistep_10step_ref.npz` with:

- schema `fddp-multistep-v1`;
- acceptance state `planning_and_contact_accepted`;
- 10 steps, 1000 running nodes, and 10.0 s duration;
- state/control shapes `(1001,25)` / `(1000,6)`;
- achieved forward CoM displacement `0.05008695 m`;
- `tau_ref_hardware_usage=analysis_only`;
- final SHA-256 `e412763e8a4ccada36d2affac0307bdc73aa3c7cd4e8081c6344db47b9e943ac`.

### Deterministic repeat

The complete production ladder was run a second time in `/tmp/fddp_multistep_repeat_20260715`, without overwriting official artifacts. For 4, 6, and 10 steps, the absolute difference between both runs was exactly zero for solver stop, gap, violation count, minimum normal force, maximum friction utilisation, minimum CoP margin, tangential speed, and forward displacement. The comparison used `rtol=1e-10`, `atol=1e-12` and returned `REPEAT_CONSISTENT True`.

### New and updated evidence

- `results/logs/fddp_multistep_planning_diagnostics.yaml`
- `results/logs/fddp_multistep_continuation_iterations.csv`
- `results/logs/fddp_multistep_contact_feasibility.csv`
- `results/logs/fddp_multistep_contact_feasibility.yaml`
- `results/figures/fddp_multistep_contact_feasibility.png`
- `transfer/trajectories/fddp_multistep_10step_planning_candidate.npz`
- `transfer/trajectories/fddp_multistep_10step_ref.npz`
- `transfer/trajectories/fddp_multistep_10step_ref.yaml`

### D-drive milestone archive

The existing non-destructive archive script was dry-run and then executed:

```powershell
& "\\wsl$\Ubuntu-22.04\home\zhijian\biped_fddp_project\tools\sync_thesis_results_to_d_drive.ps1"
```

It archived and verified 281 files totalling 110,774,617 bytes under `D:\毕业论文\机器人控制流程图\FDDP仿真结果归档` and regenerated a 281-row `SHA256_MANIFEST.csv`. Independent source/destination checks returned `match=True` for the final NPZ and this experiment record. The final NPZ hash on both sides is `E412763E8A4CCADA36D2AFFAC0307BDC73AA3C7CD4E8081C6344DB47B9E943AC`.

## Current Claim Boundary

The project now has one formally accepted, contact-feasible, 10-step/10.0-second alternating FDDP nominal trajectory with `DS -> SS -> DS` switching, first-order stop and shooting-gap gates, support/landing/clearance/joint gates, and independent normal-force/friction/CoP/slip evidence. This planning milestone is archived to D drive, but it establishes an offline FDDP planning result only. Ten-step MuJoCo position-command replay, natural-contact event ordering, integrated C0/C1/phase-only/C2-final comparison, 4 N recovery, statistical robustness, fault injection, 100000-update timing, coverage closure, final animations, and the final all-goal archive remain incomplete. Stage 8/9 remain scripted gait evidence and are not relabelled as FDDP gait. Hardware walking and hardware torque control remain out of scope.

