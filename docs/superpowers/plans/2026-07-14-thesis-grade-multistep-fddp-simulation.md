# Thesis-Grade Multi-Step FDDP Simulation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans and execute this plan inline in the shared workspace. Do not spawn sub-agents unless the user explicitly authorises delegation.

**Goal:** Deliver a reproducible, simulation-only, thesis-grade ten-step alternating FDDP position-control pipeline with contact-feasibility evidence, integrated C2-final control, bounded recovery, fault handling, statistical robustness, timing evidence, animations, and verified D-drive archival.

**Architecture:** Extend the accepted two-step FDDP result through a gated periodic-cycle and 4/6/10-step continuation ladder. Keep planning, contact analysis, MuJoCo replay, controller modes, safety, robustness aggregation, and artifact validation in independent modules connected by explicit versioned schemas. Every stage consumes only the last accepted upstream artifact and preserves failed diagnostics without overwriting accepted evidence.

**Tech Stack:** Python 3.10, Pinocchio, Crocoddyl/FDDP, MuJoCo, NumPy, Pandas, Matplotlib, PyYAML, `unittest`, `coverage.py`, PowerShell archive sync.

## Global Constraints

- Workspace: `/home/zhijian/biped_fddp_project`; interpreter: `/home/zhijian/miniforge3/envs/fddp_biped/bin/python`.
- There is no `.git`; use test-and-artifact checkpoints instead of commit steps.
- Do not modify `fddp/03n_solve_damped_6d_full_continuation.py`, `simulation/03o_replay_fddp_com_shift_mujoco.py`, `mpc/12_validate_scripted_alternating_step_locomotion.py`, or `mpc/13_run_scripted_robustness_matrix.py`.
- Preserve all existing left-step, right-step, two-step, Stage 8, and Stage 9 artifacts.
- World axes are forward `+Y`, vertical `+Z`, pitch `+X`.
- Stage 8 scripted motion is never a FDDP warm start. A larger solve may use only an accepted shorter FDDP solve.
- Hardware is out of scope. The command boundary remains `q_cmd = q_ref + delta_q`; `tau_ref` is analysis-only and must never be passed to MuJoCo actuators or described as a hardware torque command.
- Production changes follow RED -> GREEN. Dynamic loaders must insert the module into `sys.modules` before `exec_module()`.
- Failed experiments retain diagnostics and iteration histories. They never export a nominal NPZ, overwrite the last accepted reference, or set a success manifest.
- Stop for user direction if the `+/-4 N` gate requires mechanical/actuator/hardware changes or an approved claim/threshold change.
- Every accepted and failed final artifact is mirrored to `D:\毕业论文\机器人控制流程图\FDDP仿真结果归档`; the sync regenerates and verifies `SHA256_MANIFEST.csv`.

---

## Task 1: Freeze schemas, thresholds, schedule, and pure gates

**Files:**

- Create: `config/fddp_multistep_experiment.yaml`
- Create: `fddp/multistep_core.py`
- Create: `tests/test_fddp_multistep_core.py`

**Step 1: Write failing pure tests**

The tests must dynamically load `fddp/multistep_core.py` with registration and assert:

```python
spec = importlib.util.spec_from_file_location("fddp_multistep_core", MODULE_PATH)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = module
spec.loader.exec_module(module)

schedule = module.build_step_schedule(step_count=10)
self.assertEqual(len(schedule), 25)
self.assertEqual(sum(p.node_count for p in schedule), 1000)
self.assertEqual([p.mode for p in schedule[:5]], ["DS", "RIGHT_SS", "DS", "LEFT_SS", "DS"])
self.assertEqual([p.mode for p in schedule[5:10]], ["DS", "RIGHT_SS", "DS", "LEFT_SS", "DS"])
self.assertAlmostEqual(module.planned_forward_displacement(schedule), 0.050, places=12)

metadata = module.export_metadata(step_count=10, horizon_nodes=1000)
self.assertEqual(metadata["schema_version"], "fddp-multistep-v1")
self.assertEqual(metadata["tau_ref_hardware_usage"], "analysis_only")

bad = dict(module.DEFAULT_THRESHOLDS)
bad["max_shooting_gap_norm"] = -1.0
with self.assertRaises(ValueError):
    module.validate_thresholds(bad)
```

Also test exact phase alternation, the full `[DS20, right-SS70, DS20, left-SS70, DS20]` contract for each of five periodic cycles, at least 10 s duration, left/right contact flags, cycle IDs, and inclusive boundary semantics. Do not merge the terminal DS of one cycle with the initial DS of the next: the resulting consecutive DS blocks are the deliberate cycle-boundary settling interval that makes ten steps exactly 1000 nodes/10 s.

**Step 2: Run RED**

```bash
/home/zhijian/miniforge3/envs/fddp_biped/bin/python -m unittest tests/test_fddp_multistep_core.py -v
```

Expected: module/file missing, with the test itself importing cleanly.

**Step 3: Implement the minimum pure core**

Expose this stable interface:

```python
SCHEMA_VERSION = "fddp-multistep-v1"
DT = 0.010
DS_NODES = 20
SS_NODES = 70
STEP_FORWARD_Y_M = 0.010
SWING_CLEARANCE_Z_M = 0.020

@dataclass(frozen=True)
class PhaseSpec:
    name: str
    mode: str
    node_count: int
    swing_side: str | None
    cycle_id: int

def build_step_schedule(step_count: int) -> list[PhaseSpec]: ...
def phase_arrays(schedule: list[PhaseSpec]) -> dict[str, np.ndarray]: ...
def planned_forward_displacement(schedule: list[PhaseSpec]) -> float: ...
def export_metadata(step_count: int, horizon_nodes: int) -> dict: ...
def validate_thresholds(thresholds: Mapping[str, float]) -> None: ...
def evaluate_gate(metrics: Mapping[str, float], thresholds: Mapping[str, float]) -> dict: ...
```

Load all experiment values from `config/fddp_multistep_experiment.yaml`; reject unknown schema versions, non-finite values, non-even step counts, fewer than ten final steps, or schedules shorter than 10 s.

**Step 4: Run GREEN and regression**

```bash
/home/zhijian/miniforge3/envs/fddp_biped/bin/python -m unittest tests/test_fddp_multistep_core.py tests/test_fddp_two_step_alternating.py -v
```

Checkpoint: record command, interpreter, RED cause, and GREEN result in the experiment record draft.

---

## Task 2: Build and gate a periodic two-step FDDP cycle

**Files:**

- Create: `fddp/05a_solve_fddp_periodic_cycle.py`
- Create: `tests/test_fddp_periodic_cycle.py`
- Output: `results/logs/fddp_periodic_cycle_diagnostics.yaml`
- Output: `results/logs/fddp_periodic_cycle_iterations.csv`
- Output on pass only: `transfer/trajectories/fddp_periodic_cycle_ref.npz`

**Step 1: Write failing boundary and export tests**

Use synthetic states to test forward-translation removal and exact pass/fail boundaries:

```python
metrics = module.compute_cycle_boundary_metrics(
    x_start=x_start,
    x_terminal=x_terminal,
    u_start=u_start,
    u_terminal=u_terminal,
    left_start=T_l0,
    left_terminal=T_l1,
    right_start=T_r0,
    right_terminal=T_r1,
    planned_forward_y=0.020,
)
self.assertLessEqual(metrics["joint_position_max_rad"], 1e-3)
self.assertTrue(module.evaluate_cycle_acceptance(metrics)["passed"])
```

Test failures for joint position, joint velocity, base height, pitch, CoM velocity, relative foot translation/rotation, control discontinuity, contact incompatibility, `solver.stop`, and shooting gap. Test that `write_cycle_artifacts(..., accepted=False)` writes diagnostics but not NPZ.

**Step 2: Run RED**

```bash
/home/zhijian/miniforge3/envs/fddp_biped/bin/python -m unittest tests/test_fddp_periodic_cycle.py -v
```

**Step 3: Implement periodic-cycle problem and diagnostics**

Reuse only stable two-step helpers from `fddp/04c_solve_fddp_two_step_alternating.py`; start from the accepted static `x0`. Expose:

```python
def build_periodic_cycle_problem(model, x0, config): ...
def compute_cycle_boundary_metrics(...)->dict: ...
def evaluate_cycle_acceptance(metrics: Mapping[str, float]) -> dict: ...
def solve_periodic_cycle(max_iterations: int = 500) -> dict: ...
def write_cycle_artifacts(result: Mapping, accepted: bool) -> dict[str, Path]: ...
```

Acceptance after removing `+Y` translation:

- joint position `<=1e-3 rad`, joint velocity `<=1e-2 rad/s`;
- base height and pitch `<=1e-3`;
- CoM velocity `<=1e-3 m/s`;
- relative foot translation `<=1e-3 m`, rotation `<=0.2 deg`;
- control discontinuity `<=0.10` analysis units;
- compatible DS boundary contacts;
- `solver.stop<=1e-7`, gap `<=1e-7`, and all existing pose/clearance/joint gates.

**Step 4: Run tests, then solve**

```bash
/home/zhijian/miniforge3/envs/fddp_biped/bin/python -m unittest tests/test_fddp_periodic_cycle.py -v
/home/zhijian/miniforge3/envs/fddp_biped/bin/python fddp/05a_solve_fddp_periodic_cycle.py
```

Expected: solver failure remains a recorded failed experiment and blocks Task 3 nominal execution; tests must still pass.

---

## Task 3: Continue accepted FDDP solutions through 4, 6, and 10 steps

**Files:**

- Create: `fddp/05b_solve_fddp_multistep_continuation.py`
- Create: `tests/test_fddp_multistep_continuation.py`
- Output: `results/logs/fddp_multistep_continuation_iterations.csv`
- Output: `results/logs/fddp_multistep_planning_diagnostics.yaml`
- Output: `results/figures/fddp_multistep_solver_convergence.png`
- Output on planning pass only: `transfer/trajectories/fddp_multistep_10step_planning_candidate.npz`
- Output on planning pass only: `transfer/trajectories/fddp_multistep_10step_planning_candidate_summary.yaml`

**Step 1: Write failing seed-provenance and gated-export tests**

```python
ladder = module.continuation_ladder(final_step_count=10)
self.assertEqual(ladder, (2, 4, 6, 10))
self.assertEqual(module.required_seed_for(6), "accepted_fddp_4step")
with self.assertRaises(module.SeedIntegrityError):
    module.validate_seed_metadata({"source": "stage8_scripted"}, target_steps=4)

accepted = module.evaluate_segment({**passing_metrics, "solver_stop": 1e-7})
self.assertTrue(accepted["passed"])
self.assertFalse(module.evaluate_segment({**passing_metrics, "solver_stop": 1.01e-7})["passed"])
```

Test NPZ keys and shapes for `t_ref`, `q_ref`, `dq_ref`, `state_ref`, `tau_ref`, phase names, contacts, feet, CoM, cycle IDs, seed provenance, solver metadata, schema version, and `analysis_only`. The planning candidate must say `acceptance_state=planning_accepted_contact_pending`; test atomic temp-file replacement only after planning acceptance. The final `fddp_multistep_10step_ref.npz` must not be created in Task 3.

**Step 2: Run RED, implement orchestration, run GREEN**

```bash
/home/zhijian/miniforge3/envs/fddp_biped/bin/python -m unittest tests/test_fddp_multistep_continuation.py -v
```

Implement:

```python
def continuation_ladder(final_step_count: int = 10) -> tuple[int, ...]: ...
def validate_seed_metadata(metadata: Mapping, target_steps: int) -> None: ...
def extend_accepted_seed(seed, target_steps: int, model, config): ...
def evaluate_segment(metrics: Mapping[str, float]) -> dict: ...
def solve_ladder(targets=(4, 6, 10)) -> dict: ...
def gated_export(result: Mapping, destination: Path) -> bool: ...
```

Each segment must satisfy all planning gates. The 4/6/10-step problems contain 400/600/1000 nodes respectively by preserving complete 200-node two-step cycles. The ten-step reference must be `>=10 s` and `>=0.040 m` forward. Preserve per-segment iteration histories whether pass or fail.

**Step 3: Execute gated ladder**

```bash
/home/zhijian/miniforge3/envs/fddp_biped/bin/python fddp/05b_solve_fddp_multistep_continuation.py --targets 4 6 10
```

Do not proceed to nominal replay from the planning candidate. Task 4 must pass contact feasibility and atomically promote the candidate to the final `fddp_multistep_10step_ref.npz` first.

---

## Task 4: Extract FDDP contact wrench, friction, CoP, and slip evidence

**Files:**

- Create: `fddp/contact_feasibility.py`
- Create: `fddp/05c_report_fddp_multistep_contact_feasibility.py`
- Create: `tests/test_fddp_contact_feasibility.py`
- Output: `results/logs/fddp_multistep_contact_feasibility.csv`
- Output: `results/logs/fddp_multistep_contact_feasibility.yaml`
- Output: `results/figures/fddp_multistep_contact_feasibility.png`

**Step 1: Write failing analytical tests**

Use known synthetic wrenches and foot polygons:

```python
wrench = np.array([10.0, 0.0, 100.0, 0.5, 1.0, 0.0])
metrics = module.wrench_metrics(wrench, mu=0.8, polygon=FOOT_POLYGON)
self.assertAlmostEqual(metrics["friction_utilisation"], 0.125)
self.assertTrue(np.allclose(metrics["cop_xy_m"], [-0.01, 0.005]))
self.assertFalse(metrics["unilateral_violation"])
self.assertGreaterEqual(metrics["cop_margin_m"], 0.002)
```

Test sign conventions, zero/negative normal force, polygon-edge margin, utilisation `0.90` inclusive, tangential speed `1e-3 m/s` inclusive, DS load distribution summing to one, and mismatch between planning/replay friction assumptions.

**Step 2: Run RED, implement pure math, run GREEN**

Expose:

```python
def wrench_metrics(wrench, mu: float, polygon: np.ndarray) -> dict: ...
def point_polygon_signed_margin(point_xy, polygon_xy) -> float: ...
def classify_slip(tangential_speed_m_s: float) -> bool: ...
def evaluate_contact_feasibility(rows: Sequence[Mapping], config: Mapping) -> dict: ...
def assert_friction_contract(planning_mu: float, replay_mu: float) -> None: ...
```

**Step 3: Add Crocoddyl extraction and export gate**

Extract active-contact wrench and foot tangential velocity at every accepted FDDP node. Acceptance: normal force `>=-1e-5 N`, utilisation `<=0.90`, CoP margin `>=0.002 m`, tangential speed `<=1e-3 m/s`, identical `mu` assumption. Connect this report to final nominal export validation so contact failure invalidates the success manifest without deleting planning diagnostics.

On contact pass, validate the planning-candidate hash and schema, then atomically copy/promote it to `transfer/trajectories/fddp_multistep_10step_ref.npz` with a final summary that includes both planning and contact acceptance. On contact failure, retain the candidate and diagnostics but do not create or overwrite the final reference.

**Step 4: Run report**

```bash
/home/zhijian/miniforge3/envs/fddp_biped/bin/python -m unittest tests/test_fddp_contact_feasibility.py -v
/home/zhijian/miniforge3/envs/fddp_biped/bin/python fddp/05c_report_fddp_multistep_contact_feasibility.py
```

---

## Task 5: Replay the accepted ten-step reference in MuJoCo

**Files:**

- Create: `simulation/03u_replay_fddp_multistep_mujoco.py`
- Create: `tests/test_fddp_multistep_replay.py`
- Output: `results/logs/fddp_multistep_nominal_replay.csv`
- Output: `results/logs/fddp_multistep_nominal_replay.yaml`
- Output: `results/figures/fddp_multistep_nominal_replay.png`

**Step 1: Write failing loader and event-order tests**

Test strict schema/load integrity, rejection of the old two-step or CoM-shift NPZ, non-use of `tau_ref`, scheduled/observed event matching, and exact replay gates. Example:

```python
ref = module.load_multistep_reference(valid_npz)
self.assertEqual(ref.schema_version, "fddp-multistep-v1")
self.assertGreaterEqual(ref.duration_s, 10.0)
with self.assertRaises(module.ReferenceIntegrityError):
    module.load_multistep_reference(old_com_shift_npz)

events = module.match_contact_events(scheduled, observed, tolerance_s=0.08)
self.assertGreaterEqual(events["match_ratio"], 0.90)
```

**Step 2: Implement strict loader and smoke replay**

Reuse the position-actuator proxy semantics from `03r`, not historical output paths. Log q reference/command/state, pitch, height, toe/heel states, scheduled and observed events, phase offset, penetration, flight duration, and contact order.

**Step 3: Run RED/GREEN smoke and full nominal replay**

```bash
/home/zhijian/miniforge3/envs/fddp_biped/bin/python -m unittest tests/test_fddp_multistep_replay.py -v
/home/zhijian/miniforge3/envs/fddp_biped/bin/python simulation/03u_replay_fddp_multistep_mujoco.py --smoke-steps 2
/home/zhijian/miniforge3/envs/fddp_biped/bin/python simulation/03u_replay_fddp_multistep_mujoco.py
```

Acceptance: no fall, pitch `<=15 deg`, joint RMSE `<=4 deg`, torso height `>=65%` initial, no sustained penetration/unintended flight/invalid event order, event match `>=0.90`, and all liftoff/touchdown kinds in order.

---

## Task 6: Integrate one C0/C1/phase-only/C2-final controller interface

**Files:**

- Create: `mpc/multistep_controller.py`
- Create: `mpc/18_validate_fddp_multistep_c2_final.py`
- Create: `tests/test_fddp_multistep_controller.py`
- Output: `results/logs/fddp_multistep_controller_ablation.csv`
- Output: `results/logs/fddp_multistep_controller_ablation.yaml`
- Output: `results/figures/fddp_multistep_controller_ablation.png`

**Step 1: Write failing shared-interface tests**

```python
for mode in ("C0", "C1", "phase_only", "C2_final"):
    out = controller.update(sample, mode=mode)
    self.assertEqual(out.schema_version, "fddp-controller-v1")
    self.assertTrue(np.allclose(out.q_cmd, out.q_ref + out.delta_q))
    self.assertTrue(np.all(np.isfinite(out.q_cmd)))
    self.assertLessEqual(abs(out.alpha_rad), np.deg2rad(5.0))
```

Test that C0 has zero correction/phase retiming, C1 has correction only, phase-only has retiming only, C2-final has both, and final mode rejects any non-multistep reference. Test joint/rate limiter flags and required logging fields.

**Step 2: Implement common data types and controller**

```python
@dataclass(frozen=True)
class ControllerInput: ...

@dataclass(frozen=True)
class ControllerOutput:
    q_cmd: np.ndarray
    q_ref: np.ndarray
    delta_q: np.ndarray
    wall_time_s: float
    reference_time_s: float
    pitch_error_rad: float
    alpha_rad: float
    phase_offset_s: float
    matched_events: int
    skipped_events: int
    limiter_active: bool
    saturated: bool
    deadline_missed: bool
    fallback_active: bool
    controller_state: str
    schema_version: str

class MultiStepController:
    def update(self, sample: ControllerInput, mode: str) -> ControllerOutput: ...
```

Refactor by importing stable FIR-QP and phase synchroniser helpers from existing evidence modules; do not alter their historical scripts/results.

**Step 3: Implement common disturbance ablation**

Run all four modes on identical seeds/disturbances. Acceptance: C1 IAE improves C0 by `>=10%`, C1 peak no worse than `1.05*C0`, C2-final IAE `<=1.02*C1`, C2-final peak `<=1.05*C1`, and C2-final contact timing error improves fixed-phase C1 by `>=30%`.

**Step 4: Run targeted tests and experiment**

```bash
/home/zhijian/miniforge3/envs/fddp_biped/bin/python -m unittest tests/test_fddp_multistep_controller.py -v
/home/zhijian/miniforge3/envs/fddp_biped/bin/python mpc/18_validate_fddp_multistep_c2_final.py
```

---

## Task 7: Add bounded `+/-4 N` recovery authority

**Files:**

- Create: `mpc/19_tune_fddp_multistep_push_recovery.py`
- Create: `tests/test_fddp_multistep_recovery.py`
- Output: `results/logs/fddp_multistep_recovery_sweep.csv`
- Output: `results/logs/fddp_multistep_recovery_selection.yaml`
- Output: `results/figures/fddp_multistep_recovery_sweep.png`

**Step 1: Write failing recovery-bound tests**

Test the incremental authority order, `|alpha|<=5 deg`, foothold offset `<=0.015 m`, position/rate/contact/landing gates, deterministic configuration selection, and metadata propagation into contact reports.

```python
bounded = module.bound_recovery(raw_alpha, raw_foothold_y, limits)
self.assertLessEqual(abs(bounded.alpha_rad), np.deg2rad(5.0))
self.assertLessEqual(abs(bounded.foothold_y_m), 0.015)
self.assertTrue(module.recovery_terms_respect_limits(bounded, state, limits))
```

**Step 2: Implement staged search**

Search in this order: multi-step FIR-QP retune, phase-aware gain scheduling, then bounded CoM/foothold adjustment. Do not enable a later stage unless the earlier accepted candidates fail the `+/-4 N` gate. All candidates and failures remain in the CSV.

**Step 3: Run smoke seeds, then 30-seed recovery check**

```bash
/home/zhijian/miniforge3/envs/fddp_biped/bin/python -m unittest tests/test_fddp_multistep_recovery.py -v
/home/zhijian/miniforge3/envs/fddp_biped/bin/python mpc/19_tune_fddp_multistep_push_recovery.py --seeds 0 1 2
/home/zhijian/miniforge3/envs/fddp_biped/bin/python mpc/19_tune_fddp_multistep_push_recovery.py --seed-count 30
```

Acceptance is `>=27/30` in both forward and backward 4 N cases. If safe software authority is exhausted, record evidence and stop for user direction; do not lower the threshold.

---

## Task 8: Implement deterministic safety state machine and fault injection

**Files:**

- Create: `mpc/runtime_safety.py`
- Create: `mpc/20_run_multistep_fault_injection.py`
- Create: `tests/test_multistep_runtime_safety.py`
- Create: `tests/test_multistep_fault_injection.py`
- Output: `results/logs/fddp_multistep_fault_injection.csv`
- Output: `results/logs/fddp_multistep_fault_injection.yaml`

**Step 1: Write failing transition-table tests**

```python
self.assertEqual(module.RuntimeState.__members__.keys(),
                 {"NORMAL", "DEGRADED", "FALLBACK", "SAFE_HOLD", "FAILED"})
next_state = machine.handle(module.Fault.NAN_MEASUREMENT, now_s=1.2)
self.assertIn(next_state, {module.RuntimeState.FALLBACK, module.RuntimeState.SAFE_HOLD})
self.assertTrue(np.all(np.isfinite(machine.last_command)))
```

Test stale/NaN/Inf pitch, stuck sensors, missing/out-of-order events, invalid QP, deadline, saturation, actuator delay, and missing/truncated/schema-invalid reference. Every transition must log time, reason, previous/next state, and recovery outcome. Corrupt files must prevent startup.

**Step 2: Implement state machine and controller wrapper**

```python
class RuntimeState(Enum):
    NORMAL = auto()
    DEGRADED = auto()
    FALLBACK = auto()
    SAFE_HOLD = auto()
    FAILED = auto()

class RuntimeSafetyMachine:
    def validate_input(self, sample) -> list[Fault]: ...
    def validate_output(self, output) -> list[Fault]: ...
    def handle(self, fault: Fault, now_s: float) -> RuntimeState: ...
    def safe_command(self, q_last_valid, q_hold, dt_s) -> np.ndarray: ...
```

Invalid online correction falls back deterministically to C0 or safe hold. No path may emit non-finite/out-of-limit commands or uncaught controller exceptions.

**Step 3: Run targeted tests and fault campaign**

```bash
/home/zhijian/miniforge3/envs/fddp_biped/bin/python -m unittest tests/test_multistep_runtime_safety.py tests/test_multistep_fault_injection.py -v
/home/zhijian/miniforge3/envs/fddp_biped/bin/python mpc/20_run_multistep_fault_injection.py
```

---

## Task 9: Run statistical robustness matrix and Wilson intervals

**Files:**

- Create: `mpc/robustness_statistics.py`
- Create: `mpc/21_run_fddp_multistep_robustness_matrix.py`
- Create: `tests/test_fddp_multistep_robustness.py`
- Output: `results/logs/fddp_multistep_robustness_trials.csv`
- Output: `results/logs/fddp_multistep_robustness_aggregate.csv`
- Output: `results/logs/fddp_multistep_robustness_summary.yaml`
- Output: `results/figures/fddp_multistep_robustness_matrix.png`

**Step 1: Write failing scenario/statistics tests**

```python
lo, hi = module.wilson_interval(successes=27, trials=30, confidence=0.95)
self.assertLess(lo, 0.90)
self.assertGreater(hi, 0.90)
self.assertEqual(len(module.build_scenarios()), EXPECTED_SCENARIO_COUNT)
self.assertTrue(all(s.seed_count >= 30 for s in module.build_scenarios()))
```

Test the full declared scenario set, critical-case classification, deterministic seed generation, failure-category requirement, aggregation without dropping failures, and exact controller mode applicability.

**Step 2: Implement immutable scenario registry and aggregation**

Include nominal; `+/-2 N`, `+/-4 N`; `mu=0.40/0.50/0.80`; slopes `-5/-3/+3/+5 deg`; steps `2/4/6 mm`; command delays `4/12/20 ms`; sensor delays `0/20/40 ms`; gain `+/-20%`; actuator time constants `20..50 ms`; joint noise `0.002 rad`; pitch noise `0.5 deg`; and combined A/B/C exactly as approved.

**Step 3: Run smoke matrix, then full matrix**

```bash
/home/zhijian/miniforge3/envs/fddp_biped/bin/python -m unittest tests/test_fddp_multistep_robustness.py -v
/home/zhijian/miniforge3/envs/fddp_biped/bin/python mpc/21_run_fddp_multistep_robustness_matrix.py --seed-count 3 --smoke
/home/zhijian/miniforge3/envs/fddp_biped/bin/python mpc/21_run_fddp_multistep_robustness_matrix.py --seed-count 30
```

Acceptance: nominal C2-final `30/30`; each critical scenario `>=90%`; both 4 N directions `>=90%`; aggregate C2-final `>=95%`; Wilson 95% CI reported for every scenario and aggregate; zero suppressed failed seeds.

---

## Task 10: Benchmark 100,000 updates per final controller mode

**Files:**

- Create: `mpc/22_benchmark_fddp_multistep_controller.py`
- Create: `tests/test_fddp_multistep_timing.py`
- Output: `results/logs/fddp_multistep_controller_timing.csv`
- Output: `results/logs/fddp_multistep_controller_timing.yaml`
- Output: `results/figures/fddp_multistep_controller_timing.png`

**Step 1: Write failing timing-statistics tests**

Test warm-up exclusion, exact sample count, percentile calculation, separation of controller/physics/render times, and unexplained deadline classification.

**Step 2: Implement benchmark harness**

```python
def benchmark_controller(controller, mode: str, samples, warmup: int, updates: int) -> dict:
    """Measure controller.update only; exclude MuJoCo and rendering."""
    ...
```

**Step 3: Run unit and full benchmark**

```bash
/home/zhijian/miniforge3/envs/fddp_biped/bin/python -m unittest tests/test_fddp_multistep_timing.py -v
/home/zhijian/miniforge3/envs/fddp_biped/bin/python mpc/22_benchmark_fddp_multistep_controller.py --updates 100000 --modes C1 C2_final
```

Acceptance at 20 ms controller period: p99 `<5 ms`, max `<20 ms`, zero unexplained deadline misses for each final mode.

---

## Task 11: Reproducibility, schemas, coverage, animations, and experiment record

**Files:**

- Create: `tools/run_fddp_multistep_validation_pipeline.py`
- Create: `simulation/03v_render_fddp_multistep_animation.py`
- Create: `simulation/03w_render_fddp_multistep_recovery_animation.py`
- Create: `simulation/03x_render_fddp_multistep_failure_animation.py`
- Create: `tests/test_fddp_multistep_artifacts.py`
- Create: `tests/test_fddp_multistep_animations.py`
- Create: `tests/test_fddp_multistep_reproducibility.py`
- Create: `docs/2026-07-14_fddp_multistep_simulation_closure_experiment.md`
- Output: `results/animations/fddp_multistep_nominal_walking.gif`
- Output: `results/animations/fddp_multistep_4n_recovery.gif`
- Output: `results/animations/fddp_multistep_representative_failure.gif`
- Output: `results/logs/fddp_multistep_pipeline_manifest.yaml`

**Step 1: Write failing artifact-contract tests**

Validate required paths, schema versions, NPZ/CSV/YAML cross-consistency, stored seeds/config hashes, `analysis_only`, no contradictory success flags, and animation frame/duration metadata. Add a test that two synthetic repeated manifests compare within declared tolerances but reject a changed pass/fail decision.

**Step 2: Implement ordered pipeline and validators**

The pipeline may resume from validated accepted artifacts but must never infer success from file existence alone. It runs planning -> contact -> replay -> controller -> recovery -> faults -> robustness -> timing -> animations -> report and stores command, interpreter, config hash, source file hashes, seeds, thresholds, metrics, and claim boundary.

**Step 3: Render three animations**

Create nominal ten-step walking, successful 4 N recovery, and a representative failed-seed animation. Captions must say MuJoCo position-command simulation and identify controller mode/scenario; no hardware claim.

**Step 4: Run two clean validation passes and compare**

```bash
/home/zhijian/miniforge3/envs/fddp_biped/bin/python -m unittest tests/test_fddp_multistep_artifacts.py tests/test_fddp_multistep_animations.py tests/test_fddp_multistep_reproducibility.py -v
/home/zhijian/miniforge3/envs/fddp_biped/bin/python tools/run_fddp_multistep_validation_pipeline.py --run-id clean-a
/home/zhijian/miniforge3/envs/fddp_biped/bin/python tools/run_fddp_multistep_validation_pipeline.py --run-id clean-b
/home/zhijian/miniforge3/envs/fddp_biped/bin/python tools/run_fddp_multistep_validation_pipeline.py --compare clean-a clean-b
```

The record must include every command, interpreter, configuration, threshold, acceptance table, failed result, figure/table/animation path, reproducibility result, and simulation-only claim boundary.

---

## Task 12: Fresh full verification, coverage, defect audit, and D-drive archive

**Files:**

- Modify only if needed: `tools/sync_thesis_results_to_d_drive.ps1`
- Output: `D:\毕业论文\机器人控制流程图\FDDP仿真结果归档\SHA256_MANIFEST.csv`

**Step 1: Run targeted suite and full suite from a fresh process**

```bash
/home/zhijian/miniforge3/envs/fddp_biped/bin/python -m unittest \
  tests/test_fddp_multistep_core.py \
  tests/test_fddp_periodic_cycle.py \
  tests/test_fddp_multistep_continuation.py \
  tests/test_fddp_contact_feasibility.py \
  tests/test_fddp_multistep_replay.py \
  tests/test_fddp_multistep_controller.py \
  tests/test_fddp_multistep_recovery.py \
  tests/test_multistep_runtime_safety.py \
  tests/test_multistep_fault_injection.py \
  tests/test_fddp_multistep_robustness.py \
  tests/test_fddp_multistep_timing.py \
  tests/test_fddp_multistep_artifacts.py \
  tests/test_fddp_multistep_animations.py \
  tests/test_fddp_multistep_reproducibility.py -v
/home/zhijian/miniforge3/envs/fddp_biped/bin/python -m unittest discover -s tests -p 'test_*.py' -v
```

**Step 2: Measure meaningful branch coverage of new core modules**

```bash
/home/zhijian/miniforge3/envs/fddp_biped/bin/python -m coverage erase
/home/zhijian/miniforge3/envs/fddp_biped/bin/python -m coverage run --branch -m unittest discover -s tests -p 'test_*.py'
/home/zhijian/miniforge3/envs/fddp_biped/bin/python -m coverage report --include='fddp/multistep_core.py,fddp/contact_feasibility.py,mpc/multistep_controller.py,mpc/runtime_safety.py,mpc/robustness_statistics.py' --fail-under=85
```

If `coverage.py` is unavailable, record the missing dependency and ask before installation; do not silently skip the gate.

**Step 3: Audit P0/P1 defects and claims**

Check that no crash, corruption, unsafe command, invalid export, bad acceptance logic, schedule error, unreproducible decision, unexplained model mismatch, or claim/result contradiction remains. Any unresolved P0/P1 blocks completion.

**Step 4: Sync and verify D drive**

```powershell
powershell.exe -ExecutionPolicy Bypass -File tools\sync_thesis_results_to_d_drive.ps1
```

Verify every source/destination pair by SHA-256, verify the regenerated manifest, and record the manifest count/hash in the experiment document. A copy failure or hash mismatch blocks completion.

**Step 5: Final supported claim**

Only after every gate above passes, state:

> Within the declared simulation model, controller limits, scenario matrix, and statistical sample size, the project provides a reproducible thesis-grade multi-step FDDP position-control pipeline with integrated pitch correction, natural-contact phase synchronisation, contact-feasibility evidence, bounded recovery, fault handling, and archived validation artifacts.

Never claim real-hardware autonomous walking, hardware torque control, universal stability, or robustness outside the tested envelope.

