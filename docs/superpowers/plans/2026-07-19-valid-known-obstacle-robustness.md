# Valid Known-Obstacle Robustness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Replace the initially intersecting 2/4/6 mm toe-obstacle geometry with a valid future obstacle, prove it is absent at initialisation and encountered during replay, and complete the existing statistical robustness campaign without relaxing gates.

**Architecture:** Reuse the existing terrain XML injector, multi-step C2 MuJoCo runner, recovery gate, and Task9 statistical runner. Add one geometry contract, one obstacle-specific contact diagnostic, and one scenario-specific encounter gate; keep the accepted flat-ground FDDP reference and controller unchanged.

**Tech Stack:** Python 3.10, MuJoCo, NumPy, pandas, unittest, existing FDDP/C2 project modules.

## Global Constraints

- Hardware-facing control remains `q_cmd = q_ref + delta_q`; `tau_ref` is analysis-only.
- The result claim is `simulation_only_position_command`, not hardware walking.
- Do not relax pitch, height, event, flight, saturation, command, Wilson, critical-scenario, or aggregate acceptance gates.
- Preserve all failed seeds and assign every failed trial a machine-readable failure category.
- Reuse the existing output paths and runner; do not create a second planner or robustness framework.
- The repository currently has no baseline commit and all source files are untracked. Do not create a partial initial commit; use test and diff checkpoints until the user establishes a baseline.

---

### Task 1: Define and test the obstacle geometry contract

**Files:**
- Modify: `mpc/13_run_scripted_robustness_matrix.py:249-277`
- Modify: `tests/test_fddp_multistep_robustness.py`

**Interfaces:**
- Produces: `step_obstacle_geometry(initial_clearance_m: float = 0.005) -> dict[str, float]`
- Consumed by: `build_step_xml()`, Task 3 summary metadata, and geometry tests.

## Shared test helpers

```python
RUNNER_PATH = ROOT / "mpc" / "18_validate_fddp_multistep_c2_final.py"
BATCH_PATH = ROOT / "mpc" / "21_run_fddp_multistep_robustness_matrix.py"


def load_path_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_runner(name):
    return load_path_module(name, RUNNER_PATH)


def load_batch(name):
    return load_path_module(name, BATCH_PATH)


def initialise_reference_pose(runner, step_height_m):
    model = runner._build_robustness_model(step_height_m=step_height_m)
    data = mujoco.MjData(model)
    mapping = runner.proxy.resolve_joint_and_actuator_mapping(model)
    controller = runner.controller_runtime.MultiStepController()
    mujoco.mj_resetData(model, data)
    for address, value in zip(
        mapping["qpos_addresses"], controller.reference.q_ref[0]
    ):
        data.qpos[address] = value
    mujoco.mj_forward(model, data)
    return model, data
```
- [x] **Step 1: Write the failing geometry tests**

```python
def test_step_obstacle_geometry_clears_initial_toe_by_five_mm(self):
    runner = load_runner("test_valid_step_geometry")
    geometry = runner.terrain_runtime.step_obstacle_geometry()
    self.assertAlmostEqual(geometry["initial_toe_y_m"], 0.0725)
    self.assertAlmostEqual(geometry["obstacle_rear_edge_y_m"], 0.0775)
    self.assertAlmostEqual(geometry["initial_clearance_m"], 0.005)
    self.assertAlmostEqual(geometry["obstacle_center_y_m"], 0.0895)

def test_step_obstacle_geometry_rejects_insufficient_clearance(self):
    runner = load_runner("test_invalid_step_geometry")
    with self.assertRaisesRegex(ValueError, "clearance"):
        runner.terrain_runtime.step_obstacle_geometry(0.0049)
```

- [x] **Step 2: Run the two tests and verify RED**

Run:

```bash
/home/zhijian/miniforge3/envs/fddp_biped/bin/python -m unittest \
  tests.test_fddp_multistep_robustness.MultiStepRobustnessStatisticsTests.test_step_obstacle_geometry_clears_initial_toe_by_five_mm \
  tests.test_fddp_multistep_robustness.MultiStepRobustnessStatisticsTests.test_step_obstacle_geometry_rejects_insufficient_clearance
```

Expected: both tests error because `step_obstacle_geometry` does not exist.

- [x] **Step 3: Implement the minimum geometry helper and reuse it in XML**

```python
INITIAL_FOOT_CENTER_Y_M = 0.0
FOOT_HALF_LENGTH_Y_M = 0.0725
STEP_OBSTACLE_HALF_LENGTH_Y_M = 0.012
MINIMUM_STEP_INITIAL_CLEARANCE_M = 0.005

def step_obstacle_geometry(
    initial_clearance_m: float = MINIMUM_STEP_INITIAL_CLEARANCE_M,
) -> dict[str, float]:
    clearance = float(initial_clearance_m)
    if not np.isfinite(clearance) or clearance < MINIMUM_STEP_INITIAL_CLEARANCE_M:
        raise ValueError("step obstacle initial clearance is below 0.005 m")
    initial_toe = INITIAL_FOOT_CENTER_Y_M + FOOT_HALF_LENGTH_Y_M
    rear_edge = initial_toe + clearance
    center = rear_edge + STEP_OBSTACLE_HALF_LENGTH_Y_M
    return {
        "initial_foot_center_y_m": INITIAL_FOOT_CENTER_Y_M,
        "initial_toe_y_m": initial_toe,
        "obstacle_half_length_y_m": STEP_OBSTACLE_HALF_LENGTH_Y_M,
        "obstacle_rear_edge_y_m": rear_edge,
        "initial_clearance_m": clearance,
        "obstacle_center_y_m": center,
    }
```

In `build_step_xml()`, replace `0.068` and the repeated `0.012` with values from this helper.

- [x] **Step 4: Run the two tests and the existing geometry test**

Run:

```bash
/home/zhijian/miniforge3/envs/fddp_biped/bin/python -m unittest \
  tests.test_fddp_multistep_robustness.MultiStepRobustnessStatisticsTests.test_step_obstacle_geometry_clears_initial_toe_by_five_mm \
  tests.test_fddp_multistep_robustness.MultiStepRobustnessStatisticsTests.test_step_obstacle_geometry_rejects_insufficient_clearance \
  tests.test_fddp_multistep_robustness.MultiStepRobustnessStatisticsTests.test_scenario_model_changes_friction_gravity_and_step_geometry
```

Expected: `Ran 3 tests ... OK`.

- [x] **Step 5: Checkpoint**

Run `git diff --check`. Do not commit in the no-baseline repository.

---

### Task 2: Prove initial integrity and later obstacle encounter

**Files:**
- Modify: `mpc/18_validate_fddp_multistep_c2_final.py:87-135,330-620`
- Modify: `tests/test_fddp_multistep_robustness.py`

**Interfaces:**
- Produces: `_obstacle_contact_snapshot(data, contact_ids) -> dict[str, object]`
- Extends metrics with `step_obstacle_contact_detected`, `step_obstacle_contact_sample_count`, `first_step_obstacle_contact_time_s`, and `maximum_step_obstacle_penetration_m`.

- [x] **Step 1: Write the failing initial-contact integration test**

```python
def test_valid_6mm_obstacle_has_no_initial_foot_contact(self):
    runner = load_runner("test_valid_6mm_initial_contact")
    model, data = initialise_reference_pose(runner, step_height_m=0.006)
    ids = runner._resolve_foot_terrain_contact_ids(model)
    snapshot = runner._obstacle_contact_snapshot(data, ids)
    self.assertFalse(snapshot["detected"])
    self.assertEqual(snapshot["contact_count"], 0)
    self.assertEqual(snapshot["maximum_penetration_m"], 0.0)
```

- [x] **Step 2: Run the test and verify RED**

Expected: error because `_obstacle_contact_snapshot` does not exist.

- [x] **Step 3: Implement obstacle IDs and snapshot using existing contact data**

Extend `_resolve_foot_terrain_contact_ids()` with an `obstacles` set containing names beginning with `stage9_`. Implement:

```python
def _obstacle_contact_snapshot(data, contact_ids):
    feet = set(dict(contact_ids["feet"]).values())
    obstacles = set(contact_ids["obstacles"])
    count = 0
    maximum_penetration_m = 0.0
    for index in range(int(data.ncon)):
        contact = data.contact[index]
        pair = {int(contact.geom1), int(contact.geom2)}
        if pair & feet and pair & obstacles:
            count += 1
            maximum_penetration_m = max(
                maximum_penetration_m, max(0.0, -float(contact.dist))
            )
    return {
        "detected": bool(count),
        "contact_count": count,
        "maximum_penetration_m": maximum_penetration_m,
    }
```

- [x] **Step 4: Add the failing replay encounter test**

```python
def test_6mm_replay_encounters_obstacle_after_initialisation(self):
    runner = load_runner("test_6mm_obstacle_encounter")
    result = runner._run_mode("C2_final", 10.0, step_height_m=0.006)
    metrics = result["metrics"]
    self.assertTrue(metrics["step_obstacle_contact_detected"])
    self.assertGreater(metrics["step_obstacle_contact_sample_count"], 0)
    self.assertGreater(metrics["first_step_obstacle_contact_time_s"], 0.0)
    self.assertLessEqual(metrics["peak_abs_error_deg"], 15.0)
    self.assertEqual(metrics["event_match_ratio"], 1.0)
```

Expected before loop instrumentation: missing metric keys.

- [x] **Step 5: Add minimum loop instrumentation**

Immediately after the initial `mj_forward`, reject any initial obstacle contact. During the existing loop, call `_obstacle_contact_snapshot()` after `mj_step`, accumulate sample count, first time, and maximum penetration, add row diagnostics, and expose the four metrics. Do not alter commands, controller updates, contact fusion, or plant dynamics.

- [x] **Step 6: Run both Task 2 tests**

Expected: `Ran 2 tests ... OK`, and the 6 mm integration run reports encounter after `t=0`.

- [x] **Step 7: Checkpoint**

Run `git diff --check` and compile the modified runner with `python -m py_compile`.

---

### Task 3: Enforce encounter evidence in every step trial

**Files:**
- Modify: `mpc/21_run_fddp_multistep_robustness_matrix.py:66-130,195-240`
- Modify: `tests/test_fddp_multistep_robustness.py`

**Interfaces:**
- Produces: `_scenario_specific_failures(scenario, metrics) -> list[str]`
- Adds failure category: `step_obstacle_not_encountered`.

- [x] **Step 1: Write the failing gate test**

```python
def test_step_trial_requires_real_obstacle_encounter(self):
    batch = load_batch("test_step_encounter_gate")
    scenario = next(
        item for item in batch.statistics.build_scenarios()
        if item.label == "step_6mm"
    )
    self.assertEqual(
        batch._scenario_specific_failures(
            scenario, {"step_obstacle_contact_detected": False}
        ),
        ["step_obstacle_not_encountered"],
    )
    self.assertEqual(
        batch._scenario_specific_failures(
            scenario, {"step_obstacle_contact_detected": True}
        ),
        [],
    )
```

- [x] **Step 2: Run the test and verify RED**

Expected: error because `_scenario_specific_failures` does not exist.

- [x] **Step 3: Implement and connect the gate**

```python
def _scenario_specific_failures(scenario, metrics):
    if (
        float(scenario.step_height_m) > 0.0
        and not bool(metrics.get("step_obstacle_contact_detected", False))
    ):
        return ["step_obstacle_not_encountered"]
    return []
```

Append these failures after `recovery.evaluate_trial(metrics)`, derive `passed` from the final failure list, and add obstacle encounter fields to every trial row. Add `terrain_obstacle_geometry` to the YAML report using the existing terrain helper.

- [x] **Step 4: Run the gate, one-step batch, and aggregation tests**

Expected: all selected tests pass; a failed encounter remains present in the trial and aggregate rows.

- [x] **Step 5: Checkpoint**

Run `git diff --check`; do not commit.

---

### Task 4: Run focused and full three-seed smoke validation

**Files:**
- Regenerate: `results/logs/fddp_multistep_robustness_trials.csv`
- Regenerate: `results/logs/fddp_multistep_robustness_aggregate.csv`
- Regenerate: `results/logs/fddp_multistep_robustness_summary.yaml`
- Regenerate: `results/figures/fddp_multistep_robustness_matrix.png`

**Interfaces:**
- Consumes: Task 1 geometry, Task 2 metrics, Task 3 gate.
- Produces: accepted 32-scenario three-seed smoke evidence.

- [x] **Step 1: Run 2/4/6 mm one-seed trials**

Use the existing `run_trial()` for each registered step scenario. Expected for each: `passed=True`, encounter detected, first encounter time greater than zero, no runtime exception.

- [x] **Step 2: Run all related tests**

```bash
/home/zhijian/miniforge3/envs/fddp_biped/bin/python -m unittest \
  tests.test_fddp_multistep_controller \
  tests.test_fddp_multistep_robustness \
  tests.test_multistep_fault_injection \
  tests.test_multistep_runtime_safety
```

Expected: zero failures.

- [x] **Step 3: Run the full smoke matrix**

```bash
/home/zhijian/miniforge3/envs/fddp_biped/bin/python \
  mpc/21_run_fddp_multistep_robustness_matrix.py --smoke
```

Expected: `scenarios=32 trials=96`; all 96 trials pass. Smoke-level global acceptance remains false only if the script intentionally requires 30 seeds for formal acceptance.

- [x] **Step 4: Audit outputs**

Assert 96 trial rows, zero failed rows, 32 aggregate rows, all step trials encountered the obstacle, and no runtime exceptions.

---

### Task 5: Execute and verify the formal 30-seed matrix

**Files:**
- Regenerate the four Task9 evidence files from Task 4 with formal results.

**Interfaces:**
- Produces: final 32 x 30 statistical robustness report and Wilson intervals.

- [x] **Step 1: Run the formal campaign**

```bash
/home/zhijian/miniforge3/envs/fddp_biped/bin/python \
  mpc/21_run_fddp_multistep_robustness_matrix.py --seed-count 30
```

Expected: `scenarios=32 trials=960 accepted=True`.

- [x] **Step 2: Verify the formal evidence**

Check:

- exactly 960 trial rows and 32 aggregate rows;
- nominal C2 is 30/30;
- every critical scenario success rate is at least 0.90;
- both +/-4 N scenarios are at least 0.90;
- aggregate success rate is at least 0.95;
- every step trial reports a real obstacle encounter;
- no failure row has an empty failure category;
- summary claim boundary is `simulation_only_position_command`.

- [x] **Step 3: Final verification**

Run fresh compilation, the full relevant unittest command, evidence-file existence checks, and `git diff --check`. Report the actual formal result; if any gate fails, retain the evidence and state the remaining failure instead of claiming completion.
