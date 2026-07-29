# Software Robustness Matrix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add and run a software-only robustness matrix for the current scripted C2 locomotion preview.

**Architecture:** A new Stage 9 script imports the existing C2 contact observer/synchronizer, creates periodic scripted step references, applies scenario perturbations, and writes per-scenario CSV plus a combined YAML summary. Tests validate the non-MuJoCo logic before implementation.

**Tech Stack:** Python 3.10 in the `fddp_biped` conda environment, MuJoCo, NumPy, Pandas, Matplotlib, PyYAML, unittest.

## Global Constraints

- Do not claim full FDDP walking from scripted periodic stepping.
- Do not claim hardware robustness from simulation.
- Keep the existing Stage 8 script intact.
- Outputs must live under `results/logs` and `results/figures`.
- Use `/home/zhijian/miniforge3/envs/fddp_biped/bin/python` for tests and simulation.

---

### Task 1: Test Stage 9 Helper Logic

**Files:**
- Create: `tests/test_robustness_matrix.py`

**Interfaces:**
- Consumes: future `mpc/13_run_scripted_robustness_matrix.py`
- Produces: failing tests for `build_scenarios`, `expected_events_for_duration`, `build_step_xml`, and `scenario_passed`

- [ ] Write failing tests for scenario categories, event counts, step XML insertion, and summary acceptance.
- [ ] Run `python -m unittest tests/test_robustness_matrix.py -v` and verify it fails because the module is missing.

### Task 2: Implement Stage 9 Robustness Matrix

**Files:**
- Create: `mpc/13_run_scripted_robustness_matrix.py`

**Interfaces:**
- Produces:
  - `Scenario`
  - `build_scenarios() -> list[Scenario]`
  - `expected_events_for_duration(duration_s: float) -> list[dict]`
  - `build_step_xml(step_height_m: float) -> str`
  - `scenario_passed(metrics: dict) -> dict`
  - `run_scenario(scenario: Scenario) -> dict`

- [ ] Implement the smallest code needed for the tests to pass.
- [ ] Run the new unit test file and verify it passes.

### Task 3: Run the Matrix

**Files:**
- Output: `results/logs/stage9_scripted_robustness_matrix_summary.yaml`
- Output: `results/logs/stage9_scripted_robustness_matrix_table.csv`
- Output: `results/logs/stage9_*.csv`
- Output: `results/figures/stage9_scripted_robustness_matrix_pass_fail.png`
- Output: `results/figures/stage9_scripted_robustness_matrix_pitch.png`

- [ ] Run `python mpc/13_run_scripted_robustness_matrix.py --matrix full`.
- [ ] If one or more scenarios fail, keep the results and report the exact failed conditions instead of hiding them.

### Task 4: Update Thesis Evidence

**Files:**
- Modify: `reporting/generated/thesis_update_2026-07-08_c0_c1_c2_stage8.tex`
- Create: `reporting/generated/thesis_update_2026-07-08_stage9_robustness_matrix.tex`

- [ ] Add a Stage 9 robustness-matrix section with the actual pass/fail counts and strongest/weakest stress cases.
- [ ] Keep wording explicit that this is scripted simulation, not full FDDP walking or hardware validation.
