# Robustness Presentation Figures Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the two weak robustness plots with publication-style, project-factual figures that show bidirectional push recovery and all 32 scripted robustness scenarios, then provide bilingual slide wording and speaker notes.

**Architecture:** Add one focused reporting script that reads only the existing FDDP robustness CSV/YAML evidence and reuses the repository's existing publication-style helpers. Keep all statistics and plotting functions side-effect free, expose a small CLI for regeneration, and validate numerical claims before rendering. Do not change controllers, simulation logic, thresholds, or the PowerPoint file in this task.

**Tech Stack:** Python 3, pandas, NumPy, matplotlib, PyYAML, unittest, existing `reporting/regenerate_publication_figures.py` styling utilities.

## Global Constraints

- Treat the existing files in `results/logs/` as the sole evidence source; do not rerun FDDP or MuJoCo.
- Preserve the claim boundary: these figures prove scripted simulation robustness only, not hardware robustness.
- Keep all 32 scenarios visible and traceable; do not replace them with a category average.
- Show observed pass rate and Wilson 95% confidence interval as different quantities. For 30/30 trials, the lower Wilson bound is approximately 88.65%, so it does not pass a 90% lower-bound requirement; the repository's scenario acceptance gate is the observed pass rate.
- Use white backgrounds, an Okabe–Ito-compatible palette, direct labels, redundant marker shapes, and vector PDF plus 300 dpi PNG output.
- Reuse existing code and data structures. Do not add a new simulation runner or duplicate the controller implementation.
- Git commits in this plan are conditional on the user configuring a repository-local Git identity. Do not modify global Git identity automatically.

---

## Task 1: Add tested evidence-loading and summary functions

**Files:**

- Create: `reporting/generate_robustness_ppt_figures.py`
- Create: `tests/test_robustness_ppt_figures.py`
- Read: `results/logs/fddp_multistep_recovery_sweep.csv`
- Read: `results/logs/fddp_multistep_recovery_selection.yaml`
- Read: `results/logs/fddp_multistep_robustness_trials.csv`
- Read: `results/logs/fddp_multistep_robustness_aggregate.csv`
- Read: `results/logs/fddp_multistep_robustness_summary.yaml`

- [ ] **Step 1: Write failing tests for the factual dataset contract**

Add tests that require the loader to report the known project facts:

```python
class RobustnessEvidenceTests(unittest.TestCase):
    def test_push_summary_matches_recorded_trials(self):
        trials, selection = figures.load_push_evidence(ROOT / "results/logs")
        summary = figures.summarise_push_trials(trials, pitch_gate_deg=15.0)
        self.assertEqual(len(trials), 60)
        self.assertEqual(summary.loc["forward", "passed"], 30)
        self.assertEqual(summary.loc["backward", "passed"], 30)
        self.assertAlmostEqual(summary.loc["forward", "mean_peak_deg"], 9.024600, places=5)
        self.assertAlmostEqual(summary.loc["backward", "mean_peak_deg"], 13.712496, places=5)
        self.assertEqual(selection["accepted"], True)

    def test_scenario_profile_preserves_all_trials(self):
        trials, aggregate, summary = figures.load_scenario_evidence(ROOT / "results/logs")
        profile = figures.build_scenario_profile(trials, aggregate)
        self.assertEqual(len(profile), 32)
        self.assertEqual(len(trials), 960)
        self.assertEqual(int(profile["passed"].sum()), 960)
        self.assertTrue((profile["success_rate"] == 1.0).all())
        self.assertAlmostEqual(profile.iloc[0]["wilson_low"], 0.886486606826, places=10)
        self.assertEqual(summary["total_trials"], 960)
```

- [ ] **Step 2: Run the focused tests and confirm they fail for the missing module**

Run:

```bash
/home/zhijian/miniforge3/envs/fddp_biped/bin/python -m unittest tests.test_robustness_ppt_figures -v
```

Expected: import failure for `reporting.generate_robustness_ppt_figures`.

- [ ] **Step 3: Implement minimal evidence loading and validation**

Create these functions in `reporting/generate_robustness_ppt_figures.py`:

```python
def load_push_evidence(log_dir: Path) -> tuple[pd.DataFrame, dict]: ...
def summarise_push_trials(trials: pd.DataFrame, pitch_gate_deg: float) -> pd.DataFrame: ...
def load_scenario_evidence(log_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]: ...
def build_scenario_profile(trials: pd.DataFrame, aggregate: pd.DataFrame) -> pd.DataFrame: ...
```

The implementation must:

- reject missing required columns with a clear `ValueError`;
- preserve the aggregate CSV row order as the presentation order;
- compute per-scenario minimum, mean, and maximum `peak_abs_error_deg` from the trial CSV;
- merge recorded `success_rate`, `wilson_low`, `wilson_high`, `critical`, and category fields from the aggregate CSV;
- verify that aggregate trial counts equal the grouped raw trial counts;
- expose push margin as `15.0 - max_peak_deg`, not as a new acceptance criterion.

- [ ] **Step 4: Run tests and confirm factual summaries pass**

Run the focused unittest command again. Expected: all Task 1 tests pass.

- [ ] **Step 5: Commit after identity is available**

```bash
git add reporting/generate_robustness_ppt_figures.py tests/test_robustness_ppt_figures.py
git commit -m "test: validate robustness presentation evidence"
```

---

## Task 2: Build the bidirectional push-recovery figure

**Files:**

- Modify: `reporting/generate_robustness_ppt_figures.py`
- Modify: `tests/test_robustness_ppt_figures.py`
- Generate: `results/figures/presentation/robustness_bidirectional_push_profile.png`
- Generate: `results/figures/presentation/robustness_bidirectional_push_profile.pdf`

- [ ] **Step 1: Write a failing rendering test**

Test `plot_bidirectional_push_profile(...)` using a temporary output directory. Assert that both files exist, are non-empty, and that the PNG is at least 3000 pixels wide. Also assert that the returned annotation data contains `30/30`, `9.02`, and `13.71`.

- [ ] **Step 2: Implement the figure using existing style helpers**

Import and reuse:

```python
from reporting.regenerate_publication_figures import OKABE_ITO, apply_publication_style
```

Implement:

```python
def plot_bidirectional_push_profile(
    trials: pd.DataFrame,
    selection: dict,
    output_dir: Path,
) -> tuple[list[Path], dict[str, str]]: ...
```

Required visual encoding:

- x-axis: `Forward push (+4 N)` and `Backward push (-4 N)`;
- y-axis: `Peak absolute pitch error (deg)`;
- show every seed as a small deterministic-jitter point;
- overlay mean and full observed range;
- draw the 15 degree pitch gate as a red dashed horizontal line;
- label each direction with `30/30 passed`, mean peak error, and remaining worst-case margin;
- subtitle: `0.12 s pulse at t = 0.8 s; 30 seeds per direction`;
- footnote: `Scripted MuJoCo/FDDP validation; not a hardware result`;
- use blue for forward, vermillion for backward, and a distinct diamond marker for the closer-to-limit backward case.

Save a 16:9-friendly PNG at 300 dpi and a vector PDF with a white background.

- [ ] **Step 3: Run the rendering test and regenerate the production figure**

```bash
/home/zhijian/miniforge3/envs/fddp_biped/bin/python -m unittest tests.test_robustness_ppt_figures -v
/home/zhijian/miniforge3/envs/fddp_biped/bin/python reporting/generate_robustness_ppt_figures.py --figure push
```

- [ ] **Step 4: Inspect the rendered PNG visually**

Check that labels remain readable at slide size, both data clouds are visible, the gate line does not obscure annotations, and the plot remains interpretable in grayscale.

- [ ] **Step 5: Commit after identity is available**

```bash
git add reporting/generate_robustness_ppt_figures.py tests/test_robustness_ppt_figures.py results/figures/presentation/robustness_bidirectional_push_profile.png results/figures/presentation/robustness_bidirectional_push_profile.pdf
git commit -m "feat: add bidirectional push recovery figure"
```

---

## Task 3: Build the 32-scenario robustness profile

**Files:**

- Modify: `reporting/generate_robustness_ppt_figures.py`
- Modify: `tests/test_robustness_ppt_figures.py`
- Generate: `results/figures/presentation/robustness_32_scenario_profile.png`
- Generate: `results/figures/presentation/robustness_32_scenario_profile.pdf`

- [ ] **Step 1: Write failing tests for scenario ordering and visual metadata**

Require:

- exactly 32 distinct scenario labels;
- the first label is `Nominal`;
- `Combined A`, `Combined B`, and `Combined C` are present;
- critical and noncritical scenarios both exist;
- output metadata reports `32 scenarios`, `960 trials`, and `960/960 passed`;
- PNG and PDF are produced and non-empty.

- [ ] **Step 2: Add explicit English labels and category ordering**

Use a mapping rather than exposing raw CSV identifiers. Labels must include units, for example:

```python
"push_forward_4n": "Push +4 N"
"friction_040": "Friction mu = 0.40"
"slope_minus_5deg": "Slope -5 deg"
"command_delay_20ms": "Command delay 20 ms"
"actuator_tau_50ms": "Actuator tau = 50 ms"
"combined_c": "Combined C"
```

Keep the CSV order within the categories: nominal, push, friction, slope, step, command delay, sensor delay, actuator gain, actuator time constant, noise, combined.

- [ ] **Step 3: Implement a readable two-block forest profile**

Implement:

```python
def plot_robustness_scenario_profile(
    profile: pd.DataFrame,
    output_dir: Path,
) -> tuple[list[Path], dict[str, str]]: ...
```

Use two blocks of 16 scenarios so every label stays readable on a 16:9 slide. Each block has two aligned panels:

1. `Peak |pitch error| (deg)`: a central mean marker with minimum-to-maximum observed range and a 15 degree gate.
2. `Observed success rate`: a point estimate with Wilson 95% confidence interval, a 90% observed-rate gate, and a 95% aggregate target reference.

Encoding rules:

- critical scenarios: filled vermillion circles;
- noncritical scenarios: open blue circles;
- alternating subtle category bands and direct category headers;
- no green/red-only success encoding;
- header summary: `32 scenarios x 30 seeds = 960 trials; 960/960 passed`;
- footnote: `Per-scenario acceptance uses observed success >= 90%; Wilson intervals show sampling uncertainty`;
- explicitly retain the Wilson lower bound near 88.65% for 30/30, rather than clipping it at 90%;
- annotate that all displayed results are scripted simulation results.

- [ ] **Step 4: Run tests and regenerate the production figure**

```bash
/home/zhijian/miniforge3/envs/fddp_biped/bin/python -m unittest tests.test_robustness_ppt_figures -v
/home/zhijian/miniforge3/envs/fddp_biped/bin/python reporting/generate_robustness_ppt_figures.py --figure scenarios
```

- [ ] **Step 5: Inspect both color and grayscale readability**

Verify all 32 labels, category groupings, two gates, and uncertainty intervals are legible when inserted at full-slide width. Confirm orange/blue meanings remain distinguishable through fill state and marker shape in grayscale.

- [ ] **Step 6: Commit after identity is available**

```bash
git add reporting/generate_robustness_ppt_figures.py tests/test_robustness_ppt_figures.py results/figures/presentation/robustness_32_scenario_profile.png results/figures/presentation/robustness_32_scenario_profile.pdf
git commit -m "feat: add full robustness scenario profile"
```

---

## Task 4: Add bilingual slide text and speaker notes

**Files:**

- Create: `docs/2026-07-22_robustness_figures_speaker_notes_zh_en.md`

- [ ] **Step 1: Write the slide-ready English captions**

For the push figure, cover: test purpose, fixed disturbance, bidirectional result, and the fact that the backward case has less pitch margin.

For the 32-scenario figure, cover: scenario coverage, total trials, observed pass count, uncertainty interpretation, and simulation-only claim boundary.

- [ ] **Step 2: Write detailed Chinese speaker notes**

Explain in plain language:

- why bidirectional testing is more informative than one success-rate bar;
- what each dot, range, mean marker, and gate means;
- why 30/30 does not mean the true success probability is known to be 100%;
- why the Wilson lower bound is below 90% even though the observed result is 100%;
- which scenarios are most demanding according to pitch margin;
- what may be claimed as stability and robustness, and what still requires hardware evidence.

- [ ] **Step 3: Cross-check every number against the generated metadata**

Do not type independent values into the notes. Copy the values reported by the tested plotting script so the figure and narration cannot drift.

- [ ] **Step 4: Commit after identity is available**

```bash
git add docs/2026-07-22_robustness_figures_speaker_notes_zh_en.md
git commit -m "docs: add robustness figure speaker notes"
```

---

## Task 5: Final verification and handoff

**Files:**

- Verify: `reporting/generate_robustness_ppt_figures.py`
- Verify: `tests/test_robustness_ppt_figures.py`
- Verify: `results/figures/presentation/*.png`
- Verify: `results/figures/presentation/*.pdf`
- Verify: `docs/2026-07-22_robustness_figures_speaker_notes_zh_en.md`

- [ ] **Step 1: Run the complete focused verification**

```bash
/home/zhijian/miniforge3/envs/fddp_biped/bin/python -m unittest tests.test_robustness_ppt_figures -v
/home/zhijian/miniforge3/envs/fddp_biped/bin/python reporting/generate_robustness_ppt_figures.py --figure all
```

- [ ] **Step 2: Verify output inventory and file integrity**

Confirm four figure files exist, PNGs open at expected resolution, PDFs are one page each, and no raw scenario identifiers remain in visible labels.

- [ ] **Step 3: Perform claim-boundary review**

Search titles, captions, and notes for unsupported language such as `hardware validated`, `guaranteed`, or `proven under all disturbances`. The final wording must say `scripted simulation validation` and identify hardware testing as future validation.

- [ ] **Step 4: Deliver links and a concise presentation script**

Provide clickable links to both PNGs, both PDFs, the plotting source, the tests, and the bilingual notes. Explain which existing PowerPoint robustness slide each figure replaces; do not modify the deck unless the user separately authorizes it.

