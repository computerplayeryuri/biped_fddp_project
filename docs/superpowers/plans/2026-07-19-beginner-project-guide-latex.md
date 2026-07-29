# Beginner Project Guide LaTeX Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (\`- [ ]\`) syntax for tracking.

**Goal:** Produce a standalone, beginner-oriented Chinese LaTeX guide that explains the complete planar-biped project from modelling through FDDP, MuJoCo, C0/C1/C2, safety and statistical robustness, using current evidence and an honest hardware boundary.

**Architecture:** Reuse the existing 857-line learning guide as the factual and bibliographic base, then create a separate teaching-oriented source with a simpler narrative, explicit intuition boxes, current evidence and presentation-ready summaries. Keep the document standalone under \`docs/overleaf/\`; verify it with existing repository resources and static checks because no local TeX compiler is installed.

**Tech Stack:** XeLaTeX/Overleaf, \`ctexart\`, AMS math, \`booktabs\`, \`tabularx\`, \`longtable\`, \`tcolorbox\`, \`hyperref\`, existing PNG/PDF figures, current YAML/CSV evidence.

## Global Constraints

- Create \`docs/overleaf/biped_project_from_zero_guide_zh.tex\`; do not overwrite \`project_complete_learning_guide_zh.tex\` or modify \`v4.tex\`.
- Prefer current code and machine-readable results over older narrative documents.
- Reuse existing formulas, references, figures and explanations instead of recreating them without need.
- Chinese is primary; retain useful English technical terms on first use.
- Explain every major concept as: definition, motivation, intuition, minimum formula, project implementation, evidence and limitation.
- Preserve the hardware-facing command boundary \`q_cmd = q_ref + delta_q\`; \`tau_ref\` is planning/analysis only.
- Report the latest formal robustness result as 32 scenarios, 30 seeds each, 960/960 trials passed and \`accepted=true\`.
- Report recovery, fault injection, timing and robustness as separate evidence packages.
- Do not claim real-hardware walking.
- State that the repository contains no Git baseline; do not create a partial commit.
- No new controller, simulator or evidence-generation framework is permitted.

---

### Task 1: Freeze the current evidence ledger

**Files:**
- Read: \`docs/overleaf/project_complete_learning_guide_zh.tex\`
- Read: \`docs/2026-07-14_fddp_multistep_simulation_closure_experiment.md\`
- Read: \`results/logs/fddp_multistep_nominal_replay.yaml\`
- Read: \`results/logs/fddp_multistep_controller_ablation.yaml\`
- Read: \`results/logs/fddp_multistep_recovery_selection.yaml\`
- Read: \`results/logs/fddp_multistep_fault_injection.yaml\`
- Read: \`results/logs/fddp_multistep_robustness_summary.yaml\`
- Read: \`results/logs/e6_controller_timing_safety_summary.yaml\`

**Interfaces:**
- Consumes: current project evidence.
- Produces: exact facts used by all later writing tasks.

- [ ] **Step 1: Extract the nominal-planning and replay facts**

Confirm from the existing YAML and experiment record:

- ten-step planning/contact promotion status;
- replay duration;
- ordered contact-event count;
- joint RMSE, maximum torso pitch, height fraction and flight result;
- command path and claim boundary.

- [ ] **Step 2: Extract controller-ablation facts**

Confirm exact C0/C1/phase-only/C2-final values and the five acceptance checks. Preserve:

- C1 IAE improvement relative to C0;
- C2 timing improvement relative to fixed-phase C1;
- peak-pitch and correction limits;
- ordered event count.

- [ ] **Step 3: Extract recovery and fault evidence**

Confirm:

- \`fir_fast\` selected;
- forward \`+4 N\` and backward \`-4 N\` each pass 30/30;
- 15 fault scenarios accepted;
- every injected fault keeps commands finite, within limits and rate bounded;
- software-only claim boundary.

- [ ] **Step 4: Extract formal robustness evidence**

Confirm:

- \`seed_count=30\`, \`scenario_count=32\`, \`trial_count=960\`;
- \`accepted=true\` and aggregate success rate 1.0;
- \`step_6mm=30/30\`;
- all 90 step trials report actual obstacle encounter after initialisation;
- obstacle initial clearance is 0.005 m;
- maximum recorded soft-contact penetration is reported as a simulator diagnostic, not hidden.

- [ ] **Step 5: Preserve the timing limitation**

Record that \`e6_controller_timing_safety_summary.yaml\` contains 61 updates per historical direction/run, no deadline misses and sub-2 ms maxima, but no current 100,000-update final benchmark exists. Do not turn this into a hardware real-time guarantee.

- [ ] **Step 6: Run a stale-claim scan**

Run:

\`\`\`bash
grep -n -E 'Task 7|Task 8|Task 9|尚未完成|93/96|70 trials|132 repository tests' \
  docs/overleaf/project_complete_learning_guide_zh.tex
\`\`\`

Expected: historical guide locations are identified for correction in the new source; the original file remains unchanged.

---

### Task 2: Create the standalone Overleaf source and teaching style

**Files:**
- Create: \`docs/overleaf/biped_project_from_zero_guide_zh.tex\`
- Reuse: \`docs/overleaf/project_complete_learning_guide_zh.tex\`
- Reuse: \`docs/overleaf/latexmkrc\`

**Interfaces:**
- Consumes: existing standalone guide preamble and bibliography.
- Produces: a self-contained XeLaTeX source with stable teaching-box commands.

- [ ] **Step 1: Mechanically duplicate the current guide**

Use the current guide as the base so its equations, labels, bibliography and optional-image handling are preserved:

\`\`\`bash
cp docs/overleaf/project_complete_learning_guide_zh.tex \
   docs/overleaf/biped_project_from_zero_guide_zh.tex
\`\`\`

Expected: both files exist and have the same initial SHA-256; all later patches target only the new file.

- [ ] **Step 2: Replace document metadata**

Set:

\`\`\`latex
\hypersetup{
  pdftitle={从零理解平面双足机器人 FDDP 与 FIR-QP-MPC 项目},
  pdfauthor={MSc Biped FDDP Project}
}
\title{\textbf{从零理解我的平面双足机器人项目}\\[0.4em]
\large 从建模、FDDP、MuJoCo 到 FIR-QP-MPC、接触同步与鲁棒性验证}
\author{面向零基础读者的项目讲解文档}
\date{2026 年 7 月 19 日}
\`\`\`

- [ ] **Step 3: Add reusable teaching boxes**

Add \`tcolorbox\` and define:

\`\`\`latex
\usepackage[most]{tcolorbox}
\newtcolorbox{intuitionbox}{colback=blue!3,colframe=blue!45!black,title=直观理解}
\newtcolorbox{projectbox}{colback=green!3,colframe=green!40!black,title=本项目怎样做}
\newtcolorbox{evidencebox}{colback=orange!4,colframe=orange!65!black,title=证据与结果}
\newtcolorbox{warningbox}{colback=red!3,colframe=red!55!black,title=常见误解与边界}
\`\`\`

Use the boxes only where they materially improve understanding; do not wrap every paragraph.

- [ ] **Step 4: Replace the abstract and opening boundary box**

The abstract must state that the guide begins from zero knowledge and follows the real project timeline. The boundary box must distinguish:

- current defined software/simulation acceptance;
- missing formal 100,000-update timing evidence;
- hardware bench/static/walking work;
- prohibition on claiming hardware walking.

- [ ] **Step 5: Verify the skeleton statically**

Run:

\`\`\`bash
grep -n -E '\\documentclass|\\begin\{document\}|\\tableofcontents|\\end\{document\}' \
  docs/overleaf/biped_project_from_zero_guide_zh.tex
grep -n -E 'intuitionbox|projectbox|evidencebox|warningbox' \
  docs/overleaf/biped_project_from_zero_guide_zh.tex
\`\`\`

Expected: one document class, one begin/end document pair, a table of contents and four teaching-box definitions.

---

### Task 3: Rewrite the guide as a zero-knowledge project story

**Files:**
- Modify: \`docs/overleaf/biped_project_from_zero_guide_zh.tex\`

**Interfaces:**
- Consumes: evidence ledger and teaching-box commands.
- Produces: the complete beginner narrative from robot basics through online control.

- [ ] **Step 1: Add the three reading routes**

The opening section must provide:

- zero-knowledge route: read in order;
- group-meeting route: overview, results, limitations and oral summary;
- thesis route: model, FDDP, controller, experiments and references.

- [ ] **Step 2: Explain the robot before the algorithms**

Explain joint, link, degree of freedom, sagittal plane, torso pitch, CoM, support foot, swing foot, contact force and position-controlled servo. Include:

\`\`\`latex
\[
\bm q_{\mathrm{cmd}}(k)
=
\bm q_{\mathrm{ref}}(k)
+
\Delta\bm q(k).
\]
\`\`\`

Immediately explain that \`q_ref\` is the planned posture, \`Delta q\` is a small online correction and \`q_cmd\` is the actual software command target.

- [ ] **Step 3: Explain the modelling toolchain**

Use one responsibility table for URDF, MJCF, Pinocchio, Crocoddyl and MuJoCo. Explain \`nq=13\`, \`nv=12\`, the quaternion, the planar MuJoCo root and why unequal dimensions do not automatically mean a model error.

- [ ] **Step 4: Explain FDDP from the walking problem**

Start from “choose an entire sequence of postures and controls so the robot follows dynamics and contacts while minimizing cost.” Explain:

\`\`\`latex
\[
\min_{\bm x_{0:N},\bm u_{0:N-1}}
\Phi(\bm x_N)
+
\sum_{k=0}^{N-1}\ell_k(\bm x_k,\bm u_k)
\]
\`\`\`

Then explain state, control, horizon, residual, weight, node, shooting gap, backward pass, forward rollout, line search, regularisation, warm start, continuation and feasibility restoration. Separate textbook FDDP from the project’s fixed contact schedule and gate-driven export.

- [ ] **Step 5: Explain the project’s FDDP progression**

Present the real progression:

\`\`\`text
stand / CoM shift
  -> independent left and right single steps
  -> alternating two-step solution
  -> periodic-cycle seed
  -> 4-step continuation
  -> 6-step continuation
  -> accepted 10-step nominal
\`\`\`

For each stage, explain what new difficulty was introduced and what gate was added.

- [ ] **Step 6: Explain MuJoCo replay**

Explain why optimisation success is not independent validation. Define forward dynamics, actuator proxy, delay, first-order lag, natural contact event, RMSE, pitch, height and unintended flight. Preserve current ten-step replay metrics from Task 1.

- [ ] **Step 7: Explain C0, C1, phase-only and C2-final**

Use one horizontal comparison table:

\`\`\`text
C0         nominal position replay only
C1         C0 + FIR-QP pitch correction
phase-only C0 + contact-event reference retiming
C2-final   C1 + event-type local-leg retiming
\`\`\`

Explain that C1 improves recovery after disturbance rather than preventing the disturbance from occurring.

- [ ] **Step 8: Explain FIR, AR, QP and MPC word by word**

Include:

\`\`\`latex
\[
\widehat y_{k+i}
=
y^{\mathrm{free}}_{k+i}
+
\sum_{j=0}^{M-1}h_j\alpha_{k+i-j},
\qquad
\Delta\bm q_k=\bm G\alpha_k .
\]
\`\`\`

Explain impulse, coefficient, memory length, free response, prediction horizon, receding horizon, objective penalty, box constraint, scalar \`alpha\` and fixed mapping \`G\`. State why this lightweight local controller was chosen for the low-cost position-command interface.

- [ ] **Step 9: Explain contact-phase synchronisation**

Explain liftoff/touchdown, observer, debounce, scheduled versus observed events, event matching, phase offset and local-leg retiming. Make clear that C2 retimes the reference and does not generate a new foothold or re-solve whole-body dynamics.

---

### Task 4: Integrate safety, statistics and current completion status

**Files:**
- Modify: \`docs/overleaf/biped_project_from_zero_guide_zh.tex\`
- Reuse figures from: \`docs/overleaf/*.png\`
- Reuse latest figure: \`results/figures/fddp_multistep_robustness_matrix.png\`

**Interfaces:**
- Consumes: current recovery, fault and formal robustness evidence.
- Produces: current results chapter and honest completion table.

- [ ] **Step 1: Explain recovery authority**

Describe the same-protocol \(\pm4\) N experiment, 30 seeds per direction, acceptance threshold at least 27/30 and the selected \`fir_fast\` result of 30/30 in both directions. State that bounded foothold adaptation was not executed because it would require nominal/contact-feasibility regeneration.

- [ ] **Step 2: Explain runtime safety and fault injection**

Group the 15 scenarios into measurement, contact sensor/event, online controller, timing, command, actuator and startup-integrity failures. Explain NORMAL, DEGRADED, FALLBACK and SAFE_HOLD using concrete examples.

- [ ] **Step 3: Explain statistical robustness**

Explain trial, seed, scenario, critical scenario, failure category, aggregate success rate and Wilson interval. Include:

\`\`\`latex
\[
\hat p = \frac{s}{n},
\]
\`\`\`

and explain why the confidence interval matters even when all sampled trials pass.

- [ ] **Step 4: Explain the corrected obstacle experiment**

State:

- old obstacle geometry intersected the initial foot and was invalid;
- corrected rear edge is 0.0775 m, five millimetres ahead of the initial toe;
- initial foot-obstacle contact is forbidden;
- later obstacle encounter is mandatory;
- 90/90 step trials encountered the obstacle;
- maximum reported penetration is a MuJoCo soft-contact diagnostic, not a rigid-body guarantee.

- [ ] **Step 5: Add the final formal matrix table**

Include:

- 32 scenarios;
- 30 seeds per scenario;
- 960/960 trials passed;
- nominal 30/30;
- \`step_6mm\` 30/30;
- both \(\pm4\) N critical scenarios 30/30;
- 0 runtime exceptions;
- \`accepted=true\`;
- claim boundary \`simulation_only_position_command\`.

- [ ] **Step 6: Replace every stale completion statement**

The current-status table must distinguish:

- completed: model audit, FDDP progression, ten-step planning/contact, full replay, C0/C1/C2 ablation, bounded \(\pm4\) N recovery, fault injection and 32-by-30 robustness;
- limited: historical E6 timing evidence is short and not a 100,000-update final benchmark;
- remaining: hardware communication, servo calibration, limit/sign verification, protected bench test, supported standing/walking and real experiments.

The document must not contain the stale statements “Task 7/8/9 尚未完成”, “93/96” or “70 trials” as current conclusions.

---

### Task 5: Add presentation paths, glossary, file map and references

**Files:**
- Modify: \`docs/overleaf/biped_project_from_zero_guide_zh.tex\`
- Reuse: \`docs/overleaf/PPT阶段性成果证据索引.md\`
- Reuse: \`docs/overleaf/PPT参考依据_可复制.md\`
- Reuse: \`docs/overleaf/fddp_standard_vs_project_flow_references.md\`

**Interfaces:**
- Consumes: completed explanatory chapters.
- Produces: a guide usable for study, group meetings, thesis writing and defence.

- [ ] **Step 1: Add one-, three- and ten-minute oral summaries**

Each summary must follow:

\`\`\`text
problem -> method -> evidence -> contribution -> boundary
\`\`\`

The one-minute version must fit in one paragraph. The three-minute version must mention FDDP, MuJoCo, C1, C2, safety and 960-trial robustness. The ten-minute version must map directly to recommended PPT pages.

- [ ] **Step 2: Add likely supervisor questions**

Answer at least:

- Why FDDP instead of only inverse kinematics?
- Why not run FDDP online?
- Why use FIR rather than a full analytical model?
- Is the QP really MPC?
- What is the difference between C1 and C2?
- Why is MuJoCo replay independent evidence?
- Does 960/960 mean the robot will never fall?
- Why is \`tau_ref\` not sent to the servos?
- What software work remains before hardware walking?

- [ ] **Step 3: Add the bilingual glossary**

Include at least 35 terms covering dynamics, contact, optimisation, identification, MPC, safety and statistics. Every entry must contain Chinese meaning and its role in this project.

- [ ] **Step 4: Update the file and evidence map**

Index the main files under \`model/\`, \`fddp/\`, \`simulation/\`, \`mpc/\`, \`hardware/\`, \`transfer/\`, \`results/logs/\`, \`results/figures/\` and \`tests/\`. Avoid listing every file; include only files that help the reader trace one stage of the pipeline.

- [ ] **Step 5: Preserve and extend the bibliography**

Retain the existing DDP/FDDP, constrained MPC, MuJoCo, Pinocchio, Crocoddyl, control-limited DDP and humanoid locomotion references. Cite each near the algorithm statement it supports. Project-specific parameter values must cite project evidence rather than academic papers.

---

### Task 6: Validate and hand off the Overleaf source

**Files:**
- Validate: \`docs/overleaf/biped_project_from_zero_guide_zh.tex\`
- Preserve: \`docs/overleaf/project_complete_learning_guide_zh.tex\`
- Preserve: \`docs/overleaf/v4.tex\`

**Interfaces:**
- Consumes: completed LaTeX guide.
- Produces: verified source ready for Overleaf upload.

- [ ] **Step 1: Verify source preservation**

Confirm the SHA-256 of \`project_complete_learning_guide_zh.tex\` and \`v4.tex\` still matches the pre-edit values. Only the new guide and planning/spec documents may change.

- [ ] **Step 2: Run structural checks**

Run an inline check reusing \`brace_balance\` from the existing validator:

\`\`\`bash
/home/zhijian/miniforge3/envs/fddp_biped/bin/python -c \
'from pathlib import Path; from tools._validate_tex_learning_docs import brace_balance; p=Path("docs/overleaf/biped_project_from_zero_guide_zh.tex"); t=p.read_text(encoding="utf-8"); assert brace_balance(t)==0; assert t.count("\\begin{document}")==1; assert t.count("\\end{document}")==1; print("structure=OK")'
\`\`\`

Expected: \`structure=OK\`.

- [ ] **Step 3: Run label, reference and stale-claim checks**

Check:

- no duplicate labels;
- all internal references resolve;
- all cited bibliography keys exist as inline \`\\bibitem\` entries;
- no current claim contains \`93/96\`, \`Task 7/8/9 尚未完成\` or \`70 trials\`;
- required current phrases include \`960/960\`, \`accepted\` and \`simulation_only_position_command\`.

- [ ] **Step 4: Check paths and formatting**

Run:

\`\`\`bash
grep -nE '[[:blank:]]+$' docs/overleaf/biped_project_from_zero_guide_zh.tex
grep -n -E 'Overfull|Undefined control sequence|LaTeX Error' \
  docs/overleaf/biped_project_from_zero_guide_zh.log 2>/dev/null || true
\`\`\`

Expected: no trailing-whitespace output. A log check is only applicable if compilation produced a log.

- [ ] **Step 5: Attempt compilation only with an existing compiler**

Current environment discovery found no \`xelatex\`, \`latexmk\`, \`tectonic\`, \`lualatex\` or \`pdflatex\`. Do not install a TeX distribution without user approval. Therefore:

- perform static validation locally;
- state explicitly that local PDF compilation was unavailable;
- provide the source for Overleaf XeLaTeX compilation;
- if the user later authorises Overleaf push or TeX installation, compile twice and inspect the resulting PDF.

- [ ] **Step 6: Final evidence audit**

Re-read the final status table, oral summary and conclusion against the five YAML sources from Task 1. Report exact source path, line count, static-validation result and the local compilation limitation. Do not commit or push because the repository has no baseline commit.
