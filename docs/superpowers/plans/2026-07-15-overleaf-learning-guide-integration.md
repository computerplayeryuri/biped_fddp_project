# Overleaf Complete Learning Guide Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Merge the Word algorithm explanation and the complete pasted project walkthrough into `project_complete_learning_guide_zh.tex` without duplication, update every stale project claim, compile cleanly with XeLaTeX, and synchronise the verified source to Overleaf.

**Architecture:** Keep the existing LaTeX guide as the structural authority and merge only source-unique material into topic-owned sections. Treat current YAML/CSV evidence as the factual authority, turn repeated Q&A into cross-referenced compact answers, and preserve a strict separation between planning acceptance, MuJoCo replay acceptance, controller evidence, scripted gait evidence, and hardware boundaries.

**Tech Stack:** XeLaTeX, latexmk, `ctexart`, `amsmath`, `booktabs`, `longtable`, `tabularx`, PowerShell/WSL read-only evidence checks, Git/Overleaf synchronisation.

## Global Constraints

- Modify only `docs/overleaf/project_complete_learning_guide_zh.tex` and learning-guide support assets; do not modify `docs/overleaf/v4.tex`.
- The current supported claim is `simulation validation + runtime-loadability`.
- `tau_ref` is FDDP planning/analysis output only and must never be described as a hardware torque command.
- Stage 8/9 are scripted gait experiments, not FDDP-generated continuous gait.
- The accepted 10-step result is planning/contact evidence; full 10 s MuJoCo natural-contact replay is not accepted.
- Preserve the single-step `solver_return_success=false` limitation, C2 `3/4 matched` limitation, and 4 N push `0/5` negative result.
- Source priority is current YAML/CSV/code, then existing LaTeX, then DOCX and pasted text.
- Use XeLaTeX and keep the document independently compilable on Overleaf.
- Do not write a permanent test-count claim unless the full suite is rerun during this implementation.
- This workspace has no confirmed Git repository at `docs/overleaf`; do not invent commits. Synchronise only through a verified authenticated Overleaf clone/remote.

---

### Task 1: Create a source and evidence baseline

**Files:**
- Read: `docs/overleaf/project_complete_learning_guide_zh.tex`
- Read: `D:\毕业论文\机器人控制流程图\算法讲述文档\FDDP与FIR_QP_MPC算法讲述与参数说明.docx`
- Read: `C:\Users\94277\.codex\attachments\2dd63ab9-f882-4323-aa0d-2ca4b09697a9\pasted-text.txt`
- Read: `transfer/trajectories/fddp_multistep_10step_ref.yaml`
- Read: `results/logs/fddp_multistep_nominal_replay_smoke_2step.yaml`
- Read: `results/logs/fddp_single_step_left_swing_diagnostics.yaml`
- Read: `results/logs/fddp_two_step_alternating_diagnostics.yaml`
- Read: `results/logs/fddp_two_step_alternating_mujoco_replay.yaml`
- Read: `results/logs/c1_qp2_fir_mpc_tuning_summary.yaml`
- Read: `results/logs/fddp_two_step_natural_contact_c2.yaml`
- Read: `results/logs/fddp_two_step_robustness_summary.yaml`
- Read: `results/logs/e6_controller_timing_safety_summary.yaml`
- Create: `docs/overleaf/project_complete_learning_guide_zh.tex.pre_integration_backup`

**Interfaces:**
- Consumes: the approved design at `docs/superpowers/specs/2026-07-15-overleaf-learning-guide-integration-design.md`.
- Produces: an immutable pre-integration backup and a verified fact set used by every later task.

- [ ] **Step 1: Create the backup and record hashes**

Run:

```powershell
$root='\\wsl$\Ubuntu-22.04\home\zhijian\biped_fddp_project\docs\overleaf'
Copy-Item -LiteralPath "$root\project_complete_learning_guide_zh.tex" `
  -Destination "$root\project_complete_learning_guide_zh.tex.pre_integration_backup" -Force
Get-FileHash -Algorithm SHA256 -LiteralPath `
  "$root\project_complete_learning_guide_zh.tex", `
  "$root\project_complete_learning_guide_zh.tex.pre_integration_backup"
```

Expected: the two SHA-256 values are identical.

- [ ] **Step 2: Verify the authoritative 10-step acceptance fields**

Run:

```bash
cd /home/zhijian/biped_fddp_project
sed -n '1,90p' transfer/trajectories/fddp_multistep_10step_ref.yaml
```

Expected fields:

```text
acceptance_state: planning_and_contact_accepted
solver_stop: 4.410209249788341e-08
solver_iterations: 66
shooting_gap_norm: 0.0
contact_violation_count: 0
maximum_friction_utilisation: 0.5807076650281457
minimum_cop_margin_m: 0.0023913266893361013
```

- [ ] **Step 3: Verify replay/controller/robustness limitations**

Run:

```bash
cd /home/zhijian/biped_fddp_project
grep -nE 'acceptance:|passed:|event_match_ratio|matched_contact_events|skipped_contact_events|max_abs_phase_offset_ms|push_plus_4N|success_count|fallback_available' \
  results/logs/fddp_multistep_nominal_replay_smoke_2step.yaml \
  results/logs/fddp_two_step_natural_contact_c2.yaml \
  results/logs/fddp_two_step_robustness_summary.yaml \
  results/logs/e6_controller_timing_safety_summary.yaml
```

Expected: smoke replay acceptance false with 50% event matching; natural C2 is 3 matched/1 skipped; push is 0/5 in C0 and C2; E6 has no complete fallback/limiter counters.

### Task 2: Correct the front matter and project-status narrative

**Files:**
- Modify: `docs/overleaf/project_complete_learning_guide_zh.tex`

**Interfaces:**
- Consumes: Task 1 verified fact set.
- Produces: a consistent abstract, opening boundary callout, orientation section, and current conclusion vocabulary.

- [ ] **Step 1: Update the date and status macros**

Add a partial-status macro beside the existing pass/fail macros:

```latex
\newcommand{\partialstatus}{\textcolor{orange!75!black}{\textbf{PARTIAL}}}
\newcommand{\pending}{\textcolor{blue!60!black}{\textbf{PENDING}}}
```

Set the document date to `2026 年 7 月 15 日（依据当前项目证据更新）`.

- [ ] **Step 2: Replace the stale abstract and boundary callout**

The front matter must state exactly:

```latex
当前十步 FDDP nominal 已通过 planning 与独立 contact-feasibility gates；
但 full 10 s MuJoCo natural-contact replay 尚未通过最终 event-matching gate。
```

Delete every front-matter statement saying that the 10-step contact gate failed or the final nominal was not generated.

- [ ] **Step 3: Add a compact intuitive orientation table**

After `\section{先用一句话理解整个项目}`, add one table with columns `模块`, `直观角色`, and `技术职责` for FDDP, MuJoCo, C0, C1, C2, robustness, and runtime. Keep each technical cell to one or two sentences; do not repeat equations that belong to later sections.

- [ ] **Step 4: Run a front-matter stale-claim scan**

Run:

```powershell
Select-String -LiteralPath '\\wsl$\Ubuntu-22.04\home\zhijian\biped_fddp_project\docs\overleaf\project_complete_learning_guide_zh.tex' `
  -Pattern '十步.*失败','final nominal.*没有生成','contact.*pending'
```

Expected: no stale result claim remains; `contact pending` may appear only when describing historical candidate state and must be explicitly labelled historical.

### Task 3: Merge the general FDDP/MPC explanation without duplication

**Files:**
- Modify: `docs/overleaf/project_complete_learning_guide_zh.tex`

**Interfaces:**
- Consumes: the DOCX general-algorithm tables and existing Sections `sec:signals`, `sec:fddp`, `sec:contact`, and `sec:quintic`.
- Produces: one authoritative algorithm explanation and one general-versus-project comparison table.

- [ ] **Step 1: Add one general-versus-project comparison table**

Place the table after the signal dictionary. It must compare:

```text
offline/online role
model used online
optimisation variable
executed command
tau_ref interpretation
controller objective
validated scope
```

Use `longtable` with concise cells. Do not reproduce the DOCX's separate FDDP and MPC symbol tables because the existing signal dictionary already owns those definitions.

- [ ] **Step 2: Extend the FDDP section with the algorithmic iteration**

Add a subsection `FDDP 的 backward pass、forward rollout 与 regularisation` containing:

```latex
\begin{enumerate}
  \item 沿当前候选轨迹计算 dynamics derivatives 和 cost derivatives；
  \item backward pass 求局部 feedforward direction 与 feedback gain；
  \item forward rollout 使用 line search 生成新的 state/control trajectory；
  \item 根据 cost、defect 与 numerical regularisation 接受或拒绝 trial step；
  \item 重复直到 stopping criterion、feasibility threshold 或 iteration budget 被触发。
\end{enumerate}
```

Explicitly distinguish first-order solver stopping from post-solve physical/geometry gates.

- [ ] **Step 3: Add the rigid-contact dynamics interpretation**

Add the equations:

```latex
M(q)\ddot q+h(q,\dot q)=S^T\tau+J_c^T\lambda,
\qquad
J_c\ddot q+\dot J_c\dot q=0.
```

Define every symbol in prose and clarify that the 6D contact is a rigid-flat-foot modelling assumption, not proof of friction/CoP feasibility.

- [ ] **Step 4: Keep the quintic explanation single-owned**

Retain only one derivation of `s(r)=10r^3-15r^4+6r^5` and the swing arc. Elsewhere replace repeated derivations with `\hyperref[sec:quintic]{...}` links.

### Task 4: Replace the stale multi-step and replay evidence

**Files:**
- Modify: `docs/overleaf/project_complete_learning_guide_zh.tex`
- Copy if absent: `results/figures/fddp_multistep_contact_feasibility.png` to `docs/overleaf/fddp_multistep_contact_feasibility.png`
- Copy if absent: `results/figures/fddp_multistep_nominal_replay_smoke_2step.png` to `docs/overleaf/fddp_multistep_nominal_replay_smoke_2step.png`
- Copy if absent: `results/figures/fddp_two_step_alternating_mujoco_replay.png` to `docs/overleaf/fddp_two_step_alternating_mujoco_replay.png`

**Interfaces:**
- Consumes: accepted 10-step YAML, smoke replay YAML, single/two-step diagnostics.
- Produces: a current evidence ladder whose planning and replay states cannot be confused.

- [ ] **Step 1: Rewrite `sec:multistep` as an evidence ladder**

Use a table with rows:

```text
Double-support CoM shift | planning evidence | not walking
Single steps | feasibility gates pass | first-order convergence not achieved
Alternating two-step | solver and replay integrity pass
Periodic seed | continuation evidence
10-step nominal | planning/contact accepted
10-step-reference replay | smoke event gate failed
```

- [ ] **Step 2: Insert the accepted 10-step metric table**

Use the exact values:

```text
duration 10.0 s
steps 10
iterations 66
stop 4.410e-8
gap 0
support error 55.27 um
landing error 35.73 um
clearance 19.993 mm
joint margin 11.790 mrad
forward CoM 50.087 mm
normal force 9.913 N
friction utilisation 0.5807
CoP margin 2.391 mm
contact violations 0/1300 active-contact rows
```

- [ ] **Step 3: Replace the negative contact-feasibility caption**

The figure caption must say it is accepted post-solve planar-sagittal resultant contact evidence and mention the out-of-plane diagnostic-only limitation. Delete the old caption claiming 75 violations, friction 2.165, and CoP -416.6 mm.

- [ ] **Step 4: Add the smoke replay limitation table**

Include finite true, fall false, RMSE 2.880 deg, pitch 9.391 deg, height fraction 0.979, penetration 0.244 mm, and event matching 2/4. Conclude with `\partialstatus`: numerically stable smoke execution, final event gate failed.

- [ ] **Step 5: Update `sec:gaps`, Q9--Q15, and the oral conclusion**

Replace every statement that ten-step planning/contact is incomplete. The remaining primary replay limitation is natural liftoff timing/event matching; planning/contact must remain marked accepted.

### Task 5: Integrate FIR, AR, QP-MPC, C1, and C2 details

**Files:**
- Modify: `docs/overleaf/project_complete_learning_guide_zh.tex`
- Copy if absent: `results/figures/c1_qp2_forward_pitch_comparison.png` to `docs/overleaf/c1_qp2_forward_pitch_comparison.png`
- Copy if absent: `results/figures/fddp_two_step_natural_contact_c2.png` to `docs/overleaf/fddp_two_step_natural_contact_c2.png`

**Interfaces:**
- Consumes: DOCX controller tables, pasted FIR/AR/QP explanations, C1 tuning YAML, and natural-contact C2 YAML.
- Produces: one mathematically connected controller chapter with scope-labelled results.

- [ ] **Step 1: Correct the C0/C1/C2 relationship**

Use the equations:

```latex
\begin{aligned}
C0:&\quad q_{cmd}=q_{ref}(t),\\
C1:&\quad q_{cmd}=q_{ref}(t)+G\alpha,\\
C2:&\quad q_{cmd}=q_{ref}(t+\Delta t_{phase})+G\alpha.
\end{aligned}
```

State that these are incremental controller/ablation levels. Also state that the current two-step natural-C2 experiment did not simultaneously validate the complete C1+C2 expression.

- [ ] **Step 2: Add a dedicated AR subsection**

Define the local disturbance predictor as:

```latex
d_{k+1}=\sum_{i=1}^{n_a}a_i d_{k+1-i},
```

and explain that FIR predicts correction-induced response while AR predicts autonomous disturbance evolution. Keep CoM-shift and two-step identification protocols separate; do not merge their sample times, model orders, or delay values.

- [ ] **Step 3: Complete the QP-MPC explanation**

Retain the existing objective in Equation `eq:qp`, define `q`, `r`, and slew penalty, state the 20 ms controller tick and 12-step/240 ms horizon for the balanced CoM-shift C1 evidence, and explain first-move-only receding horizon execution.

- [ ] **Step 4: Add a scope-labelled C1 result table**

For the balanced CoM-shift evidence use:

```text
forward IAE 0.2710 -> 0.2474 deg*s, 8.7%, settling 0.248 -> 0.166 s, max correction 0.711 deg
backward IAE 0.2681 -> 0.2295 deg*s, 14.4%, settling 0.246 -> 0.166 s, max correction 0.669 deg
```

Keep the two-step pulse/ablation result in a separate paragraph and do not present it as the same protocol.

- [ ] **Step 5: Extend the contact observer and C2 limitation**

Describe toe/heel OR logic, 0.25 N threshold, 10 ms debounce, event ordering, 3/4 matching, one skipped event, and 47.375 ms maximum offset. Preserve the result that phase alignment did not guarantee improved pitch IAE.

### Task 6: Add robustness, Stage 8/9, runtime, and learning closure

**Files:**
- Modify: `docs/overleaf/project_complete_learning_guide_zh.tex`
- Copy if absent: `results/figures/fddp_two_step_robustness_success_rates.png` to `docs/overleaf/fddp_two_step_robustness_success_rates.png`
- Copy if absent: `results/figures/e6_controller_timing_distribution.png` to `docs/overleaf/e6_controller_timing_distribution.png`

**Interfaces:**
- Consumes: pasted-text unique sections, robustness summary, E6 summary, and current runtime boundary.
- Produces: evidence-disciplined robustness/runtime chapters, compact FAQ, and comprehension checks.

- [ ] **Step 1: Add the 70-trial robustness section**

Document 7 scenarios, 5 seeds, and C0/C2 modes. Include 5/5 results for nominal, friction 0.50, slope 3 deg, step 4 mm, command delay 12 ms, and sensor delay 20 ms. Highlight push +4 N as 0/5 in both modes and explain the Wilson lower bound of approximately 0.5655 for a 5/5 sample.

- [ ] **Step 2: Add a strict Stage 8/9 separation box**

Use a three-row table for double-support FDDP CoM shift, FDDP single/two/ten-step planning, and scripted Stage 8/9 gait. The third row must state that scripted locomotion validates infrastructure/robustness protocols but is not FDDP-generated continuous gait.

- [ ] **Step 3: Add compute-time and runtime boundaries**

State 50 Hz/20 ms, four runs of 61 updates, worst p95 1.082 ms, worst maximum 1.508 ms, and zero observed deadline misses. Immediately list excluded hardware latencies and missing complete limiter/fallback evidence. Describe runtime-loadability and command computation separately from servo communication, calibration, E-stop, fault handling, and verified scheduling.

- [ ] **Step 4: Add project-directory and complete-scenario subsections**

Add one compact directory-responsibility table and one numbered end-to-end scenario using:

```latex
q_{cmd}=q_{ref}(t_{phase})+G\alpha.
```

End the scenario by stating that simultaneous two-step C1 correction plus natural C2 retiming has not yet been validated in one complete branch.

- [ ] **Step 5: Convert the 20-question section to compact FAQ form**

Each answer must be at most three sentences plus one `\hyperref`. Remove duplicated tables, formulas, and result lists from the FAQ. Preserve all 20 question labels so existing cross-references do not break.

- [ ] **Step 6: Add a three-question comprehension check**

Use the source questions about `tau_ref` versus `q_cmd`, C1 versus C2, and why accepted multi-step planning is not autonomous continuous hardware walking.

### Task 7: Run deduplication and factual consistency audits

**Files:**
- Inspect: `docs/overleaf/project_complete_learning_guide_zh.tex`

**Interfaces:**
- Consumes: Tasks 2--6 merged guide.
- Produces: a source that is internally consistent before compilation.

- [ ] **Step 1: Scan for stale facts and forbidden claims**

Run:

```powershell
$p='\\wsl$\Ubuntu-22.04\home\zhijian\biped_fddp_project\docs\overleaf\project_complete_learning_guide_zh.tex'
Select-String -LiteralPath $p -Pattern `
  '75 个 violation','2\.165','-416\.6','final nominal.*没有','真实硬件自主行走已','tau_ref.*下发','60/60'
```

Expected: no stale value or forbidden positive claim remains. Historical references, if any, must be explicitly labelled as historical and not current status.

- [ ] **Step 2: Scan section labels and duplicate headings**

Run a script that extracts `\section`, `\subsection`, and `\label` values and reports duplicates. Expected: no duplicate labels and no repeated full-topic heading owned by two sections.

- [ ] **Step 3: Compare source-topic coverage against the design**

Check that the final guide contains searchable anchors for:

```text
backward pass
AR model
10-step
event match ratio
Wilson
Stage 8/9
runtime-loadability
SafetyLimiter/fallback boundary
理解检查
```

Expected: every topic is present once in its owning section or cross-referenced from the FAQ.

### Task 8: Compile and visually verify the LaTeX document

**Files:**
- Build: `docs/overleaf/project_complete_learning_guide_zh.tex`
- Inspect: `docs/overleaf/project_complete_learning_guide_zh.log`
- Inspect: `docs/overleaf/project_complete_learning_guide_zh.pdf`

**Interfaces:**
- Consumes: Task 7 audited source and copied figures.
- Produces: a compile-clean and visually reviewed PDF.

- [ ] **Step 1: Compile twice with the project latexmk configuration**

Run:

```bash
cd /home/zhijian/biped_fddp_project/docs/overleaf
latexmk -xelatex -interaction=nonstopmode -halt-on-error project_complete_learning_guide_zh.tex
latexmk -xelatex -interaction=nonstopmode -halt-on-error project_complete_learning_guide_zh.tex
```

Expected: exit code 0 and a non-empty PDF.

- [ ] **Step 2: Audit the log**

Run:

```bash
grep -nE 'Undefined control sequence|LaTeX Error|Citation.*undefined|Reference.*undefined|Overfull \\hbox' project_complete_learning_guide_zh.log
```

Expected: no errors, undefined citations/references, or unacceptable overfull boxes. Any small intentional underfull warning must be reviewed rather than ignored blindly.

- [ ] **Step 3: Render every PDF page to PNG**

Run:

```bash
mkdir -p /tmp/project_complete_learning_guide_zh_render
pdftoppm -png -r 120 project_complete_learning_guide_zh.pdf \
  /tmp/project_complete_learning_guide_zh_render/page
```

Expected: one PNG per PDF page.

- [ ] **Step 4: Inspect all pages**

Review every page at full size for Chinese glyphs, TOC wrapping, equation clipping, table overflow, figure/caption separation, orphan headings, and large blank regions. Fix the source and repeat Steps 1--4 after every layout-sensitive change.

### Task 9: Synchronise the verified guide to Overleaf and archive evidence

**Files:**
- Sync: `docs/overleaf/project_complete_learning_guide_zh.tex`
- Sync: only figures actually referenced by the guide
- Preserve: `docs/overleaf/v4.tex`
- Archive copy: `D:\毕业论文\机器人控制流程图\算法讲述文档\project_complete_learning_guide_zh.tex`

**Interfaces:**
- Consumes: Task 8 verified source/PDF.
- Produces: matching local, Overleaf, and D-drive source copies.

- [ ] **Step 1: Locate or create a clean authenticated Overleaf clone**

Use the known project ID `6a44c809e8122d234b0f12a5`. Do not print or embed the access token in logs, commands shown to the user, source files, or Git remote URLs. If no authenticated clone is available, stop before network mutation and request a secure authentication path.

- [ ] **Step 2: Copy only the verified guide and referenced figures**

Confirm `v4.tex` is unchanged by comparing its SHA-256 before and after the copy.

- [ ] **Step 3: Compile in the Overleaf clone before pushing**

Run the same two-pass `latexmk -xelatex` command in the clone. Expected: exit code 0.

- [ ] **Step 4: Commit and push the scoped change**

Commit message:

```text
docs: integrate complete FDDP and FIR-QP-MPC learning guide
```

Push only after confirming the diff contains the guide and required figures, with no credential or unrelated-file changes.

- [ ] **Step 5: Archive to D drive and verify hashes**

Copy the final `.tex` to the requested D-drive folder, compute SHA-256 for local, clone, and D-drive copies, and require all three hashes to match.

- [ ] **Step 6: Record final status**

Report the compile command, page count, warning audit, files changed, Overleaf push status, D-drive destination, and SHA-256. Do not claim success if the push or hash comparison did not complete.

