# Project Learning Guide Overleaf Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce a standalone Chinese Overleaf learning guide that answers all questions in the source DOCX and surgically update stale factual statements in `v4.tex`.

**Architecture:** Keep the learning guide independent from the thesis by giving it its own `ctexart` preamble and hyperlinked table of contents. Reuse current project evidence as the source of truth, while limiting `v4.tex` edits to claim-boundary corrections that distinguish validated single/two-step results from the rejected ten-step contact-feasibility candidate.

**Tech Stack:** LaTeX/ctex, hyperref, amsmath, booktabs, longtable, Git, PowerShell/WSL static validation.

## Global Constraints

- The learning guide must not be included by `v4.tex` or added as an appendix.
- Chinese explanations retain English technical terms.
- `q_cmd = q_ref + delta_q`; `tau_ref` is planning/analysis output only.
- Stage 8/9 remain scripted gait, not FDDP-generated continuous gait.
- The ten-step candidate is not a final nominal because contact feasibility fails.
- No hardware walking claim is permitted.
- The source workspace has no `.git`; only the Overleaf clone is committed.

---

### Task 1: Build the evidence ledger and question map

**Files:**
- Read: `D:\毕业论文\关于我的项目的问题.docx`
- Read: `/home/zhijian/biped_fddp_project/docs/2026-07-13_fddp_single_step_left_swing_experiment.md`
- Read: `/home/zhijian/biped_fddp_project/docs/2026-07-13_fddp_two_step_alternating_experiment.md`
- Read: `/home/zhijian/biped_fddp_project/docs/2026-07-14_fddp_multistep_simulation_closure_experiment.md`
- Read: `/home/zhijian/biped_fddp_project/docs/2026-07-13_urdf_mujoco_consistency_audit.md`

**Interfaces:**
- Consumes: Source questions, current code constants, current experiment metrics.
- Produces: A numbered 20-question map and a fact ledger used by Tasks 2 and 3.

- [x] **Step 1: Preserve the 20 source questions verbatim in working notes.**
- [x] **Step 2: Map each question to one or more detailed guide sections.**
- [x] **Step 3: Mark each numeric statement as code-derived, artifact-derived, or interpretation.**
- [x] **Step 4: Reject any statement that crosses the simulation/hardware or planning/final-nominal boundary.**

### Task 2: Create the standalone Chinese learning guide

**Files:**
- Create: `/tmp/overleaf_project_6a44c/project_complete_learning_guide_zh.tex`
- Read: `/tmp/overleaf_project_6a44c/sample.bib`
- Read: `/tmp/overleaf_project_6a44c/fddp_swing_step_evidence_update.tex`

**Interfaces:**
- Consumes: Task 1 question map and fact ledger.
- Produces: A self-contained LaTeX document with its own preamble, TOC, labels, cross-references, equations, tables, and references.

- [x] **Step 1: Add a `ctexart` preamble with `hyperref`, `amsmath`, `booktabs`, `longtable`, `enumitem`, `xcolor`, and safe page geometry.**
- [x] **Step 2: Write the end-to-end project explanation following the 16-section design outline.**
- [x] **Step 3: Add equations for state dimensions, shooting gap, quintic interpolation, command composition, FIR prediction, QP objective, and C2 phase-offset update.**
- [x] **Step 4: Add evidence tables that separate threshold, measured result, status, and claim boundary.**
- [x] **Step 5: Add all 20 questions verbatim with direct answers and hyperlinks to the detailed sections.**
- [x] **Step 6: Add a file-index table so the learner can trace each explanation to code or evidence.**

### Task 3: Correct stale statements in v4

**Files:**
- Modify: `/tmp/overleaf_project_6a44c/v4.tex`
- Read: `/tmp/overleaf_project_6a44c/fddp_swing_step_evidence_update.tex`

**Interfaces:**
- Consumes: Current validated evidence and rejected-candidate boundary.
- Produces: A minimally changed thesis draft whose abstract/scope/RQ/discussion/conclusion statements agree with current evidence.

- [x] **Step 1: Replace double-support-only scope statements with an evidence-layered description.**
- [x] **Step 2: Update RQ1 and the chapter roadmap to mention the independent single steps, two-step result, and planning-only multi-step candidate.**
- [x] **Step 3: Update traceability, limitations, conclusion, and future-work paragraphs without claiming continuous FDDP walking.**
- [x] **Step 4: Confirm the standalone learning guide is not referenced anywhere in `v4.tex`.**

### Task 4: Validate, publish, and archive

**Files:**
- Create: `/home/zhijian/biped_fddp_project/docs/2026-07-14_project_learning_guide_overleaf_sync.md`
- Modify remotely: Overleaf project `6a44c809e8122d234b0f12a5`, branch `main`

**Interfaces:**
- Consumes: Tasks 2 and 3 deliverables.
- Produces: Verified Overleaf commit plus project/D-drive archival record.

- [x] **Step 1: Run static LaTeX checks for brace balance, internal labels, asset references, and forbidden overclaims.**
- [x] **Step 2: Run `git diff --check` and inspect the complete diff.**
- [x] **Step 3: Compile locally if a LaTeX engine exists; otherwise record static-validation-only status.**
- [x] **Step 4: Commit and push the two Overleaf changes, then verify local HEAD equals `origin/main`.**
- [x] **Step 5: Record commands, files, commit hash, validation, and claim boundary in the project record.**
- [x] **Step 6: Run the existing D-drive archive sync and verify SHA-256 integrity.**

