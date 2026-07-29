# Thesis Evidence Update Implementation Plan

**Goal:** Rewrite the existing Overleaf thesis content to accurately describe the current biped FDDP project while preserving its structure and adding verified project figures.

**Architecture:** Treat the local project repository as the evidence source and the cloned Overleaf repository as the publication target. Audit each existing thesis section against code, configuration, logs, tables, and figures; then update prose, equations, tables, captions, and references without adding, removing, or reordering headings.

**Tech Stack:** LaTeX, BibTeX, existing PNG/PDF figure assets, project Python/YAML/CSV/NPZ evidence, latexmk or available TeX engine, Poppler rendering for visual review.

---

### Task 1: Freeze the original structure and audit the current thesis

**Files:**
- Inspect: `docs/overleaf/thesis_6a57ac33/thesis.tex`
- Inspect: `docs/overleaf/thesis_6a57ac33/chapter1.tex` through `chapter8.tex`
- Inspect: `docs/overleaf/thesis_6a57ac33/appendixA.tex`
- Inspect: `docs/overleaf/thesis_6a57ac33/appendixB.tex`
- Inspect: `docs/overleaf/thesis_6a57ac33/references.bib`

1. Record all chapter/section/subsection headings in order.
2. Record every figure, table, equation label, citation, TODO, and unsupported result claim.
3. Build a section-to-evidence map without changing files.

### Task 2: Verify current project evidence

**Files:**
- Inspect: `transfer/trajectories/fddp_multistep_10step_ref.npz`
- Inspect: accepted planning/contact/replay/controller/robustness summaries under `results/`
- Inspect: relevant FDDP, replay, MPC, phase-sync, and hardware-runtime code/configuration.

1. Re-read exact numerical metrics from accepted artefacts.
2. Trace each important number to its generating code or result file.
3. Mark claims as demonstrated, supported but limited, diagnostic, or future work.

### Task 3: Prepare figures

**Files:**
- Copy/update: `docs/overleaf/thesis_6a57ac33/figures/`
- Source: current diagrams under `docs/overleaf/`
- Source: current plots under `results/figures/`

1. Select only figures that answer a thesis question.
2. Copy existing current figures rather than recreating them.
3. Use consistent filenames and verify resolution/readability.
4. Add LaTeX figure references and evidence-rich captions inside existing sections.

### Task 4: Rewrite methodology chapters within the existing headings

**Files:**
- Modify: relevant existing chapter files covering model, FDDP, MPC, and controllers.

1. Correct the model/state/control definitions.
2. Rewrite the FDDP algorithm-versus-engineering-workflow explanation.
3. Rewrite the FIR-QP-MPC formulation and all project-specific parameters.
4. Rewrite C0/C1/phase-only/C2 descriptions and command interfaces.
5. Preserve every existing heading and its order.

### Task 5: Rewrite experiments and results within the existing headings

**Files:**
- Modify: relevant existing chapter files covering validation, results, discussion, and conclusion.

1. Organise results by planning, replay, ablation, phase sync, robustness, timing, and deployment boundary.
2. Insert exact tables and values from current accepted artefacts.
3. Interpret stability, robustness, and agility conservatively.
4. State limitations and remaining hardware validation explicitly.

### Task 6: Update introduction, literature grounding, appendices, and bibliography

**Files:**
- Modify: relevant introduction/background/conclusion chapter files.
- Modify: `docs/overleaf/thesis_6a57ac33/appendixA.tex`
- Modify: `docs/overleaf/thesis_6a57ac33/appendixB.tex`
- Modify: `docs/overleaf/thesis_6a57ac33/references.bib`

1. Align the research questions and contributions with demonstrated work.
2. Use primary references for FDDP/DDP, MPC, contact modelling, and biped control.
3. Update appendices with reproducibility and runtime details where existing headings allow.
4. Remove or qualify obsolete unsupported claims.

### Task 7: Compile and validate

**Files:**
- Build: complete Overleaf clone.

1. Compile from a clean auxiliary-output state using the available TeX toolchain.
2. Check undefined citations, undefined references, missing figures, fatal warnings, and page-layout problems.
3. Render the PDF to images and inspect representative pages.
4. Compare the final heading sequence with the frozen original sequence.
5. Review the final Git diff for accidental structural changes and credential leakage.

### Task 8: Deliver and publish

**Files:**
- Finalise: Overleaf clone and compiled PDF.

1. Summarise changed chapters, key evidence, and known limitations.
2. Commit only intended thesis and figure changes.
3. Push to the configured Overleaf remote after successful validation.
