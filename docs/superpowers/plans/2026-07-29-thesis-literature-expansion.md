# Thesis Literature Expansion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add concise, traceable literature support to the existing bilingual thesis without changing its structure or overstating the project evidence.

**Architecture:** `references.bib` remains the single bibliography source. English and Chinese chapter files receive matched nearby citations, while methodological and experimental results remain project evidence rather than literature-derived claims.

**Tech Stack:** LaTeX, BibTeX, shell-based static citation audit, Git/Overleaf.

## Global Constraints

- Preserve the current chapter and section structure.
- Use primary peer-reviewed sources or official manufacturer documentation.
- Add citations immediately after the supported claim.
- Preserve the simulation-only and position-command claim boundary.
- Do not alter recorded numerical results.

---

### Task 1: Add verified bibliography metadata

**Files:**
- Modify: `docs/overleaf/thesis_6a57ac33/references.bib`

- [ ] Add entries for Gauss--Newton multiple shooting, FIR-based constrained MPC,
  contact-event phase reset, contact-wrench feasibility, ST3215 official
  documentation, and the Wilson score interval.
- [ ] Check keys and DOI/URL metadata for duplicates and transcription errors.

### Task 2: Strengthen the bilingual literature review

**Files:**
- Modify: `docs/overleaf/thesis_6a57ac33/chapters/chapter2.tex`
- Modify: `docs/overleaf/thesis_6a57ac33/chapters_zh_full/chapter2.tex`

- [ ] Add matched citations and concise explanatory sentences at the existing
  FDDP, FIR-QP-MPC, phase-reset, and validation discussions.
- [ ] Keep the distinction between published precedent and the project's
  identified coefficients, joint map, and acceptance gates.

### Task 3: Cite hardware, contact, and statistics claims

**Files:**
- Modify: `docs/overleaf/thesis_6a57ac33/chapters/chapter3.tex`
- Modify: `docs/overleaf/thesis_6a57ac33/chapters_zh_full/chapter3.tex`
- Modify: `docs/overleaf/thesis_6a57ac33/chapters/chapter4.tex`
- Modify: `docs/overleaf/thesis_6a57ac33/chapters_zh_full/chapter4.tex`
- Modify: `docs/overleaf/thesis_6a57ac33/chapters/chapter6.tex`
- Modify: `docs/overleaf/thesis_6a57ac33/chapters_zh_full/chapter6.tex`

- [ ] Cite the ST3215 official specification at the hardware-interface claim.
- [ ] Cite contact-wrench theory at the planar contact-feasibility definition.
- [ ] Cite Wilson's paper at the interpretation of the 30/30 lower bound.

### Task 4: Verify and publish

**Files:**
- Inspect all modified `.tex` and `.bib` files.

- [ ] Run the thesis citation/label/graphics static audit.
- [ ] Run the existing project tests.
- [ ] Review the exact Git diff for scope and claim-boundary preservation.
- [ ] Commit the thesis changes and push `main` to the configured Overleaf remote.

