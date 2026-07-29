# Expanded Bilingual MPC Diagram Set Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create eight independent, editable, bilingual MPC figures and their matching PNG/PDF exports, with enough algorithmic detail for thesis writing and group-meeting presentation.

**Architecture:** Reuse the existing white-box diagrams.net visual language and direct-connector geometry. Build one Chinese and one English source for each of four figure roles, keep each language pair geometrically identical, export all sources through the installed draw.io desktop CLI, and validate XML, language, pixels, PDF rendering, and desktop-copy hashes.

**Tech Stack:** diagrams.net `.drawio` XML, existing FDDP/MPC figure styles, PowerShell XML/System.Drawing validation, draw.io desktop CLI, bundled Poppler and pypdf.

## Global Constraints

- Create exactly eight `.drawio`, eight PNG, and eight PDF files under `docs/overleaf/`.
- Use the eight base names specified in `docs/superpowers/specs/2026-07-16-mpc-bilingual-expanded-diagrams-design.md`.
- Use opaque white canvas and `fillColor=#FFFFFF` for every process box.
- Use 26 px for ordinary text; titles and section headers may be larger.
- Use Cambria Math with italic symbols for equations and variable names.
- Chinese and English files in each pair must have identical page size, node geometry, edge geometry, colour, and style.
- Every process connector is a single direct vertical centre-to-centre segment with no `<Array as="points">`.
- English sources must contain no CJK characters.
- Preserve the exact C1 and C2 evidence boundaries from the design spec.
- Reuse existing styles and draw.io CLI paths; do not add a new diagram-generation program or dependency.
- Do not modify or overwrite the earlier simplified MPC files.
- Do not initialise or commit Git history; `master` is still unborn.

---

### Task 1: Expanded Standard MPC vs Project MPC Pair

**Files:**
- Read: `docs/overleaf/mpc_standard_vs_project_flow.drawio`
- Create: `docs/overleaf/mpc_standard_vs_project_expanded_zh.drawio`
- Create: `docs/overleaf/mpc_standard_vs_project_expanded_en.drawio`

**Interfaces:**
- Consumes: existing two-column MPC styles and the seven-stage bilingual content in the design spec.
- Produces: two same-geometry diagrams with 14 process nodes and 12 direct edges each.

- [ ] **Step 1: Build the Chinese source**

Use the existing simplified two-column diagram as the style reference. Create seven aligned boxes in each column on an approximately `3300 x 2600` white canvas. Include the complete standard MPC chain and project chain from the design spec, including `E=E_free+SA`, the compact QP objective, `alpha_k=A_0^*`, `Delta q_k=G alpha_k`, and `q_cmd=q_ref(t_phi)+Delta q_k`.

- [ ] **Step 2: Build the English source with identical geometry**

Duplicate the structural XML and replace only titles, node values, and footer text with concise English. Preserve every `mxGeometry`, edge source/target, style string, page dimension, and colour.

- [ ] **Step 3: Validate the pair**

Parse both files as UTF-8 XML and assert:

```text
14 process nodes per file
12 edges per file
zero edge waypoint arrays
all source/target x-centres equal for every edge
all process boxes white and 26 px
identical geometry signatures between zh and en
English CJK character count = 0
required formulas and tau_ref/q_cmd boundary present
```

Expected: `PAIR1_XML=PASS`.

### Task 2: C0 Reference-Only Detailed Pair

**Files:**
- Create: `docs/overleaf/c0_reference_only_detailed_zh.drawio`
- Create: `docs/overleaf/c0_reference_only_detailed_en.drawio`

**Interfaces:**
- Consumes: blue reference-only style and the seven-stage C0 chain.
- Produces: two same-geometry diagrams with seven process nodes, six direct edges, and one evidence note each.

- [ ] **Step 1: Build the Chinese C0 source**

Create a vertical process figure on an approximately `2400 x 2500` white canvas. Include FDDP full-reference load, fixed clock, interpolation, `Delta q=0`, `q_cmd=q_ref(t)`, position proxy/MuJoCo application, and baseline metric logging.

- [ ] **Step 2: Build the English C0 source**

Translate node and evidence text only. Preserve geometry and styles exactly. The evidence note must say that C0 is a reference-only disturbance baseline, not an MPC controller.

- [ ] **Step 3: Validate the pair**

Require seven nodes, six direct vertical edges, white boxes, 26 px ordinary text, same geometry signature, no CJK in English, the `Delta q=0` and `q_cmd=q_ref(t)` formulas, and the baseline evidence note.

Expected: `PAIR2_XML=PASS`.

### Task 3: C1 FIR-QP Pitch-Recovery Detailed Pair

**Files:**
- Create: `docs/overleaf/c1_fir_qp_pitch_recovery_detailed_zh.drawio`
- Create: `docs/overleaf/c1_fir_qp_pitch_recovery_detailed_en.drawio`

**Interfaces:**
- Consumes: green controller style and the ten-stage C1 chain.
- Produces: two same-geometry diagrams with ten process nodes, nine direct edges, offline/online section labels, and one evidence note each.

- [ ] **Step 1: Build the Chinese C1 source**

Create a vertical diagram on an approximately `2600 x 3400` canvas. Use a three-node offline-identification band followed by a seven-node online-control chain. Include FIR coefficients `h`, stabilised AR model, pitch error, known delayed response subtraction, `E_free+SA`, the full q/r/slew QP objective and bounds, `alpha_k=A_0^*`, `Delta q_k=G alpha_k`, `q_cmd=q_ref(t)+Delta q_k`, and acceptance metrics.

- [ ] **Step 2: Build the English C1 source**

Translate text only while keeping code variable names and geometry unchanged. Preserve the claim: C1 improves disturbance recovery relative to C0 but does not eliminate the initial disturbance peak.

- [ ] **Step 3: Validate the pair**

Require ten nodes, nine direct vertical edges, same geometry, no English CJK, all formulas, q/r/slew penalties and amplitude/slew bounds, the five metric groups, and the C1 evidence statement.

Expected: `PAIR3_XML=PASS`.

### Task 4: C2 Contact-Phase Synchronisation Detailed Pair

**Files:**
- Create: `docs/overleaf/c2_contact_phase_sync_detailed_zh.drawio`
- Create: `docs/overleaf/c2_contact_phase_sync_detailed_en.drawio`

**Interfaces:**
- Consumes: purple phase-synchronisation style and the ten-stage C2 chain.
- Produces: two same-geometry diagrams with ten process nodes, nine direct edges, preparation/online section labels, and one evidence-boundary note each.

- [ ] **Step 1: Build the Chinese C2 source**

Create a vertical diagram on an approximately `2600 x 3400` canvas. Include retained C1 controller, scheduled contact transitions, toe/heel event detection, event matching window, bounded phase-offset update, `t_phi=clip(t+Delta t_phi,0,T)`, phase-aligned interpolation, unchanged C1 correction, `q_cmd=q_ref(t_phi)+G alpha_k`, and event/phase/pitch/QP logging.

- [ ] **Step 2: Build the English C2 source**

Translate text only and preserve identical geometry. Include the full current-evidence boundary: double-support CoM-shift has no scheduled contact transition; monitoring and event-capable compatibility are supported, walking-gait phase synchronisation is not validated.

- [ ] **Step 3: Validate the pair**

Require ten nodes, nine direct vertical edges, same geometry, no English CJK, all event and phase formulas, all logging categories, and the complete evidence boundary.

Expected: `PAIR4_XML=PASS`.

### Task 5: Export and Verify All PNG/PDF Files

**Files:**
- Create: matching `.png` and `.pdf` for all eight base names under `docs/overleaf/`.

**Interfaces:**
- Consumes: eight validated `.drawio` sources.
- Produces: 16 matching render artifacts.

- [ ] **Step 1: Stage sources in the existing Windows visualisation directory**

Copy the eight sources to:

```text
C:\Users\94277\.codex\visualizations\2026\07\16\019f6a22-fd3f-7450-a28c-db7d4e926864
```

- [ ] **Step 2: Export PNG and PDF with the existing draw.io executable**

Use:

```text
C:\Users\94277\AppData\Local\Programs\draw.io\draw.io.exe
```

For each source, invoke `--export --format png --scale 1` and `--export --format pdf`. Poll each output for up to 30 seconds because the Electron launcher may return before file creation. Copy outputs back to `docs/overleaf/`.

- [ ] **Step 3: Verify all rendered artifacts**

For each PNG, require a longer side of at least 2200 px and four opaque-white corner pixels. For each PDF, require one page, successful first-page Poppler rendering, and non-empty text extraction through bundled pypdf.

Expected: `PNG_STATUS=PASS (8/8)` and `PDF_STATUS=PASS (8/8)`.

### Task 6: Final Cross-Language and Desktop Delivery Gate

**Files:**
- Verify: 24 artifacts under `docs/overleaf/`.
- Copy: 24 artifacts to `C:\Users\94277\Desktop\Codex_Overleaf_Deliverable`.
- Create: `C:\Users\94277\Desktop\Codex_Overleaf_Deliverable\mpc_bilingual_expanded_diagrams_bundle.zip`.

**Interfaces:**
- Consumes: the complete eight-figure artifact set.
- Produces: verified desktop copies and one ZIP containing exactly 24 entries.

- [ ] **Step 1: Run fresh XML and pair-consistency checks**

Reparse all eight sources and report node/edge counts, zero-waypoint counts, white-box counts, formula/evidence-token coverage, English CJK counts, and four geometry-pair matches.

- [ ] **Step 2: Copy and bundle**

Copy all artifacts to the established desktop directory and create the ZIP without modifying earlier FDDP or simplified MPC deliverables.

- [ ] **Step 3: Compare hashes and ZIP contents**

Compare SHA-256 hashes between every WSL artifact and desktop copy. Open the ZIP and require exactly 24 entries.

Expected:

```text
XML_STATUS=PASS (8/8)
PAIR_GEOMETRY=PASS (4/4)
DESKTOP_HASHES=PASS (24/24)
ZIP_ENTRIES=24
```

