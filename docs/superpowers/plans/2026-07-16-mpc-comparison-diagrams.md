# MPC Comparison Diagrams Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce two concise, editable, publication-ready MPC comparison figures: standard MPC versus the project FIR-QP-MPC, and a horizontal C0/C1/C2 capability comparison.

**Architecture:** Reuse the typography, colour palette, white-background convention, and formula formatting from the existing FDDP comparison source. Build two independent diagrams.net XML files, export each source to matching PNG and PDF outputs through the already-installed draw.io desktop CLI, then validate structure, pixels, PDF renderability, and desktop-copy hashes.

**Tech Stack:** diagrams.net `.drawio` XML, Windows draw.io desktop CLI, PowerShell XML/System.Drawing checks, bundled Poppler `pdfinfo`/`pdftoppm`.

## Global Constraints

- Create both `.drawio`, opaque-white PNG, and vector PDF for each figure.
- Use 26 px for all ordinary in-diagram text; titles may be larger.
- Use Cambria Math and italic styling for variables and formulas.
- Every process/card box must use `fillColor=#FFFFFF`.
- Every connector must be a single direct segment between adjacent centre anchors; no `<Array as="points">`, elbow, return path, crossing, or automatic detour.
- Figure 1 has two vertical five-box columns; Figure 2 has three horizontal cards.
- Preserve the C2 evidence boundary: the current double-support reference has no scheduled contact transition, so walking phase synchronisation is not claimed.
- Do not modify `docs/overleaf/fddp_standard_vs_project_flow.*`.
- Do not initialise or commit Git history; the current `master` branch has no commits.

---

### Task 1: Build Standard MPC vs Project FIR-QP-MPC Source

**Files:**
- Read: `docs/overleaf/fddp_standard_vs_project_flow.drawio`
- Create: `docs/overleaf/mpc_standard_vs_project_flow.drawio`

**Interfaces:**
- Consumes: the existing FDDP figure's XML style conventions and the approved MPC design spec.
- Produces: a UTF-8 diagrams.net XML document containing two five-node vertical flows.

- [ ] **Step 1: Create the editable XML source using existing styles**

Use `apply_patch` to add `docs/overleaf/mpc_standard_vs_project_flow.drawio`. Set the canvas to approximately `3000 x 1900`, use a white background, place the standard flow at the left and the project flow at the right, and reuse the FDDP diagram's blue/green border palette.

The left column must contain exactly these five semantic nodes:

```text
Measure state and read reference
Predict over horizon: X = F(x_k,U)
Solve: U_k^* = arg min_U J(x_k,U)
Apply: u_k = u_{0|k}^*
Plant evolves; repeat at next sample
```

The right column must contain exactly these five semantic nodes:

```text
FDDP reference + measured pitch/contact
t_phi = t + Delta t_phi; e_k = theta_k - theta_ref(t_phi)
FIR + AR prediction; A^* = arg min_A J_QP(A)
alpha_k = A_0^*; Delta q_k = G alpha_k
q_cmd = q_ref(t_phi) + Delta q_k
```

Each adjacent pair must use a direct vertical connector with `exitX=0.5;exitY=1;entryX=0.5;entryY=0` and a relative `mxGeometry` that contains no waypoint array.

- [ ] **Step 2: Parse and structurally test the source**

Run a PowerShell UTF-8 XML check that asserts:

```text
canvas background == #FFFFFF
10 process nodes exist
8 edges exist
all process nodes contain fillColor=#FFFFFF
all ordinary node styles contain fontSize=26
all edges contain edgeStyle=orthogonalEdgeStyle
no edge contains mxGeometry/Array
required formula tokens are present
```

Expected output: `FIGURE1_XML=PASS`.

### Task 2: Build C0/C1/C2 Horizontal Comparison Source

**Files:**
- Create: `docs/overleaf/c0_c1_c2_horizontal_comparison.drawio`

**Interfaces:**
- Consumes: approved C0/C1/C2 wording and the same style constants used by Task 1.
- Produces: a UTF-8 diagrams.net XML document with three equal-size horizontal cards and two direct arrows.

- [ ] **Step 1: Create three concise comparison cards**

Use `apply_patch` to add `docs/overleaf/c0_c1_c2_horizontal_comparison.drawio`. Set the canvas to approximately `3300 x 1150` and create:

```text
C0 — Reference only
FDDP nominal reference
Fixed reference time
q_cmd = q_ref(t)
Baseline replay

C1 — FIR-QP pitch recovery
Pitch-error feedback
FIR + AR + constrained QP
q_cmd = q_ref(t) + G alpha_k
Active recovery, fixed phase

C2 — Contact-phase synchronisation
C1 controller retained
Toe/heel event observation
q_cmd = q_ref(t_phi) + G alpha_k
Phase-aware reference alignment
```

Place the cards left to right and connect `C0 -> C1 -> C2` with two direct horizontal connectors using `exitX=1;exitY=0.5;entryX=0;entryY=0.5`. Do not add internal connectors or a capability arrow.

Add one evidence note below C2:

```text
Current double-support reference: no scheduled contact transition; monitoring is validated, but walking phase synchronisation is not yet claimed.
```

- [ ] **Step 2: Parse and structurally test the source**

Assert:

```text
canvas background == #FFFFFF
3 card nodes and 2 edges exist
all cards contain fillColor=#FFFFFF and fontSize=26
all edges are direct and contain no waypoint array
C0, C1, C2 command formulas are present
the full evidence-boundary sentence is present
```

Expected output: `FIGURE2_XML=PASS`.

### Task 3: Export Matching PNG and PDF Files

**Files:**
- Create: `docs/overleaf/mpc_standard_vs_project_flow.png`
- Create: `docs/overleaf/mpc_standard_vs_project_flow.pdf`
- Create: `docs/overleaf/c0_c1_c2_horizontal_comparison.png`
- Create: `docs/overleaf/c0_c1_c2_horizontal_comparison.pdf`

**Interfaces:**
- Consumes: the two validated `.drawio` sources from Tasks 1 and 2.
- Produces: raster and vector exports from the same final XML state.

- [ ] **Step 1: Stage sources in the existing Windows visualisation directory**

Copy both `.drawio` sources to:

```text
C:\Users\94277\.codex\visualizations\2026\07\16\019f6a22-fd3f-7450-a28c-db7d4e926864
```

This avoids the known Electron/WSL UNC boundary in the existing export helper.

- [ ] **Step 2: Export PNG files and wait for file creation**

Invoke the existing executable:

```text
C:\Users\94277\AppData\Local\Programs\draw.io\draw.io.exe
```

Use `--export --format png --scale 1 --output ...`. Poll for each output for at most 30 seconds because the Electron launcher may return before the file is written. Copy the completed PNG files back to `docs/overleaf/`.

Expected: both PNG files exist and each longer side is at least 2200 px.

- [ ] **Step 3: Export PDF files with the same polling rule**

Use `--export --format pdf --output ...`, wait for file creation, and copy both PDFs back to `docs/overleaf/`.

Expected: two non-empty PDF files.

### Task 4: Final Quality Gate and Desktop Delivery

**Files:**
- Verify: all six files under `docs/overleaf/`
- Copy: all six files to `C:\Users\94277\Desktop\Codex_Overleaf_Deliverable`
- Create: `C:\Users\94277\Desktop\Codex_Overleaf_Deliverable\mpc_comparison_diagrams_bundle.zip`

**Interfaces:**
- Consumes: all final sources and exports.
- Produces: verified desktop copies and a single ZIP bundle.

- [ ] **Step 1: Re-run complete XML and content checks**

Parse both sources fresh as UTF-8 XML. Verify white canvas, white boxes, exact node/edge counts, 26 px ordinary text, direct anchors, zero waypoint arrays, all formulas, and the C2 evidence sentence.

Expected: `XML_STATUS=PASS` with zero failures.

- [ ] **Step 2: Verify PNG pixels and dimensions**

Load both PNG files with `System.Drawing.Bitmap`. Check width/height, require the longer side to be at least 2200 px, and require all four corner pixels to be opaque white `(255,255,255,255)`.

Expected: `PNG_STATUS=PASS` for both figures.

- [ ] **Step 3: Verify PDF metadata and rendering**

Use bundled Poppler:

```text
C:\Users\94277\.cache\codex-runtimes\codex-primary-runtime\dependencies\native\poppler\Library\bin\pdfinfo.exe
C:\Users\94277\.cache\codex-runtimes\codex-primary-runtime\dependencies\native\poppler\Library\bin\pdftoppm.exe
```

Require one page per PDF and successful first-page PNG rendering.

Expected: `PDF_STATUS=PASS` for both figures.

- [ ] **Step 4: Copy, bundle, and compare hashes**

Copy all six deliverables to the established desktop folder, create `mpc_comparison_diagrams_bundle.zip`, and compare SHA-256 hashes between WSL originals and desktop copies.

Expected: six `HashMatch=True` rows and a ZIP containing exactly six entries.

