# MPC Comparison Diagrams Design

Date: 2026-07-16

## Objective

Create two concise, thesis-ready MPC comparison figures that reuse the visual language of the existing FDDP comparison while reducing information density.

Required deliverables for each figure:

- editable `.drawio` source;
- opaque-white PNG export;
- vector PDF export.

All process boxes use a white fill. Variables and formulas use Cambria Math with italic mathematical symbols. Normal node text uses 26 px.

## Figure 1: Standard MPC vs Project FIR-QP-MPC

### Layout

Two equal-width vertical columns. Each column contains five aligned boxes. Every connector is a single vertical segment from the bottom-centre anchor of one box to the top-centre anchor of the next box. No return arrow, elbow, crossing, waypoint, or automatic detour is permitted.

### Left column: Standard receding-horizon MPC

1. Measure state and read reference
2. Predict over horizon
3. Solve finite-horizon optimisation
4. Apply first optimal input
5. Plant evolves; repeat at next sample

Core formulas only:

- `X = F(x_k, U)`
- `U_k^* = arg min_U J(x_k,U)`
- `u_k = u_{0|k}^*`

### Right column: Project FIR-QP-MPC

1. FDDP reference plus measured pitch/contact
2. Phase-aligned pitch error: `t_phi = t + Delta t_phi`, `e_k = theta_k - theta_ref(t_phi)`
3. FIR plus AR prediction and box-constrained QP
4. First action and joint mapping: `alpha_k = A_0^*`, `Delta q_k = G alpha_k`
5. Position command: `q_cmd = q_ref(t_phi) + Delta q_k`

The QP box uses one compact formula:

`A^* = arg min_A q||E_free + SA||^2 + r||A||^2 + s||DA-d_prev||^2`

The figure contains one short footer: the optimisation repeats at each control sample; contact-event phase synchronisation changes reference time only when scheduled contact transitions exist.

## Figure 2: C0/C1/C2 Horizontal Comparison

### Layout

Three equal-size cards arranged left to right:

`C0  ->  C1  ->  C2`

Only two single-segment horizontal arrows are used. No internal card connectors and no feedback loop are used.

### C0 card

Title: `C0 — Reference only`

- Input: FDDP nominal reference
- Clock: fixed reference time
- Command: `q_cmd = q_ref(t)`
- Purpose: baseline replay under disturbance

### C1 card

Title: `C1 — FIR-QP pitch recovery`

- Adds measured pitch error
- FIR response plus AR free-response forecast
- Box/slew-constrained QP
- Command: `q_cmd = q_ref(t) + G alpha_k`
- Purpose: active pitch recovery at fixed phase

### C2 card

Title: `C2 — Contact-phase synchronisation`

- Retains the complete C1 controller
- Adds toe/heel contact-event observation
- Bounded event-based reference-time offset
- Command: `q_cmd = q_ref(t_phi) + G alpha_k`
- Purpose: phase-aware reference alignment

### Evidence boundary

A one-line note below C2 states:

`Current double-support reference has no scheduled contact transition; C2 monitoring is validated, but walking phase synchronisation is not yet claimed.`

This qualification is required because the current source code explicitly expects the phase layer to remain inactive for the double-support CoM-shift validation.

## Visual Style

- canvas: opaque white;
- process/card fill: white;
- standard MPC border: blue;
- project/C1 border: green;
- C0 border: neutral grey-blue;
- C2 border: purple;
- text: dark grey/black;
- restrained colour is used only for borders, headings, and arrows;
- no shadows, gradients, icons, or decorative containers;
- generous whitespace and concise noun phrases;
- longer PNG side at least 2200 px.

## Verification

Before delivery:

1. Parse both `.drawio` files as UTF-8 XML.
2. Confirm every process/card box has `fillColor=#FFFFFF`.
3. Confirm all connectors attach directly between adjacent centre anchors and contain no bend points.
4. Confirm required formulas and the C2 evidence-boundary sentence are present.
5. Export matching PNG and PDF files from the same sources.
6. Confirm PNG dimensions, opaque-white corner pixels, and PDF page renderability.
7. Copy all deliverables to `C:\Users\94277\Desktop\Codex_Overleaf_Deliverable` and verify hashes against the WSL originals.

