# Expanded Bilingual MPC Diagram Set Design

Date: 2026-07-16

## Objective

Replace the overly simplified MPC figures with eight separate, editable, thesis-ready diagrams. Chinese and English versions use identical geometry but independent text, so each version remains readable at normal PPT and A4 scale.

## Deliverable Set

Each base name produces `.drawio`, `.png`, and `.pdf` files.

1. `mpc_standard_vs_project_expanded_zh`
2. `mpc_standard_vs_project_expanded_en`
3. `c0_reference_only_detailed_zh`
4. `c0_reference_only_detailed_en`
5. `c1_fir_qp_pitch_recovery_detailed_zh`
6. `c1_fir_qp_pitch_recovery_detailed_en`
7. `c2_contact_phase_sync_detailed_zh`
8. `c2_contact_phase_sync_detailed_en`

The final desktop bundle therefore contains 24 diagram artifacts plus one ZIP archive.

## Shared Visual Rules

- Opaque white canvas and white process boxes.
- Ordinary diagram text uses 26 px; titles and section headers may be larger.
- Mathematical variables use Cambria Math with italic symbols and formatted subscripts/superscripts.
- Chinese and English versions preserve the same node positions, sizes, colours, and connectors.
- Use one dominant top-to-bottom reading direction.
- Every process arrow is a direct vertical connection between adjacent centre anchors.
- No return loop, automatic detour, elbow, crossing, or waypoint array.
- Colour is limited to borders and headings: blue for standard/reference-only, green for FIR-QP control, purple for contact-phase synchronisation, orange for acceptance/evidence checks.
- Each exported PNG has a longer side of at least 2200 px and opaque-white corner pixels.

## Figure Pair 1: Standard MPC vs Project MPC

### Layout

Two equal vertical columns with seven aligned process boxes per column. A short footer states that the optimisation is repeated at every sample and that the hardware-facing output is a position command.

### Standard MPC column

1. State measurement / estimation: `x_hat_k`
2. Reference and constraint update: `r_{k:k+N}`, `x in X`, `u in U`
3. Horizon prediction: `X = F(x_hat_k,U)`
4. Objective construction: tracking, control effort, terminal cost
5. QP/NLP solution: `U_k^* = arg min_U J(x_hat_k,U)`
6. First action: `u_k = u_{0|k}^*`
7. Plant evolution and next sampling instant: `x_{k+1}=f(x_k,u_k)`

### Project MPC column

1. Load offline FDDP nominal reference and C0-aligned pitch reference
2. Measure root pitch, pitch rate, and toe/heel contact signals
3. Contact-event phase alignment: `t_phi = clip(t + Delta t_phi)`
4. Reference-aligned pitch error and known-command separation
5. AR free-response forecast plus empirical FIR command response: `E = E_free + SA`
6. Box/slew-constrained QP and first correction: `alpha_k=A_0^*`, `Delta q_k=G alpha_k`
7. Position command and actuator proxy: `q_cmd=q_ref(t_phi)+Delta q_k`

The project column includes one compact QP expression:

`A^* = arg min_A q||E_free+SA||^2 + r||A||^2 + s||DA-d_prev||^2`

The footer states: `tau_ref` remains analysis-only; `q_cmd` is the runtime-facing command.

## Figure Pair 2: C0 Reference-Only Detailed Flow

### Process chain

1. Load FDDP full reference: `t_ref`, `q_ref`, `dq_ref`, phase/contact metadata
2. Use fixed reference clock: `t_ref_query=t`
3. Interpolate nominal joint and pitch references
4. No feedback correction: `Delta q_k=0`
5. Form position command: `q_cmd=q_ref(t)`
6. Position proxy and MuJoCo actuator application
7. Apply pitch disturbance and log baseline metrics

### Evidence statement

C0 is a reference-only disturbance baseline, not an MPC controller. It provides the comparison reference used to evaluate C1.

## Figure Pair 3: C1 FIR-QP Pitch-Recovery Detailed Flow

### Offline identification band

1. FDDP reference plus C0-aligned pitch reference
2. Empirical pulse-response identification: correction input `alpha` to root-pitch deviation, producing FIR coefficients `h`
3. Fit a stabilised AR model for natural disturbance evolution

### Online control chain

4. Measure pitch and form `e_k=theta_k-theta_ref(t)`
5. Subtract the known delayed response of previous commands
6. Predict `E_free` with the AR model and construct FIR dynamic matrix `S`
7. Solve the QP with tracking, magnitude, and slew penalties plus amplitude/slew bounds
8. Apply only the first solution element: `alpha_k=A_0^*`
9. Map to joint correction and position command: `Delta q_k=G alpha_k`, `q_cmd=q_ref(t)+Delta q_k`
10. Evaluate IAE, peak/final pitch error, contacts, correction magnitude, and QP solve time

### Evidence statement

C1 demonstrates improved disturbance recovery relative to C0; it does not claim elimination of the initial disturbance peak.

## Figure Pair 4: C2 Contact-Phase Synchronisation Detailed Flow

### Reference/event preparation

1. Retain the complete C1 FIR-QP controller
2. Load scheduled left/right contact transitions from the FDDP reference

### Online phase-alignment chain

3. Observe toe/heel contact signals and detect touchdown/liftoff events
4. Match observed and expected events within the configured time window
5. Compute bounded phase-offset update with gain, maximum step, and maximum absolute offset
6. Query phase-aligned time: `t_phi=clip(t+Delta t_phi,0,T)`
7. Interpolate `q_ref(t_phi)` and `theta_ref(t_phi)`
8. Run the unchanged C1 FIR-QP correction at the phase-aligned reference
9. Form runtime command: `q_cmd=q_ref(t_phi)+G alpha_k`
10. Log matched/skipped events, timing error, phase offset, contacts, pitch metrics, and QP time

### Evidence boundary

The current double-support CoM-shift reference contains no scheduled contact transition. The present C2 evidence therefore supports contact monitoring, event-capable implementation, and compatibility with C1, but not validated walking-gait phase synchronisation.

## Language Policy

- Chinese figures retain English technical terms where they are standard in the code or literature: FDDP, MPC, FIR, AR, QP, phase offset, position proxy.
- English figures use concise academic terminology and do not contain Chinese text.
- Formula symbols, variable names, units, and C0/C1/C2 labels are identical across both languages.

## Verification

For each of the eight `.drawio` sources:

1. Parse as UTF-8 XML.
2. Check exact expected process-node and edge counts.
3. Check all process boxes are white and ordinary text is 26 px.
4. Check every edge is direct, aligned between adjacent centres, and has no waypoint array.
5. Check required formulas and evidence statements.
6. Check English versions contain no CJK characters.
7. Export matching PNG and PDF files from the same source.
8. Verify PNG dimensions and opaque-white corners.
9. Verify each PDF is one page and renders successfully with Poppler.
10. Copy all 24 artifacts to `C:\Users\94277\Desktop\Codex_Overleaf_Deliverable`, create `mpc_bilingual_expanded_diagrams_bundle.zip`, and compare SHA-256 hashes with WSL originals.

