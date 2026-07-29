# FDDP Two-Step Alternating Design

## Approved scope
Create the minimum alternating FDDP experiment with fixed 2.0 s contact schedule:
DS(20) -> right-stance/left-swing SS(70) -> DS landing(20) -> left-stance/right-swing SS(70) -> DS landing(20), at DT=0.010 s.
This is a two-step feasibility demonstration, not continuous walking and not a hardware claim.

## Architecture
A new independent solver owns the 200-node horizon, static x0 seed, phase-dependent ContactModel6D, SE(3) swing tracking for the currently swinging foot, active-support tracking, CoM tracking, state/control regularisation, joint safety barrier, diagnostics, and gated export.
The replay is a new independent MuJoCo position-command script. It reads only the gated NPZ and uses q_cmd through the established proxy. tau_ref remains planning and analysis data only.

## Acceptance
Use the established single-step numerical gates for shooting gap, support/landing pose errors, 10 mm clearance, joint margin, and forward CoM progress. Add a two-landing diagnostic and require each swing to satisfy clearance and landing gates. Export no nominal NPZ on failure.
Replay must pass the existing position-command integrity limits and separately log per-foot toe/heel contact evidence. The initial claim remains simulation validation only.

## Tests
Before solver code, tests must verify phase names/counts, contact side at both SS segments, two swing references, static x0 seed, and export-gate behaviour. Before replay code, a test must verify reference shapes, five phase labels, tau_ref analysis-only documentation, and independent output paths.
