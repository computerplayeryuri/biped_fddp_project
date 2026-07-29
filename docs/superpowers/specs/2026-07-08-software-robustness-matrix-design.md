# Software Robustness Matrix Design

## Goal

Build a repeatable software-only robustness validation matrix for the current C2 contact-phase locomotion preview. The matrix extends the existing Stage 8 scripted alternating-step MuJoCo validation into longer and more varied stress tests while keeping the thesis claim boundary honest: this is not a full FDDP walking gait and not hardware validation.

## Scope

The new validation covers the software-side gaps that can be tested with the current model:

- Long-duration scripted contact-transition runs at 10 s, 30 s, and 60 s.
- Ground friction variants using MuJoCo geom friction updates.
- Slope-equivalent terrain loading using a rotated gravity vector in the planar model.
- Small-step contact tests using temporary MuJoCo XML variants with toe-region raised boxes.
- Forward/backward push sweeps using external forces on the electronics body.

The validation does not claim:

- A newly solved full FDDP walking trajectory.
- Real hardware walking.
- 3D lateral balance beyond the current planar MuJoCo model.

## Architecture

Create a new script, `mpc/13_run_scripted_robustness_matrix.py`, that reuses the existing C2 `ContactObserver` and `ContactPhaseSynchronizer` from `mpc/08_run_contact_phase_synchronised_fir_qp_mpc.py`. The script generates a periodic left/right scripted step reference, applies scenario-specific model perturbations, runs MuJoCo, and writes one CSV per scenario plus a combined YAML summary.

The script is separate from `mpc/12_validate_scripted_alternating_step_locomotion.py` so the original Stage 8 evidence remains unchanged and the stronger robustness matrix is clearly labelled as a new Stage 9 software validation.

## Data Flow

Scenario definition -> MuJoCo model setup -> periodic q reference -> actuator proxy -> optional push/terrain/friction/step perturbation -> C2 contact event matching -> row logging -> per-scenario metrics -> combined matrix summary.

## Acceptance Metrics

Each scenario reports:

- fall detected or not
- minimum torso height
- peak absolute torso pitch
- pitch RMS
- minimum contact count
- scheduled, observed, matched, and skipped contact events
- maximum absolute C2 phase offset
- forward displacement

A scenario passes if:

- no fall is detected
- minimum contact count is at least 1
- peak absolute pitch stays below 30 deg
- minimum torso height stays above 0.12 m
- at least 85 percent of scheduled contact events are matched
- max absolute phase offset stays below 90 ms

The combined matrix passes only if every scenario passes. If some stress cases fail, the YAML still gives useful evidence and identifies the software/hardware conditions that need redesign.

## Testing

Add `tests/test_robustness_matrix.py` before production implementation. Tests cover scenario generation, periodic event generation, temporary step XML generation, and summary pass/fail logic without requiring a full MuJoCo run.

## Claim Boundary

Use this wording in the thesis:

> A software robustness matrix was completed for the current scripted alternating-step C2 locomotion preview. The tested cases cover long-duration operation, friction changes, slope-equivalent loading, small step contact perturbations, and external push pulses. These results support stability of the current software stack under the tested simulation conditions, but do not replace a full FDDP walking-gait validation or hardware walking experiments.
