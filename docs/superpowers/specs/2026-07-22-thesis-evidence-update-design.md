# Thesis Evidence Update Design

## Objective

Revise the cloned Overleaf thesis so that it accurately and professionally reports the current biped FDDP project. Preserve the existing chapter, section, and subsection structure. Rewrite content within that structure where necessary and add evidence figures without claiming unperformed hardware walking experiments.

## Narrative

The thesis will present one traceable control and validation chain:

1. offline FDDP nominal trajectory planning;
2. MuJoCo replay and contact-feasibility validation;
3. online reference-aligned FIR-QP-MPC pitch correction;
4. C0/C1/phase-only/C2 controller ablation;
5. scripted robustness, recovery, timing, and fault-injection evaluation;
6. hardware-facing position-command export and the remaining hardware-validation boundary.

Each technical section will answer: why the component is required, how the project implements it, how it is tested, what the recorded result is, and what conclusion the evidence supports.

## Claim Boundaries

- `tau_ref` is an offline planning and analysis output, not the hardware command used by the current runtime.
- The hardware-facing command is joint-position based, represented as `q_cmd = q_ref + delta_q`.
- Simulation completion and runtime loadability may be claimed when supported by logs.
- Physical walking, physical push recovery, and hardware robustness must remain future work until measured on the robot.
- Planar contact checks are acceptance evidence; out-of-plane checks remain diagnostic for this planar model.

## Method Content

### FDDP

Distinguish the standard FDDP inner optimisation loop from the project's outer trajectory-engineering workflow. Describe the 13-configuration/12-velocity state representation, six control inputs, fixed contact phases, periodic construction, seven-stage homotopy, 2-to-4-to-6-to-10-step continuation, and post-solution planning/contact acceptance gates.

### FIR-QP-MPC

Distinguish generic receding-horizon MPC from the implemented reference-aligned FIR-QP controller. Report the 20 ms control period, 12-step prediction horizon, 23-tap FIR memory, third-order autoregressive pitch model, projected-gradient box-QP solve, normal/recovery weights, correction limits, rate limits, and 30 ms command advance.

### Controllers

Explain C0 as nominal reference replay, C1 as FIR-QP pitch recovery, phase-only as contact-timing correction without pitch correction, and C2 as C1 plus contact-phase synchronisation. Avoid claiming C2 dominates C1 on every metric; report the observed trade-off between pitch-integral error and contact-timing error.

## Evidence and Figures

Reuse existing project-generated assets and copy only relevant figures into the Overleaf `figures/` directory. Prioritise:

- standard versus project-specific FDDP workflow;
- standard versus project-specific MPC workflow;
- C0, C1, and C2 controller workflows;
- accepted ten-step nominal replay;
- contact-feasibility audit;
- controller ablation pitch comparison;
- robustness matrix and recovery sweep;
- contact-phase error comparison;
- controller timing distribution;
- scripted fault-injection evidence where readable.

All captions will state the test context, principal numerical result, and evidence boundary instead of merely naming plotted variables.

## Results Hierarchy

The results chapter will separate:

1. planning convergence and contact feasibility;
2. nominal MuJoCo replay fidelity;
3. controller ablation and stability response;
4. contact-timing synchronisation;
5. robustness and recovery sweeps;
6. computational timing and deployment readiness;
7. remaining hardware experiments.

Current accepted evidence includes a 10 s/10-step reference, zero support-node contact violations across 1000 support nodes, 20/20 scheduled contact-event matches in nominal replay, controller-ablation metrics, 960/960 accepted scripted simulation trials across 32 scenarios and 30 seeds, and 15 accepted software fault-injection scenarios. Exact values will be re-read from current artefacts before insertion.

## Writing and Citation Style

- Preserve the existing English thesis language and heading structure.
- Prefer precise subject-verb statements and define every symbol at first use.
- Separate algorithm description from project-specific parameter choices.
- Cite primary papers and official documentation for algorithmic claims.
- Tie every quantitative claim to a table, figure, log-derived value, or referenced source.
- Use `simulation-only`, `software-in-the-loop`, or equivalent wording where required.

## Verification

After editing, compile the complete LaTeX project, resolve missing references and assets, inspect the log for undefined citations/references and overfull boxes, render the PDF, and visually review representative pages containing equations, tables, and figures. Confirm that headings and their ordering match the original thesis structure.
