# Robustness Figure Redesign

## Communication job

By the end of slides 24 and 25, an unfamiliar robotics audience should understand not only that the declared simulations passed, but also which disturbances were tested, which direction or scenario was hardest, how close the results came to the predefined gates, and why the evidence remains simulation-only.

## Scope

Create two English, presentation-ready scientific figures from existing project results. Do not rerun the planner or generate new robustness claims. Preserve all current acceptance definitions and the simulation-only, position-command claim boundary.

Primary inputs:

- `results/logs/fddp_multistep_recovery_sweep.csv`
- `results/logs/fddp_multistep_recovery_selection.yaml`
- `results/logs/fddp_multistep_robustness_trials.csv`
- `results/logs/fddp_multistep_robustness_aggregate.csv`
- `results/logs/fddp_multistep_robustness_summary.yaml`

## Design references

The design follows common evidence patterns in legged-locomotion papers without copying their artwork:

- perturbation response and recovery metrics rather than an isolated success bar;
- direction or condition-dependent recovery envelopes;
- success rate accompanied by sample count and uncertainty;
- matrix or grouped views for multi-condition robustness evaluation.

Reference examples:

- Gu, Boyd, and Zhao, *Reactive Locomotion Decision-Making and Robust Motion Planning for Real-Time Perturbation Recovery*, especially the direction/phase-dependent recovery figures.
- Haarnoja et al., *Learning to Walk via Deep Reinforcement Learning*, especially the perturbation-recovery time-history presentation.
- Song et al., *Bipedal Stepping Controller Design Considering Model Uncertainty: A Data-Driven Perspective*, especially the success-rate, heat-map, and push-recovery figures.

## Figure 1: Bidirectional push recovery

Purpose: replace two identical 100% bars with evidence that exposes both repeated success and directional difficulty.

Layout:

1. Main panel: individual peak absolute pitch-error values for all 30 forward and 30 backward trials, with a compact box/range summary.
2. A horizontal gate at 15 degrees.
3. Direct annotations for `30/30 passed`, mean peak error, and remaining pitch margin.
4. A small supporting panel or aligned callout showing the fixed test conditions: `+/-4 N`, push start `0.8 s`, duration `0.12 s`, and `n=30` per direction.

Required interpretation:

- both directions have an observed success rate of 100%;
- backward recovery is substantially closer to the 15-degree pitch gate than forward recovery;
- the plot must not imply hardware recovery or unlimited push resistance.

## Figure 2: Full 32-scenario robustness profile

Purpose: retain every declared scenario while replacing the unreadable 32-bar chart with a grouped, statistically interpretable profile.

Layout:

1. Thirty-two scenario rows, grouped by category: nominal, push, friction, slope, step, command delay, sensor delay, actuator gain, actuator time constant, noise, and combined cases.
2. Left quantitative panel: peak absolute pitch error across 30 seeds for each scenario, shown as a central marker plus observed range, with a vertical 15-degree gate.
3. Right statistical panel: observed success rate plus 95% Wilson interval, with 90% per-critical-scenario and 95% aggregate acceptance references.
4. Critical boundary scenarios use an Okabe-Ito orange encoding; intermediate scenarios use blue-grey. Shape or marker fill provides redundant encoding for greyscale readability.
5. Summary annotation: `32 scenarios x 30 seeds = 960 trials; observed aggregate success 960/960`.

Required interpretation:

- every scenario remains individually traceable;
- success-rate uncertainty is visible even when all point estimates equal 100%;
- the Wilson lower bound for 30/30 is approximately 88.65%, so the figure and narration must not claim that every per-scenario confidence interval exceeds 90%;
- the acceptance decision is based on the declared observed-rate gates;
- the result supports robustness only inside the declared MuJoCo simulation envelope.

## Styling and deliverables

- White background and slide-readable typography.
- Okabe-Ito colourblind-safe palette with greyscale-redundant markers.
- Minimal grid lines; no gradients, shadows, or 3D effects.
- Explicit units, sample sizes, thresholds, panel labels, and uncertainty definition.
- Export each figure as high-resolution PNG and vector PDF.
- Keep the plotting source in `reporting/` and build from existing CSV/YAML evidence.
- Provide concise Chinese explanation plus English slide wording and speaker notes.

## Verification

Before delivery:

1. Recompute every displayed statistic from the source CSV files.
2. Confirm scenario count is 32 and total trial count is 960.
3. Confirm push-recovery counts are 30/30 in each direction.
4. Confirm all thresholds and test conditions match the YAML configuration.
5. Inspect both PNGs at full size and at approximate PowerPoint size.
6. Check that labels do not overlap and remain readable in greyscale.
7. Confirm PDF and PNG outputs are generated from the same figure source.

## Explicit exclusions

- No hardware claims.
- No invented baselines or unexecuted controller comparisons.
- No additional simulation sweep unless separately requested.
- No modification of the PowerPoint file in this task; the figures are delivered for insertion, with slide wording and narration guidance.
