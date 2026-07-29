from __future__ import annotations

import sys
from pathlib import Path


def replace_once(text: str, old: str, new: str) -> str:
    count = text.count(old)
    if count < 1:
        raise RuntimeError(f"expected exactly one match, found {count}: {old[:120]!r}")
    return text.replace(old, new, 1)


def main() -> None:
    path = Path(sys.argv[1])
    text = path.read_text(encoding="utf-8")

    replacements = [
        (
            "The current evidence supports simulation-based recovery improvement, FDDP convergence logging, FIR sign-held-out prediction, contact-event phase synchronisation under software-injected timing mismatch, and measured QP timing.",
            "The current evidence supports double-support FDDP convergence, independently solved left- and right-swing FDDP steps, a converged two-step alternating reference, position-command replay, FIR sign-held-out prediction, contact-event phase retiming, and measured QP timing. A ten-step planning candidate is retained as negative evidence because its friction and CoP gates fail; it is not promoted to a final nominal trajectory.",
        ),
        (
            r"\item Develop a contact-consistent offline FDDP planning pipeline for double-support CoM-shift trajectories.",
            r"\item Develop a contact-consistent offline FDDP planning pipeline progressing from double-support CoM shifts to fixed-schedule single- and two-step contact transitions.",
        ),
        (
            r"\item Evaluate contact-event phase synchronisation and controller timing within the current simulation scope.",
            r"\item Evaluate contact-event phase synchronisation under both controlled timing mismatches and the available two-step natural-contact events, together with controller timing.",
        ),
        (
            "The work does not claim hardware validation, full walking-gait validation, or hard real-time execution.",
            "The FDDP evidence now includes double-support CoM shifts, independent DS--SS--DS swing steps, and a converged two-step alternating reference. It does not establish continuous multi-step FDDP walking: the ten-step candidate remains planning-only because contact-force/friction/CoP feasibility has not passed. The work does not claim hardware validation or hard real-time execution.",
        ),
        (
            "RQ1 & Can offline FDDP generate contact-consistent nominal double-support CoM-shift trajectories for the six-DoF planar biped model? & FDDP convergence records, derivative-consistency audit, contact-mode checks, joint-limit checks, CoM reference tracking, and exported trajectory metadata.",
            "RQ1 & How far can offline FDDP progress from double-support CoM shifts to fixed-schedule alternating contact trajectories for the six-DoF planar biped model? & FDDP convergence and gate records, derivative-consistency audit, DS--SS contact-mode checks, support/landing/clearance errors, joint limits, exported trajectory metadata, and the multi-step force/friction/CoP audit.",
        ),
        (
            "The thesis should use cautious claim language. The current evidence supports simulation-based recovery improvement, FDDP convergence logging, FIR sign-held-out prediction, contact-event phase synchronisation under software-injected timing mismatch, and measured QP timing. It does not support hardware validation, full walking-gait robustness, universal real-time performance, or experimentally identified actuator dynamics.",
            "The thesis should use cautious claim language. The current evidence supports double-support and two-step FDDP convergence, specified feasibility gates for independent left/right swing steps, MuJoCo position-command replay, FIR sign-held-out prediction, contact-event phase synchronisation under software-injected timing mismatch, natural-event phase retiming on the two-step reference, and measured QP timing. It does not support a final ten-step FDDP nominal, continuous walking-gait robustness, hardware validation, universal real-time performance, or experimentally identified actuator dynamics.",
        ),
        (
            "Chapter~\\ref{ch:experiments} reports the experimental validation and results, including the C0/C1 comparison, C2 phase-synchronisation evidence, and the Stage~9 software robustness matrix.",
            "Chapter~\\ref{ch:experiments} reports the experimental validation and results, including the incremental FDDP swing-step/multi-step evidence, C0/C1 comparison, C2 phase-synchronisation evidence, and the Stage~9 scripted robustness matrix.",
        ),
        (
            "FDDP is used as an offline nonlinear planner. Its role is to generate contact-consistent nominal trajectories for double-support CoM shifts. The online controller described later does not rerun full FDDP at high frequency.",
            "FDDP is used as an offline nonlinear planner. Its validated roles progress from double-support CoM shifts to independent DS--SS--DS swing steps and a two-step alternating reference. Periodic and 4/6/10-step continuation further demonstrate planning scalability, but the ten-step candidate is not a final nominal because its friction and CoP audit fails. The online controller described later does not rerun full FDDP at high frequency.",
        ),
        (
            r"The FDDP problem is formulated using a floating-base Pinocchio model, two 6D flat-foot contacts, double-support contact mode, CoM tracking, state tracking, nominal virtual torque tracking, and joint-limit checking. The solver is \texttt{crocoddyl.SolverFDDP}.",
            r"The FDDP problem is formulated using a floating-base Pinocchio model, phase-dependent 6D flat-foot contacts, CoM tracking, state/control regularisation, swing-foot SE(3) tracking, support/landing tracking, and joint-limit barriers. Double support activates both foot contacts; single support removes the swing-foot contact and retains the stance-foot contact. The solver is \texttt{crocoddyl.SolverFDDP}. Planning torque remains an analysis output rather than a position-actuator command.",
        ),
        (
            "FDDP nominal trajectories are contact-consistent for double-support CoM shifts & FDDP logs, contact status, derivative audit, exported NPZ metadata & Partially ready & Double-support CoM shift only",
            "FDDP planning progresses from double support through single/two-step contact switches & FDDP logs, support/landing/clearance gates, replay artifacts, and contact-feasibility report & Layered evidence & Single/two-step accepted; ten-step final nominal withheld after friction/CoP failure",
        ),
        (
            "Controller maintains support contact & Contact-count log and minimum contact count & Ready for current cases & Not walking or rough-terrain validation",
            "Controller maintains the declared contact minimum in tested cases & Contact-count and per-foot event logs & Ready for stated cases & Not continuous gait or rough-terrain validation",
        ),
        (
            "Double-support nominal CoM shift & Implemented and simulated & Main result \\\\",
            "\n".join(
                [
                    "Double-support nominal CoM shift & Implemented and simulated & E1 convergence result \\\\",
                    "Independent left/right FDDP swing steps & Physical gates and position-command replay passed & Not first-order convergence success \\\\",
                    "Two-step alternating FDDP reference & Converged, gated, replayed & Fixed two-step evidence; not continuous gait \\\\",
                    "Ten-step FDDP planning candidate & Planning gates passed; friction/CoP gate failed & Final nominal and replay withheld \\\\",
                ]
            ),
        ),
        (
            "Contact-event phase synchronisation C2 & Implemented in simulation & Event-layer timing mismatch evidence; not terrain dynamics",
            "Contact-event phase synchronisation C2 & Implemented in simulation & Injected mismatch evidence plus two-step natural-event retiming; no universal recovery gain",
        ),
        (
            "Walking-gait validation & Not completed & Explicit limitation",
            "Continuous multi-step FDDP replay & Not completed & Ten-step contact gate currently fails",
        ),
        (
            "Third, the C2 phase-synchronisation evidence covers software-injected deterministic timing mismatches, not the stochastic contact-timing variation of real terrain.",
            "Third, C2 has both deterministic event-injection evidence and a two-step natural-contact demonstration, but the latter matched three of four scheduled events and did not improve the common pitch-disturbance metric over C1; stochastic terrain timing remains unvalidated.",
        ),
        (
            "Fifth, sensor noise, IMU drift, encoder noise, timing jitter, and explicit runtime safety counters remain outside the current validated scope.",
            "Fifth, sensor noise, IMU drift, encoder noise, timing jitter, and explicit runtime safety counters remain outside the current validated scope. Sixth, the ten-step planning candidate fails the current friction-utilisation and CoP-margin gates, so there is no final continuous multi-step FDDP nominal or replay.",
        ),
        (
            r"\noindent\textbf{RQ1: Can offline FDDP generate contact-consistent nominal double-support CoM-shift trajectories for the six-DoF planar biped model?}",
            r"\noindent\textbf{RQ1: How far can offline FDDP progress from double-support CoM shifts to fixed-schedule alternating contact trajectories for the six-DoF planar biped model?}",
        ),
        (
            r"Yes. The E1 continuation sweep (Section~\ref{sec:detailed_evidence}) demonstrates that offline FDDP reliably converges for $\pm\SI{6}{\milli\metre}$ CoM shifts. All 24 continuation stages passed convergence and feasibility gates, the final shooting gap reached zero, and the exported trajectories maintain foot-contact constraints and joint limits throughout. The derivative-consistency audit confirmed that a damping value of $10^{-8}$ yields accurate dynamics derivatives ($F_x$ relative error $\approx 5\times10^{-6}$). This result is bounded to double-support CoM shifts and the current model; it does not cover single-support or walking trajectories.",
            r"FDDP now progresses through several evidence levels. The E1 continuation sweep reliably converges for $\pm\SI{6}{\milli\metre}$ double-support CoM shifts, with zero final shooting gap and audited derivatives. Independent left- and right-swing DS--SS--DS problems pass their specified physical feasibility gates and position-command replay integrity, although both exhaust the 500-iteration budget and therefore are not first-order convergence successes. The two-step alternating problem converges with solver stop $2.84\times10^{-8}$, zero gap, accepted support/landing/clearance gates, and a completed replay. Periodic and 4/6/10-step planning continuation pass their planning gates, but the ten-step candidate fails the subsequent friction-utilisation and CoP-margin audit. The correct RQ1 answer is therefore positive for double-support, individual swing steps, and a fixed two-step reference, but negative for a final continuous multi-step nominal.",
        ),
        (
            "The evidence includes FDDP iteration-level convergence logging, an FIR sign-held-out validation, C1/C2 contact-event timing comparison, QP timing statistics, C0/C1 recovery comparison, and a Stage~9 scripted robustness envelope spanning 20 scenarios across five stress families.",
            "The evidence includes double-support FDDP convergence, independent swing-step gates, a converged and replayed two-step reference, a rejected ten-step contact-feasibility candidate, FIR sign-held-out validation, C1/C2 contact-event timing comparisons, QP timing statistics, C0/C1 recovery comparisons, and a Stage~9 scripted robustness envelope. These evidence classes remain explicitly separated so that scripted stepping and planning-only multi-step candidates are not presented as continuous FDDP walking.",
        ),
        (
            r"\item \textbf{FDDP-generated continuous walking references.} The current work is limited to double-support CoM shifts and scripted alternating steps. Full walking gaits with single-support phases, swing-leg trajectories, and variable step timing would test the planner and controller under more realistic contact sequences.",
            r"\item \textbf{Close the multi-step FDDP contact-feasibility gap.} Single-support swing steps and a converged two-step alternating reference now exist, but the ten-step candidate fails friction-utilisation and CoP-margin gates. Force regularisation, friction/wrench-cone and CoP constraints, verified derivatives, and continuation should be added before any continuous multi-step replay is attempted.",
        ),
        (
            "FDDP generated contact-consistent nominal double-support CoM-shift trajectories for the current planar biped model. & FDDP convergence records, contact constraints, derivative audit, exported trajectory metadata. & Say current model and double-support CoM shifts; do not generalise to walking.",
            "\n".join(
                [
                    "FDDP generated converged double-support CoM shifts and a converged two-step alternating reference; independent swing steps passed specified physical gates. & E1 records, single/two-step diagnostics, replay artifacts, and exported metadata. & Distinguish first-order convergence from gate passing; do not generalise to continuous gait.",
                    "The ten-step FDDP planning candidate was rejected by the final contact gate. & Friction/CoP report, failed-node plot, and absence of the final nominal NPZ. & Present as negative evidence; do not claim multi-step replay.",
                ]
            ),
        ),
        (
            r"\item Do not claim that C1 works for walking, changing contact phases, or arbitrary disturbances.",
            r"\item Do not generalise the tested C1/two-step result to continuous gait, arbitrary contact phases, or arbitrary disturbances.",
        ),
    ]

    for old, new in replacements:
        text = replace_once(text, old, new)

    anchor = (
        "Researchers applying FDDP to small biped models should be aware that "
        "damping tuning and derivative auditing are necessary preconditions for "
        "reliable trajectory optimisation, not optional checks."
    )
    discussion = r"""

% ------------------------------------------------------------------
\section{Interpretation of Incremental Swing-Step and Multi-Step FDDP Evidence}
% ------------------------------------------------------------------

The new swing-step results extend the planner beyond double support, but the evidence is deliberately layered. The independent left- and right-swing problems achieve zero shooting gap and pass support, landing, clearance, joint-margin, and forward-CoM gates, yet both terminate at the 500-iteration budget with a first-order stopping criterion near $10^{-3}$. They demonstrate specified physical feasibility and replayability, not first-order convergence. The two-step alternating problem is stronger convergence evidence: it reaches a stopping criterion of $2.84\times10^{-8}$ and passes the declared trajectory gates before position-command replay.

The periodic-cycle and 4/6/10-step continuation results show that longer FDDP planning is numerically possible. However, the ten-step candidate fails a separate planar support-resultant audit at 75 of 1000 support nodes, with maximum friction utilisation 2.165 against a 0.90 gate and minimum CoP margin $-416.6$~mm against a required 2~mm margin. This negative result is important: zero shooting gap and accurate foot placement do not guarantee physically acceptable contact wrenches. Withholding the final nominal and its replay is therefore part of the validation method, not an unfinished reporting step."""
    text = replace_once(text, anchor, anchor + discussion)

    path.write_text(text, encoding="utf-8")
    print(f"patched {path} with {len(replacements) + 1} exact replacements")


if __name__ == "__main__":
    main()
