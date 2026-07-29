from __future__ import annotations

import sys
from pathlib import Path


def replace_once(text: str, old: str, new: str) -> str:
    if text.count(old) != 1:
        raise RuntimeError(f"expected one match, found {text.count(old)}")
    return text.replace(old, new, 1)


def main() -> None:
    path = Path(sys.argv[1])
    text = path.read_text(encoding="utf-8")

    old_single = r"""\begin{figure}[htbp]
\centering
\includegraphics[width=0.48\linewidth]{fddp_single_step_left_swing_com_clearance.png}\hfill
\includegraphics[width=0.48\linewidth]{fddp_single_step_right_swing_com_clearance.png}
\caption{左右独立 FDDP 单步的 CoM 与 swing clearance。}
\end{figure}"""
    new_single = r"""\begin{figure}[htbp]
\centering
\IfFileExists{fddp_single_step_left_swing_com_clearance.png}{%
  \includegraphics[width=0.48\linewidth]{fddp_single_step_left_swing_com_clearance.png}%
}{\fbox{\parbox[c][3.6cm][c]{0.45\linewidth}{\centering 左脚单步图未随本文件提供}}}%
\hfill
\IfFileExists{fddp_single_step_right_swing_com_clearance.png}{%
  \includegraphics[width=0.48\linewidth]{fddp_single_step_right_swing_com_clearance.png}%
}{\fbox{\parbox[c][3.6cm][c]{0.45\linewidth}{\centering 右脚单步图未随本文件提供}}}
\caption{左右独立 FDDP 单步的 CoM 与 swing clearance。图片为可选资源，缺失时不影响本文档编译。}
\end{figure}"""
    text = replace_once(text, old_single, new_single)

    old_multi = r"""\begin{figure}[htbp]
\centering
\includegraphics[width=0.90\linewidth]{fddp_multistep_contact_feasibility.png}
\caption{十步 planning candidate 的 contact-feasibility negative evidence。失败结果阻止 candidate 晋升为 final nominal。}
\end{figure}"""
    new_multi = r"""\begin{figure}[htbp]
\centering
\IfFileExists{fddp_multistep_contact_feasibility.png}{%
  \includegraphics[width=0.90\linewidth]{fddp_multistep_contact_feasibility.png}%
}{\fbox{\parbox[c][4.2cm][c]{0.82\linewidth}{\centering 十步 contact-feasibility 图未随本文件提供；数值结论保留在正文。}}}
\caption{十步 planning candidate 的 contact-feasibility negative evidence。失败结果阻止 candidate 晋升为 final nominal。}
\end{figure}"""
    text = replace_once(text, old_multi, new_multi)

    old_bib = r"""\bibliographystyle{plain}
\bibliography{sample}"""
    new_bib = r"""\begin{thebibliography}{9}

\bibitem{jacobson1970differential}
David H. Jacobson and David Q. Mayne.
\newblock \emph{Differential Dynamic Programming}.
\newblock Elsevier, 1970.

\bibitem{mayne2000constrained}
David Q. Mayne, James B. Rawlings, Christopher V. Rao, and Pierre O. M. Scokaert.
\newblock Constrained Model Predictive Control: Stability and Optimality.
\newblock \emph{Automatica}, 36(6):789--814, 2000.

\bibitem{todorov2012mujoco}
Emanuel Todorov, Tom Erez, and Yuval Tassa.
\newblock MuJoCo: A Physics Engine for Model-Based Control.
\newblock In \emph{IEEE/RSJ IROS}, pages 5026--5033, 2012.

\bibitem{carpentier2019pinocchio}
Justin Carpentier, Florian Valenza, Nicolas Mansard, et al.
\newblock The Pinocchio C++ Library: A Fast and Flexible Implementation of Rigid Body Dynamics Algorithms and Their Analytical Derivatives.
\newblock In \emph{IEEE/SICE SII}, pages 614--619, 2019.

\bibitem{mastalli2020crocoddyl}
Carlos Mastalli, Rohan Budhiraja, Wolfgang Merkt, et al.
\newblock Crocoddyl: An Efficient and Versatile Framework for Multi-Contact Optimal Control.
\newblock In \emph{IEEE ICRA}, pages 2536--2542, 2020.

\bibitem{kuindersma2016optimization}
Scott Kuindersma, Robin Deits, Maurice Fallon, et al.
\newblock Optimization-Based Locomotion Planning, Estimation, and Control Design for the Atlas Humanoid Robot.
\newblock \emph{Autonomous Robots}, 40(3):429--455, 2016.

\bibitem{tassa2014control}
Yuval Tassa, Tom Erez, and Emanuel Todorov.
\newblock Control-Limited Differential Dynamic Programming.
\newblock In \emph{IEEE ICRA}, pages 1168--1175, 2014.

\end{thebibliography}"""
    text = replace_once(text, old_bib, new_bib)

    path.write_text(text, encoding="utf-8")
    print(f"made self-contained: {path}")


if __name__ == "__main__":
    main()
