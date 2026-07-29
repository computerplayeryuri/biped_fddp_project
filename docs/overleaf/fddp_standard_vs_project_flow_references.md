# FDDP 对比流程图：可复制参考文献与图注

## 建议的正文引用方式

一般 FDDP 流程可概括为：给定有限时域最优控制问题和可能不满足动力学连续性的初始轨迹，交替执行带缺口处理的 backward pass、forward rollout 与 line search，直至代价和可行性收敛 [1], [2]。其局部反馈控制律写为

\[
\delta u_k = k_k + K_k\,\delta x_k.
\]

DDP 的经典理论基础可追溯至 Jacobson 与 Mayne [4]；控制受限 DDP 的处理可参考 Tassa 等 [3]。本项目使用 Pinocchio 构建刚体动力学模型 [5]，通过 Crocoddyl/FDDP 生成名义轨迹 [1], [2]，再在 MuJoCo 中进行独立重放和控制验证 [6]。

## IEEE 风格参考文献（可直接复制）

[1] C. Mastalli, R. Budhiraja, W. Merkt, G. Saurel, B. Hammoud, M. Naveau, J. Carpentier, S. Vijayakumar, and N. Mansard, “Crocoddyl: An Efficient and Versatile Framework for Multi-Contact Optimal Control,” in *Proc. IEEE Int. Conf. Robotics and Automation (ICRA)*, 2020, pp. 2536–2542, doi: 10.1109/ICRA40945.2020.9196673.

[2] C. Mastalli, R. Budhiraja, W. Merkt, G. Saurel, B. Hammoud, M. Naveau, J. Carpentier, S. Vijayakumar, and N. Mansard, “A Feasibility-Driven Approach to Control-Limited DDP,” *Autonomous Robots*, vol. 46, pp. 985–1005, 2022, doi: 10.1007/s10514-022-10061-w.

[3] Y. Tassa, N. Mansard, and E. Todorov, “Control-Limited Differential Dynamic Programming,” in *Proc. IEEE Int. Conf. Robotics and Automation (ICRA)*, 2014, pp. 1168–1175, doi: 10.1109/ICRA.2014.6907001.

[4] D. H. Jacobson and D. Q. Mayne, *Differential Dynamic Programming*. New York, NY, USA: American Elsevier, 1970.

[5] J. Carpentier, G. Saurel, G. Buondonno, J. Mirabel, F. Lamiraux, O. Stasse, and N. Mansard, “The Pinocchio C++ Library—A Fast and Flexible Implementation of Rigid Body Dynamics Algorithms and Their Analytical Derivatives,” in *Proc. IEEE/SICE Int. Symp. System Integration (SII)*, 2019.

[6] E. Todorov, T. Erez, and Y. Tassa, “MuJoCo: A Physics Engine for Model-Based Control,” in *Proc. IEEE/RSJ Int. Conf. Intelligent Robots and Systems (IROS)*, 2012, pp. 5026–5033, doi: 10.1109/IROS.2012.6386109.

## 每篇文献在图中的作用

- [1] 支持 Crocoddyl 的多接触最优控制框架、DDP/FDDP 软件实现背景。
- [2] 是流程图左侧“标准 FDDP”的主要算法依据，尤其是允许 infeasible initialization、处理 shooting gaps 和逐步恢复可行性。
- [3] 支持控制约束与受限 DDP 的理论背景。
- [4] 支持经典 DDP 的基本递推框架。
- [5] 支持本项目右侧模型与动力学计算层使用 Pinocchio。
- [6] 支持本项目将 MuJoCo 作为 FDDP 之后的独立动力学仿真验证环境。

## 建议图注（中文）

**图 X  标准 FDDP 求解循环与本文双足机器人 FDDP 工程流程对比。** 左侧给出文献中的一般 FDDP 求解循环：从可能不可行的名义轨迹出发，计算代价与 shooting gaps，通过 backward pass 得到局部控制律 \(\delta u_k=k_k+K_k\delta x_k\)，并在 forward rollout 与 line search 中更新轨迹直至收敛。右侧给出本文实现：固定接触序列的 OCP、两步验证、周期步态七阶段 homotopy、2→4→6→10 步 continuation、规划门槛与接触可行性门槛，最终导出名义轨迹。虚线框中的 MuJoCo 重放和 C0/C1/C2 控制验证属于 FDDP 后的独立验证，不属于 FDDP 求解器本体。

## 建议图注（英文）

**Fig. X. Comparison between the standard FDDP solution loop and the project-specific FDDP engineering pipeline.** The standard loop starts from a possibly infeasible nominal trajectory, evaluates the cost and shooting gaps, computes the local policy \(\delta u_k=k_k+K_k\delta x_k\) in the backward pass, and updates the trajectory through forward rollout and line search. The project-specific pipeline adds a fixed contact schedule, a two-step validation problem, a seven-stage homotopy for the periodic gait cycle, 2→4→6→10-step continuation, and separate planning and contact-feasibility gates. MuJoCo replay and the C0/C1/C2 controller tests are post-FDDP validation stages rather than part of the FDDP solver.

## LaTeX 插图模板

```latex
\begin{figure*}[t]
  \centering
  \includegraphics[width=\textwidth]{fddp_standard_vs_project_flow.png}
  \caption{标准 FDDP 求解循环与本文双足机器人 FDDP 工程流程对比。左侧为一般求解器循环，右侧为本文从接触 OCP、homotopy 和 continuation 到双门槛验收及独立 MuJoCo 验证的流程。}
  \label{fig:fddp_standard_vs_project}
\end{figure*}
```

## 原始来源链接

- [Crocoddyl / ICRA 2020](https://doi.org/10.1109/ICRA40945.2020.9196673)
- [Feasibility-Driven DDP / Autonomous Robots 2022](https://doi.org/10.1007/s10514-022-10061-w)
- [Control-Limited DDP / ICRA 2014](https://doi.org/10.1109/ICRA.2014.6907001)
- [Pinocchio official repository and citation](https://github.com/stack-of-tasks/pinocchio)
- [MuJoCo / IROS 2012](https://doi.org/10.1109/IROS.2012.6386109)

