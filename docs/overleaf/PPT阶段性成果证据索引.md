# PPT阶段性成果证据索引

更新时间：2026-07-16

## 使用原则

- 每页优先放一张主图和三到四个结果数字，不要把YAML或代码截图直接放进PPT。
- YAML、CSV和NPZ是答辩追问时的原始证据；PNG/PDF是直接插入PPT的材料。
- 当前准确结论是：核心FDDP规划、MuJoCo重放及C0/C1/C2集成控制仿真已经形成闭环；Stage9鲁棒性矩阵仅14/20场景通过，真实硬件尚未验证。

## 第1页：项目目标与总体路线

建议标题：`基于FDDP与FIR-QP-MPC的低成本双足机器人控制`

主图：

- `fddp_standard_vs_project_flow.png`
- `mpc_standard_vs_project_expanded_zh.png`

支撑证据：

- `urdf_mjcf_consistency_audit.yaml`：模型映射审计，115项matched、0项unexpected mismatch；同时说明URDF/MJCF并非完全等价，部分参数仍是估计或代理。
- `biped_v1.urdf`、`biped_planar.xml`：规划模型和MuJoCo模型的原始文件。

页面结论：FDDP离线生成名义轨迹，MuJoCo做独立位置执行重放，C1/C2做在线修正；硬件侧命令是`q_cmd`，不是`tau_ref`。

## 第2页：FDDP名义轨迹规划

建议标题：`阶段成果一：10-step接触感知名义轨迹`

主图：

- `e1_fddp_cost_and_gap.png`
- `fddp_multistep_contact_feasibility.png`

支撑证据：

- `e1_fddp_convergence_summary.yaml`：FDDP convergence evidence。
- `fddp_multistep_contact_feasibility.yaml`：最终1000节点接触可行性，0个violation，最大摩擦利用率0.5807，最小CoP margin 2.391 mm。
- `fddp_multistep_10step_ref.npz`：最终10步、10秒名义轨迹文件。

页面结论：得到可供重放和控制器使用的10-step nominal，并通过最终contact feasibility gate。

注意：不要使用`fddp_multistep_planning_diagnostics.yaml`作为最终成功证据；它保留的是较早的4-step失败记录。

## 第3页：MuJoCo独立重放

建议标题：`阶段成果二：独立动力学环境完整重放`

主图：

- `fddp_multistep_nominal_replay.png`

支撑证据：

- `fddp_multistep_nominal_replay.yaml`：10秒完整重放、20/20接触事件匹配、无跌倒。

页面数字：

- maximum joint RMSE：1.124°；
- maximum torso pitch：9.479°；
- minimum height fraction：97.9%；
- contact events：20/20；
- unintended flight与sustained penetration均为0。

页面结论：名义轨迹不仅在FDDP中收敛，也能通过位置执行器代理在独立MuJoCo模型中完整重放。

## 第4页：C1 FIR-QP-MPC扰动恢复

建议标题：`阶段成果三：在线俯仰扰动恢复`

主图：

- `fddp_multistep_controller_ablation.png`
- `e4_fir_holdout_measured_vs_predicted.png`

方法图：

- `c1_fir_qp_pitch_recovery_detailed_zh.png`
- `c1_fir_qp_pitch_recovery_detailed_en.png`

支撑证据：

- `fddp_multistep_controller_ablation.yaml`：最终10-step同协议C0/C1/C2对比。
- `e4_fir_holdout_summary.yaml`：FIR模型独立holdout证据。

页面数字：C0 IAE 3.7588 deg·s，C1 IAE 3.3534 deg·s，改善10.79%；peak由1.990°降至1.822°。

页面结论：C1提高扰动后的恢复能力，不声称完全消除扰动施加瞬间的初始峰值。

## 第5页：C2接触相位同步

建议标题：`阶段成果四：接触事件驱动的参考相位修正`

主图：

- `fddp_multistep_controller_ablation.png`
- `c2_contact_switching_demo_contacts.png`
- `c2_contact_switching_demo_phase_offset.png`

方法图：

- `c2_contact_phase_sync_detailed_zh.png`
- `c2_contact_phase_sync_detailed_en.png`

支撑证据：

- `fddp_multistep_controller_ablation.yaml`：C2-final正式同协议结果。
- `ppt_kpi_c0_c1_c2_evaluation_summary.yaml`：PPT指标汇总。
- `e5_c1_c2_contact_mismatch_summary.yaml`：接触错位实验记录。

页面数字：C1平均接触时间误差56.6 ms，C2-final为35.0 ms，改善38.16%；20/20事件匹配；C2 IAE为3.3202 deg·s。

页面结论：C1改善俯仰恢复，C2进一步改善物理接触事件与参考步态的时间对齐。

## 第6页：进度、鲁棒性与剩余工作

建议标题：`当前完成度与下一阶段工作`

主图：

- `stage8_scripted_alternating_step_contacts.png`
- `stage9_scripted_robustness_matrix_pass_fail.png`
- `stage9_scripted_robustness_matrix_pitch.png`

支撑证据：

- `stage8_scripted_alternating_step_locomotion_summary.yaml`：2步scripted preview，4/4事件匹配、无跌倒。
- `stage9_scripted_robustness_matrix_summary.yaml`：20个场景中14个通过、6个失败，all_scenarios_passed=false。

页面结论：核心规划、重放和集成控制仿真已完成；scripted robustness仍需补强，真实硬件尚未验证。

## 组会收尾表述

`目前已经形成从接触感知FDDP规划、独立MuJoCo位置重放，到FIR-QP俯仰恢复和接触事件相位同步的核心仿真闭环。正式10-step replay和四模式控制器消融已经通过；但Stage9鲁棒性矩阵仍有6/20场景失败，真实硬件也尚未验证。因此下一阶段是先补齐强扰动、故障安全和重复统计，再进行硬件标定与实机实验。`

