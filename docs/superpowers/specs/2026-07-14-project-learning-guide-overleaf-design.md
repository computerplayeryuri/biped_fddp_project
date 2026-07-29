# 项目完整学习文档与 Overleaf 更新设计

## 目标

基于 `D:\毕业论文\关于我的项目的问题.docx` 最后一段指示，制作一份可独立编译、可在 Overleaf 中直接查看和继续补写的中文项目学习文档；同时仅对 `v4.tex` 中已经被新实验事实淘汰的陈述做局部更新。

## 已确认的结构决定

- 学习文档是独立 `.tex`，不通过 `\\input`、Appendix 或其他方式接入 `v4.tex`。
- 学习文档面向“完全不了解项目”的读者，从机器人、模型、信号、FDDP、MuJoCo、C0/C1/C2 到 E1--E6 逐层解释。
- Word 中 20 个问题全部保留并逐题回答；每个问题链接到对应的详细讲解章节。
- 中文为主，保留 FDDP、shooting gap、holdout validation、contact phase synchronisation 等 English technical terms。
- `v4.tex` 只做事实更新，不重构论文、不把学习文档并入论文。

## 信息来源优先级

1. 当前 workspace 的 solver、controller 与 test 源码。
2. 当前 YAML/CSV/NPZ 实验产物与 2026-07-13/14 实验记录。
3. Overleaf 当前 `v4.tex`、`fddp_swing_step_evidence_update.tex` 与 `sample.bib`。
4. 旧的《FDDP 与 FIR-QP-MPC 算法讲述与参数说明》仅作为讲解结构参考；数值和结论以 1--3 为准。

## 独立学习文档结构

1. 阅读方法、证据状态和 claim boundary。
2. 项目目标、机器人结构、coordinate convention 与 position-command boundary。
3. `nq=13`、`nv=12`、free-flyer quaternion 和六个 pitch joints。
4. URDF 与 MJCF 的职责差异、为什么需要 MJCF、consistency audit 的流程和当前结果。
5. `q_ref`、`dq_ref`、`delta_q`、`q_cmd`、`tau_ref` 的单位、来源、消费者和禁止误读。
6. FDDP 从 optimal control problem 到 backward pass、forward rollout、line search、feasibility restoration 的完整流程。
7. node、multiple shooting、shooting gap、6D contact、DS/SS contact schedule、SE(3) swing tracking。
8. quintic progress、clearance、weights、acceptance gates、左右单步与两步结果，以及 solver-return/first-order convergence 的区别。
9. MuJoCo position-command replay、actuator proxy 和模型一致性边界。
10. C0、C1 fixed mapping、FIR identification、balanced candidate、QP-MPC、holdout validation。
11. C2 contact event、phase offset 更新公式、自然事件结果与不应声称的性能提升。
12. E1--E6 证据矩阵。
13. 当前完整软件流程图与数据流。
14. Word 原始 20 个问题的逐题答案与内部超链接。
15. 当前不足、multi-step contact-feasibility RED 结果和后续改进路线。
16. 可核对的项目文件索引与参考文献。

## `v4.tex` 更新边界

需要修正的旧表述包括：

- 把贡献和 scope 仍写成只有 double-support CoM shift 或没有任何 FDDP walking-contact transition evidence；
- RQ1、chapter roadmap、evidence traceability、discussion/conclusion 中把 FDDP evidence 限定为 double support；
- future work 仍把 single-support/contact-switching FDDP 当作完全未实现。

更新后的准确边界必须同时写明：

- 左/右独立 DS--SS--DS FDDP 单步已通过指定 physical gates 并完成 MuJoCo position-command replay；两者均未满足 500-iteration budget 内的 first-order convergence success，不能把 gate pass 写成 solver convergence success。
- 两步交替 FDDP 已通过轨迹 gates、完成 replay，并验证 natural-contact C2 retiming；C2 在该 pitch protocol 下没有优于 C1。
- periodic two-step cycle 和 4/6/10-step planning continuation 已通过相应 planning gates。
- 10-step candidate 在 unilateral force、friction utilisation 与 CoP gates 上失败，最终 nominal NPZ 被保留不导出；不得声称 continuous/multi-step FDDP walking replay。
- Stage 8/9 仍是 scripted gait，不是 FDDP-generated continuous gait。
- 全部结论仍限于 simulation validation + runtime-loadability；hardware walking 不在当前结论内。

## 验证

- 对独立 `.tex` 做 brace balance、引用文件存在性、禁止错误 claim 的静态扫描。
- 对 `v4.tex` 做定向 stale-phrase 扫描和 `git diff --check`。
- 若本机没有 LaTeX compiler，只声明 static validation；不声称本地 PDF compile 成功。
- Git push 后确认 local HEAD 与 `origin/main` 一致。
- 在项目 `docs/` 写同步记录，并同步到 `D:\毕业论文\机器人控制流程图` 的结果归档，核对 SHA-256。

