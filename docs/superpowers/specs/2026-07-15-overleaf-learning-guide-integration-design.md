# Overleaf 完整学习指南去重整合设计

日期：2026-07-15  
目标文件：`docs/overleaf/project_complete_learning_guide_zh.tex`

## 1. 目标

将以下两个来源的有效内容整理进现有中文学习指南：

1. `FDDP与FIR_QP_MPC算法讲述与参数说明.docx`；
2. 用户提供的完整项目讲解 pasted text。

最终文档应面向希望从入门逐步达到专业理解的读者，保持一条连贯主线，不按来源文档分章，不重复解释同一概念，并以当前项目日志作为事实依据。

## 2. 非目标

- 不把学习指南改写成论文正文或论文附录。
- 不修改 `v4.tex`。
- 不声称真实硬件行走、hardware real-time validation 或 fully robust continuous walking。
- 不把 Stage 8/9 scripted gait 写成 FDDP-generated gait。
- 不把 `tau_ref` 写成 hardware torque command。
- 不保留来源材料中已被当前项目结果取代的旧状态。

## 3. 内容架构

### 3.1 导读与项目边界

- 一句话说明项目；
- 直观角色类比；
- planner、simulator、controller、runtime 的职责边界；
- 当前可以和不可以声称的结论。

### 3.2 模型、坐标与接口

- 6-DoF planar biped 与坐标约定；
- `nq=13, nv=12`；
- URDF/Pinocchio/Crocoddyl 与 MJCF/MuJoCo；
- consistency audit；
- `q_ref`、`dq_ref`、`delta_q`、`q_cmd`、`tau_ref` 信号字典；
- position-command hardware boundary。

### 3.3 FDDP 原理与本项目实现

- finite-horizon optimal control；
- multiple shooting、node、shooting gap；
- backward pass、forward rollout、line search 和 regularisation；
- fixed DS--SS--DS schedule；
- phase-dependent 6D contacts；
- swing-foot SE(3) residual；
- quintic swing reference；
- cost weights、scaling 和 continuation；
- optimisation 内部 residual/bounds 与 post-solve gates 的区别。

### 3.4 FDDP 证据阶梯

- double-support CoM shift；
- left/right single step；
- alternating two-step；
- periodic seed；
- accepted 10-step planning/contact result；
- 10-step-reference MuJoCo replay event-gate limitation。

正式 10-step 事实使用：10 s、10 steps、66 iterations、stop `4.410209e-8`、gap 0、forward CoM 50.087 mm、0/1300 contact violations、max friction utilisation 0.5807、min CoP margin 2.391 mm。明确 planning/contact accepted 不等于 full replay accepted。

### 3.5 MuJoCo replay

- position-actuator proxy、reference interpolation 和模型 mismatch；
- `tau_ref` 不进入 `data.ctrl`；
- single/two-step replay 指标；
- global contact count 与逐脚 natural events 的区别；
- 10-step smoke replay 2/4 event matching 和 acceptance false。

### 3.6 C0、C1、C2 与 FIR-QP-MPC

- C0/C1/C2 是 incremental layers，不是独立算法集合；
- C1 fixed mapping `delta_q=G alpha`；
- FIR delayed response；
- AR disturbance evolution；
- QP objective、constraints、receding horizon；
- C1 CoM-shift evidence 与 two-step evidence 分开；
- C2 contact observer、debounce、event matching、bounded phase offset；
- 当前 two-step C1 correction 与 natural C2 retiming 尚未在同一 branch 完整验证。

### 3.7 Robustness、timing 与 runtime

- 70-trial C0/C2 robustness matrix；
- 通过场景与 4 N push 0/5 negative result；
- Wilson interval 和小样本限制；
- E5 software-event injection 与 natural-contact C2 的区别；
- E6 compute-time evidence 与 hardware real-time boundary；
- runtime 当前能计算 position commands，但缺少完整 servo bridge、fault handling、E-stop 和 verified scheduling。

### 3.8 证据、文件与学习闭环

- E1--E6 evidence matrix；
- project directory responsibilities；
- 从模型到 archive 的完整流程；
- 当前最强证据、主要不足和下一步；
- 论文核心逻辑；
- 一段完整运行场景；
- 理解检查。

## 4. 去重规则

1. 每个概念只在一个主章节完整解释。
2. 其他位置使用一至两句摘要和 `\hyperref` 跳转。
3. 现有 20 问答改为 compact FAQ：保留问题、短答案和详细章节链接，不重复公式、表格或完整论证。
4. Word 中的“通用算法/本项目实现”表格只保留一张综合对照表。
5. pasted text 的直观类比只用于导读，不在技术章节重复。
6. 同一组实验数值只保留一张主表；其他章节引用表号或结果章节。
7. 同一 claim boundary 在开头完整定义，后续用统一短语提示。

## 5. 事实更新规则

- 代码和当前 YAML/CSV 日志优先于 Word、pasted text 和旧 LaTeX。
- 删除“10-step contact gate 失败、final nominal 未生成”的旧结论。
- 替换为“10-step planning/contact accepted；full 10 s natural-contact replay 尚未通过最终 event gate”。
- 保留 single-step `solver_return_success=false` 的限制。
- 保留 C2 3/4 matched、1 skipped 的限制。
- 保留 4 N push C0/C2 均 0/5。
- 不把测试数量作为永久事实；只有在本次重新运行并记录后才写入具体 test count，否则使用定性表述。
- 对不同实验 protocol 的 delay/lag/speed 参数分开标注，禁止混用。

## 6. LaTeX 排版设计

- 继续使用 `ctexart + XeLaTeX`。
- 保留现有页眉、目录、交叉引用和 bibliography。
- 长表使用 `longtable/tabularx`，避免窄列塞入长段落。
- 公式使用 `amsmath`；变量首次出现时定义单位和物理含义。
- 结果状态统一使用 `PASS / PARTIAL / FAIL / PENDING` 视觉语义。
- 图片通过 `\IfFileExists` 提供 graceful fallback，缺图不阻止编译。
- 避免孤立标题、表格越界、caption 与图片跨页分离和过大的空白。

## 7. 实施与验证

1. 保存原始 LaTeX 备份或通过 diff 保留可追踪性。
2. 先更新事实冲突，再按主题合并独有内容。
3. 扫描重复标题、重复公式、重复表述和滞后关键词。
4. 使用 XeLaTeX/latexmk 完整编译至少两遍。
5. 检查 undefined references、missing citations、overfull/underfull boxes 和 missing files。
6. 渲染 PDF 并逐页检查中文字体、目录、公式、表格、图片和分页。
7. 只有本地编译与视觉检查通过后，才同步目标 `.tex` 及其必要资源到 Overleaf。
8. 同步后核对 Overleaf 版本与本地目标文件 SHA-256。

## 8. 验收标准

- 两个新来源中的有效独有主题均有明确落点。
- 不存在按来源复制产生的大段重复。
- 10-step、replay、C1、C2、robustness 和 runtime 状态与当前证据一致。
- claim boundary 在摘要、结论和关键实验章节一致。
- `project_complete_learning_guide_zh.tex` 可在 Overleaf 使用 XeLaTeX 正常编译。
- 无未解析引用、致命 LaTeX error、不可接受的 overfull box 或明显排版缺陷。
- 不修改 `v4.tex`。

