# 平面双足机器人项目零基础 LaTeX 讲解文档设计

日期：2026-07-19
Status: approved by user

## 1. 目标

生成一份可以直接上传 Overleaf、使用 XeLaTeX 独立编译的中文讲解文档。目标读者假设没有机器人学、最优控制或 MPC 基础。文档要从“机器人为什么会倒、走路为什么困难”开始，逐步解释本项目从建模、FDDP 规划、MuJoCo 重放、C0/C1/C2 在线修正，到安全、故障注入和统计鲁棒性验证的完整过程。

文档既用于个人学习，也应支持论文写作、组会汇报和答辩复习。

## 2. 交付物

- 新建独立源文件：`docs/overleaf/biped_project_from_zero_guide_zh.tex`。
- 不覆盖现有 `project_complete_learning_guide_zh.tex`，但优先复用其已经核对过的公式、术语、参考文献和项目事实。
- 生成本地编译 PDF（本机工具允许时），并检查目录、分页、表格、公式与中文字体。
- 文档不依赖 `v4.tex`，也不自动并入论文正文。

## 3. 信息来源优先级

1. 当前 workspace 的 FDDP、simulation、MPC、hardware runtime 和 tests 源码。
2. 当前 YAML、CSV、NPZ、PNG 与实验记录，尤其是最新 32 场景乘 30 种子正式结果。
3. 现有完整学习指南和 Overleaf 参考文献材料。
4. 原始论文与官方项目文档，用于说明算法出处，不用于替代本项目实测证据。

如果旧文档与当前产物冲突，以当前可执行代码和机器可读结果为准，并明确旧结果属于历史阶段。

## 4. 讲解方式

每个重要概念都采用同一教学结构：

1. 它是什么；
2. 为什么项目需要它；
3. 一个不依赖数学的直观例子；
4. 最少但必要的公式，并逐个解释变量；
5. 本项目具体怎样使用；
6. 代码与结果证据在哪里；
7. 常见误解和不能夸大的结论。

首次出现的术语保留中英文，例如“模型预测控制（Model Predictive Control, MPC）”。公式不连续堆叠；每个公式后必须用中文解释输入、输出、单位和实际意义。

## 5. 文档结构

### 5.1 导读

- 一句话项目目标；
- 机器人从离线规划到位置命令的总流程；
- 当前声明边界；
- 零基础、论文写作、组会准备三条阅读路线。

### 5.2 先理解机器人本身

- 六关节平面双足机器人；
- sagittal plane、torso pitch、CoM、support foot、swing foot；
- 位置舵机与力矩电机的区别；
- 为什么最终命令是 `q_cmd = q_ref + delta_q`；
- 为什么 `tau_ref` 只是规划与分析量。

### 5.3 建模工具链

- URDF、MJCF、Pinocchio、Crocoddyl、MuJoCo 的职责；
- `nq`、`nv`、状态、控制、坐标系；
- 为什么规划模型与仿真模型不同仍然可以验证；
- 模型一致性 audit 的作用。

### 5.4 FDDP：先在计算机里规划怎样走

- optimal control、finite horizon、trajectory、cost、constraint；
- DDP、FDDP、backward pass、forward rollout、line search；
- shooting node、shooting gap 与 feasibility；
- 双支撑/单支撑接触调度；
- residual、weight、regularisation、warm start、continuation；
- 项目怎样从站立、CoM shift、单步、两步逐渐扩展到十步；
- planning gate、contact wrench、friction、CoP、clearance 和 joint limit。

### 5.5 MuJoCo 重放：为什么规划成功不等于机器人能执行

- forward simulation 与 trajectory optimisation 的区别；
- position actuator proxy、delay、lag、command advance；
- natural contact event、tracking RMSE、pitch、height、flight；
- 十步名义重放证据。

### 5.6 C0、C1、C2 的能力递增

- C0：只执行名义轨迹；
- C1：固定参考时钟的俯仰恢复；
- phase-only：只修正时间而不做姿态修正；
- C2-final：C1 加接触事件相位同步；
- 用一张横向对比表说明输入、输出、能力和局限。

### 5.7 FIR、AR、QP 与 MPC 逐词解释

- impulse response、finite impulse response；
- 为什么用历史输入预测未来 pitch；
- AR free response；
- prediction horizon 与 receding horizon；
- QP 的目标项和 box constraints；
- 标量 `alpha`、固定映射 `G`、关节修正 `delta_q = G alpha`；
- 为什么这是面向低成本位置舵机的局部、快速 MPC，而不是在线重求全身 FDDP。

### 5.8 接触相位同步

- 为什么实际 liftoff/touchdown 不会严格按计划发生；
- contact observer、debounce、event matching；
- phase offset、event-type memory、local-leg retiming；
- C2 为什么改善接触时序而不是重新规划落脚点。

### 5.9 安全、故障和实时性

- limiter、saturation、fallback、finite command；
- NaN/Inf、stale sensor、event loss、QP failure 等故障；
- runtime-loadability 与真实串口/舵机运行的区别；
- 位置命令路径的硬件边界。

### 5.10 鲁棒性与统计验证

- disturbance、friction、slope、step、delay、gain、noise 与 combined scenarios；
- seed、trial、success gate、failure category；
- 为什么 3 次通过不能代替 30 次；
- Wilson confidence interval 的直观含义；
- 有效台阶几何、初始零穿透和后续真实遇障证据；
- 最新 32 乘 30 等于 960 次正式结果。

### 5.11 项目时间线与阶段成果

- 从 Stage 0 建模到最终 Task 9 的顺序；
- 每一步遇到的问题、修正思路、得到的证据；
- 历史失败如何推动后续设计；
- 已完成和未完成项目表。

### 5.12 论文、组会和答辩怎么讲

- 一分钟、三分钟和十分钟版本；
- 每一页 PPT 应放的图与证据；
- 导师可能追问的问题；
- 安全、准确且不过度声明的回答模板。

### 5.13 术语表、文件地图与参考文献

- 中英文术语；
- 主要脚本、配置、轨迹、日志和结果图索引；
- FDDP、Crocoddyl、Pinocchio、MuJoCo、MPC 等参考文献；
- 推荐的进一步学习顺序。

## 6. LaTeX 排版

- 使用 `ctexart`、A4、11pt 和 XeLaTeX。
- 使用目录、页眉页脚、超链接、`booktabs`、`tabularx`、`longtable`。
- 使用浅色解释框区分“直觉”“公式”“本项目实现”“证据”“常见误解”。
- 总流程使用可编译的 TikZ 或表格式流程图；外部结果图均用 `\IfFileExists` 包装，缺图时显示说明框而不导致编译失败。
- 路径、下划线和长代码名必须安全换行，避免 Overleaf overfull box。

## 7. 事实与声明边界

- 正式结果必须更新为当前 32 场景、960 trials、960 passed、`accepted=true`。
- 十步名义规划、MuJoCo 重放、C0/C1/C2、恢复、安全与正式鲁棒性分别引用各自证据，不能混成一个数字。
- 允许表述“当前定义的软件和仿真验收链已经完成”。
- 不允许表述“真实机器人已经行走”或“硬件已经验证”。
- 最大障碍物穿透等软接触指标如实列出，并说明 MuJoCo soft-contact 的含义。

## 8. 验证标准

- XeLaTeX 编译成功且 PDF 非空；
- 无未定义引用、缺失图片导致的编译失败或明显 overfull box；
- 所有章节均含直观解释和项目落地说明；
- 最新正式统计与 YAML/CSV 一致；
- 文档中没有“Task 7/8/9 尚未完成”等过时表述；
- 参考文献可复制，正文引用能够解析；
- 完成后对关键章节和 PDF 页面做人工视觉检查。

## 9. 非目标

- 不修改论文 `v4.tex`；
- 不新增控制算法或重新运行 960 次仿真；
- 不把仿真结果夸大为硬件实验；
- 不用新框架替代项目已有脚本和证据。
