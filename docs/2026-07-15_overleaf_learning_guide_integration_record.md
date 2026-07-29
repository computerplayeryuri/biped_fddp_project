# 2026-07-15 Overleaf 完整中文学习指南整合记录

## 1. 任务

将以下两份学习材料去重、校订并整合到独立 Overleaf 文档：

- FDDP与FIR_QP_MPC算法讲述与参数说明.docx
- Codex attachment pasted-text.txt

目标文件：docs/overleaf/project_complete_learning_guide_zh.tex

本任务没有修改 v4.tex，学习指南也没有作为论文附录引入。

## 2. 主要更新

- 补充 FDDP dynamics、optimal-control formulation、backward/forward pass、regularisation 与 shooting gap。
- 统一 6D contact、DS--SS--DS、quintic swing、planning gate 与 replay gate。
- 将 10-step 当前状态更新为 planning_and_contact_accepted。
- 明确 full 10-step MuJoCo natural-contact replay 尚未通过 event matching。
- 合并 C0/C1/C2、FIR、AR、QP、balanced candidate 与 holdout validation。
- 增加 70-trial robustness matrix、Stage 8/9 scripted-gait 边界、E6 timing/safety 与 runtime-loadability。
- 修正 FAQ、下一步优先级与答辩总述中的滞后表述。
- 将长路径改为可换行形式，消除 PDF 横向溢出。

## 3. 当前关键证据

### 10-step FDDP planning/contact

- acceptance state: planning_and_contact_accepted
- solver stop: 4.410209249788341e-08
- iterations: 66
- max shooting gap: 0
- active-support translation error: 55.2748 um
- landing translation error: 35.7256 um
- minimum clearance: 19.9929 mm
- minimum joint margin: 11.7898 mrad
- forward CoM: 50.0869 mm
- minimum normal resultant: 9.9126 N
- maximum friction utilisation: 0.5807077
- minimum CoP margin: 2.3913 mm
- active contact-row violations: 0/1300

### 10-step nominal 的 two-step smoke replay

- finite: true
- no fall: true
- max joint RMSE: 2.880 deg
- max pitch: 9.391 deg
- minimum height fraction: 0.979
- maximum penetration: 0.244 mm
- event match: 2/4
- acceptance: false

### C2、robustness 与 E6

- two-step C2 event match: 3/4
- maximum event offset: 47.375 ms
- 70-trial scripted robustness matrix 中 forward push +4 N 为 C0/C2 均 0/5
- 以上 robustness 结果不能外推为 FDDP 10-step gait robustness
- 50 Hz controller deadline: 20 ms
- 4 conditions x 61 updates = 244 updates
- worst p95: 1.082 ms
- observed maximum: 1.508 ms
- deadline misses: 0
- fallback_available=false，不能声称完整 safety/fallback 已完成

## 4. 验证

临时 build engine：Tectonic 0.16.9。源文件兼容 XeLaTeX，Overleaf root 的 latexmkrc 设置 pdf_mode=5。

执行：

    /tmp/tectonic_overleaf_env/bin/tectonic \
      project_complete_learning_guide_zh.tex \
      --keep-logs --keep-intermediates

结果：

- exit code: 0
- PDF pages: 24
- Undefined control sequence: 0
- LaTeX Error: 0
- Overfull hbox: 0
- begin/end environment: 56/56
- 剩余 warnings 为窄表格单元中的 Underfull hbox；visual QA 未发现 clipping 或 boundary overflow
- 已用 Poppler 渲染页面，重点复查第 23 页 file index 与第 24 页 conclusion/references
- TeX SHA-256: 497e68168598dc494bb14e4be9e1c3485d8ed1045dbc1aed5b48039b3c4094e3
- PDF SHA-256: 9fb935266b511b88a0314740bb577213ee99fa3ff00707e1b3a2efc2a08201c9

## 5. Overleaf

- project: 6a44c809e8122d234b0f12a5
- branch: main
- commit: 50232db3a7b49a4b60739cac7426b243c23655f8
- subject: update complete Chinese learning guide and evidence
- push 后已独立检查 remote refs/heads/main

Git push、本地 standalone compile 与 visual QA 已验证。由于本会话没有可用的 Chrome browser-control runtime，未在 Overleaf UI 中独立观察 cloud compile 状态，因此不把 UI compile 写成已验证。

## 6. Claim boundary

当前可声称：

- simulation validation
- runtime-loadability
- 10-step FDDP planning/contact feasibility accepted
- local standalone learning-guide build passed
- Overleaf Git remote updated

当前不可声称：

- full 10-step natural-contact replay accepted
- FDDP continuous autonomous walking 已完整闭环
- full safety/fallback 已验证
- 真实硬件自主行走


## 7. D 盘同步

归档目录：

- D:/毕业论文/机器人控制流程图/FDDP仿真结果归档/docs/overleaf/
- D:/毕业论文/机器人控制流程图/科研测试记录/

最终核对：TeX MATCH=true，PDF MATCH=true，record MATCH=true，8 张证据图 MATCH=true。
