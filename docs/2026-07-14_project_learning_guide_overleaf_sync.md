# 项目学习文档与 v4 事实更新的 Overleaf 同步记录

## 1. 本次任务

- 日期：2026-07-14
- Workspace：`/home/zhijian/biped_fddp_project`
- Overleaf project：`6a44c809e8122d234b0f12a5`
- Overleaf branch：`main`
- 状态：**PUSHED；static validation passed；local/remote commit matched**

本次任务完成两项相互独立的文档工作：

1. 新增中文 standalone 学习文档 `project_complete_learning_guide_zh.tex`，用于从零学习项目；
2. 更新论文草稿 `v4.tex` 中已经滞后的事实表述。

学习文档没有通过 `\\input`、`\\include` 或 appendix 引入 `v4.tex`，两者保持独立。

## 2. 输出文件

### 2.1 Workspace source of truth

- `docs/overleaf/project_complete_learning_guide_zh.tex`
- `docs/overleaf/v4.tex`

### 2.2 Overleaf root

- `project_complete_learning_guide_zh.tex`
- `v4.tex`

### 2.3 D 盘学习文档副本

- `D:\\毕业论文\\机器人控制流程图\\Overleaf学习文档\\平面双足机器人_FDDP与FIR_QP_MPC_完整学习指南.tex`
- `D:\\毕业论文\\机器人控制流程图\\Overleaf学习文档\\v4_事实表述更新版_2026-07-14.tex`

## 3. 主要内容与事实边界

学习文档逐项回答源 DOCX 的 20 个问题，并从 robot/model、offline FDDP、MuJoCo replay、C0/C1/C2、scripted robustness 到实验文件索引解释完整软件链路。

`v4.tex` 的更新采用 evidence-layered 表述：

- double-support FDDP CoM shift 有 convergence/feasibility 证据；
- left-swing 与 right-swing single-step 已通过指定 physical gates 和 MuJoCo position-command replay，但不能把 `solver_return_success=False` 的单步结果写成 first-order convergence 成功；
- two-step alternating FDDP 是已收敛的最小左右交替结果，但仍不是 continuous gait；
- ten-step candidate 仅是 planning-only candidate，因为当前 friction-utilisation/CoP gate 失败，不能导出为 final nominal；
- Stage 8/9 仍是 scripted gait，不是 FDDP-generated continuous gait；
- `tau_ref` 仅是 FDDP planning/analysis output，hardware-facing command 仍是 `q_cmd = q_ref + delta_q`；
- 当前只支持 simulation validation 与 runtime-loadability，不声称真实硬件自主行走。

## 4. Git 同步过程

初始本地提交建立在 Overleaf commit `25501c8d1f28409e84169226af5d6ce13dfe562d` 上。首次安全推送前检测到远端已经前进至 `44e72ed55f8770feb4a3e528cf3bf33aa05e78b3`，因此自动拒绝推送，没有覆盖远端内容。

检查显示远端新提交仅新增 `learn.tex`。本地提交随后无冲突 rebase 到最新 `origin/main`，并推送为：

- Commit：`72c531794568d50141f030e37385b56906068850`
- Message：`add_project_learning_guide_and_refresh_v4_evidence`
- Verified local HEAD：`72c531794568d50141f030e37385b56906068850`
- Verified `origin/main`：`72c531794568d50141f030e37385b56906068850`
- Final worktree：`main...origin/main`，无未提交文件

认证采用一次性 `GIT_ASKPASS` 和 process-local environment variable。token 未写入 remote URL、Git config、项目文件或本记录；临时 askpass 文件由 shell trap 删除。

## 5. 验证命令与结果

### 5.1 Static TeX validation

```bash
/home/zhijian/miniforge3/envs/fddp_biped/bin/python \
  tools/_validate_tex_learning_docs.py \
  /tmp/overleaf_project_6a44c/project_complete_learning_guide_zh.tex \
  /tmp/overleaf_project_6a44c/v4.tex \
  /tmp/overleaf_project_6a44c
```

结果：

```text
static TeX validation passed: labels=49, refs=15, questions=20, citations=7, figures=3
```

检查范围包括 brace balance、internal label/reference、20 个问题标签、citation key、figure asset、unsafe underscore、standalone isolation、required v4 updates 和 forbidden stale phrases。

### 5.2 Git 与内容一致性

```bash
git diff --check origin/main
git status --short --branch
git rev-parse HEAD
git rev-parse origin/main
sha256sum project_complete_learning_guide_zh.tex v4.tex
```

结果：`git diff --check` 无输出；local HEAD 与 `origin/main` 完全相同；Overleaf clone 与 workspace 两份源文件的 SHA-256 分别一致。

| 文件 | SHA-256 |
|---|---|
| `project_complete_learning_guide_zh.tex` | `85a8bfd72ffa763198ed46409771a7e2059b52d4179b58ed76580ddbb47352b4` |
| `v4.tex` | `60e41c7fa7d95e07ea914edcf5e880585d18ff44b4dfe5930f07d4e6d076531d` |

### 5.3 Compile status

WSL 中检查 `xelatex`、`latexmk`、`lualatex` 和 `pdflatex`，均为 `not found`。因此本次只确认 static validation 与 Overleaf Git publishing；没有把本地 PDF compilation 或 Overleaf online compilation 写成已完成。

## 6. D 盘归档结果

执行：

```powershell
& '\\wsl$\Ubuntu-22.04\home\zhijian\biped_fddp_project\tools\sync_thesis_results_to_d_drive.ps1'
```

首次同步回报：

```text
Archived and verified 278 files (97361875 bytes).
Destination: D:\毕业论文\机器人控制流程图\FDDP仿真结果归档
Manifest: D:\毕业论文\机器人控制流程图\FDDP仿真结果归档\SHA256_MANIFEST.csv
```

关键文件逐字节比较结果：`RECORD_ARCHIVE_MATCH=yes`、`GUIDE_ARCHIVE_MATCH=yes`、`V4_ARCHIVE_MATCH=yes`。记录追加完成后再次运行同一同步脚本，以最终 manifest 为准。

## 7. 最终结论

两个 LaTeX 文档已经推送到指定 Overleaf `main`，并通过静态结构、事实边界、Git remote commit 和 SHA-256 一致性检查。学习文档保持 standalone；`v4.tex` 的陈述已更新到当前软件/仿真证据，但没有扩大为 continuous FDDP walking 或 real-hardware walking claim。
