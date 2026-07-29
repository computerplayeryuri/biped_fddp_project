# NotebookLM 组会 PPT 修改包

用途：修订 `7.17.pptx`，形成一份面向专业 robotics/control 听众的 15 分钟组会演示。

## 推荐上传顺序

NotebookLM 桌面版支持 PPTX、Markdown、TXT、CSV 和常见图片格式。ZIP 仅用于传输，请先解压，不要把 ZIP 本身当作唯一 source。

### 核心必传

1. `01_原始PPT/7.17.pptx`
2. `02_项目事实/项目证据摘要_2026-07-15.md`
3. `03_修改指令/NotebookLM_PPT修改提示词.txt`
4. `04_关键图片/` 中所有 PNG

### 建议补传

- `05_关键日志/` 中的 YAML，用于核对精确数值和 claim boundary。
- `06_动画/` 中的 GIF，用于选择组会演示片段。

## 使用方法

1. 新建一个 NotebookLM notebook。
2. 上传“核心必传”文件，再按需要上传 YAML 和 GIF。
3. 将 `NotebookLM_PPT修改提示词.txt` 的全文粘贴到 chat。
4. 要求 NotebookLM 先输出逐页 revision plan，再生成或修改 deck。
5. 人工复核所有数值、公式、图注和通过/失败状态。

## 不可突破的边界

- 当前证据属于 simulation validation 与 runtime-loadability。
- 不可声称真实硬件行走或 hardware real-time validation。
- `tau_ref` 只用于 FDDP planning/analysis，不是 hardware torque command。
- Stage 8/9 是 scripted gait，不是 FDDP-generated continuous gait。
- 10-step planning/contact gates 已通过，但 full 10 s MuJoCo replay 尚未通过最终 natural-contact event gate。

