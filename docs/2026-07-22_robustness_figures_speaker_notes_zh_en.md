# Robustness Figures: Slide Text and Bilingual Speaker Notes

This note accompanies the two presentation figures generated from the recorded
FDDP/MuJoCo evidence. All numerical statements below are simulation results.

## Figure 1 — Bidirectional Push-Recovery Robustness

### Recommended English slide text

**Title:** Bidirectional Bounded-Push Recovery

**Purpose:** Test whether the C2 controller recovers without violating the
stability and command-safety gates when the same disturbance is applied in
opposite directions.

- Disturbance: ±4 N torso push, applied at 0.8 s for 0.12 s.
- Coverage: 30 seeded trials per direction; 60 trials in total.
- Result: 30/30 forward and 30/30 backward trials passed all recorded gates.
- The backward push is more demanding: mean peak pitch error is 13.71°, with
  only 1.25° worst-case margin to the 15° gate.

**Take-away:** Recovery succeeds in both directions, but the asymmetric pitch
margin identifies the backward push as the more critical case.

**Claim boundary:** Scripted MuJoCo/FDDP validation; not a hardware result.

### 中文讲稿（约 50–70 秒）

这一页不是只想说“成功率是百分之百”，而是想回答两个更重要的问题：第一，
控制器能不能承受正反两个方向的扰动；第二，即使都成功了，哪一个方向离失效边界
更近。

测试是在 0.8 秒时向机器人躯干施加持续 0.12 秒、大小为 4 N 的推力。正向和反向
各运行 30 个 seed，所以一共是 60 次独立仿真。图中的每一个小点代表一次试验的
最大绝对俯仰误差；黑色短横线表示平均值，黑色误差线表示本次试验实际观测到的
最小值到最大值，而红色虚线是项目采用的 15° 俯仰角验收上限。

两个方向都是 30/30 通过，但它们并不等价。正向推扰的平均峰值误差是 9.02°，
最差一次仍有 5.96° 的余量；反向推扰的平均值达到 13.71°，最差一次只剩 1.25°
余量。因此正确结论不是“系统非常稳，两个方向完全一样”，而是“在当前边界内两侧
都能恢复，但反向推扰明显更接近稳定性限制，是下一阶段硬件试验首先应该覆盖的
critical case”。

### How to read every visual element

- **Blue open circle:** one forward-push seeded trial.
- **Orange filled diamond:** one backward-push seeded trial; the different
  marker remains distinguishable in grayscale.
- **Black centre mark:** mean peak absolute pitch error.
- **Black capped range:** minimum-to-maximum value actually observed in the 30
  trials; it is not a confidence interval.
- **Red dashed line:** the 15° per-trial pitch acceptance gate.
- **Worst-case margin:** `15° - maximum observed peak pitch error`; it is a
  descriptive margin, not an additional gate.

### Questions likely to be asked

**Why do the 30 dots almost form a horizontal line?**  The scripted campaign
uses the same bounded pulse and the controller response is highly repeatable;
the seeds produce only a narrow spread in peak pitch. The horizontal jitter is
only used to make overlapping trials visible.

**Can 30/30 prove that recovery will always work?**  No. It proves that all 30
recorded trials passed this specified simulation protocol. It does not prove
performance outside ±4 N, outside the tested timing, or on hardware.

**Why emphasise the backward direction if it also passed?**  Robustness is not
only pass/fail. Remaining margin indicates which accepted condition is nearest
to the design boundary and therefore deserves priority in future testing.

## Figure 2 — Full 32-Scenario Robustness Campaign

### Recommended English slide text

**Title:** Robustness Evidence Across the Full 32-Scenario Campaign

**Purpose:** Test whether the final C2 controller preserves the complete trial
acceptance contract under disturbances, model variation, latency, actuator
variation, noise, terrain changes, and combined conditions.

- Coverage: 32 scenarios × 30 seeds = 960 scripted simulation trials.
- Result: 960/960 trials passed; every scenario has a 100% observed pass rate.
- All recorded peak pitch errors remain below the 15° per-trial gate.
- The most demanding scenario by maximum pitch error is `Push -4 N`, reaching
  13.70° in this campaign.
- A 30/30 result has a Wilson 95% lower confidence bound of 88.65%; therefore
  the interval communicates sample uncertainty and must not be described as a
  guaranteed 100% true success probability.

**Take-away:** The controller passes a broad, pre-defined simulation campaign,
while the pitch-margin and confidence-interval views reveal severity and
statistical uncertainty that a uniform green bar chart hides.

**Claim boundary:** Simulation robustness is supported; hardware robustness
still requires repeatable physical disturbance, terrain, delay, and safety
tests.

### 中文讲稿（约 80–110 秒）

这一页把 robustness 从一个简单的“全绿柱状图”改成了两个互补问题。左侧问的是：
在每种条件下，机器人的姿态响应有多严重？右侧问的是：在有限试验次数下，我们对
通过率掌握了多少统计证据？

整个 campaign 有 32 个场景，每个场景 30 个 seed，总计 960 次仿真。覆盖的类别
包括 nominal、正反向推扰、摩擦系数、正负坡度、台阶高度、命令延迟、传感器延迟、
执行器增益、执行器时间常数、噪声以及三种 combined condition。最终记录是
960/960 通过，因此每个场景的 observed pass rate 都是 100%。

每一行左侧的点表示该场景 30 次试验的平均峰值俯仰误差，横线表示实际最小到最大
范围，红色虚线是 15° 上限。这样可以看出，不同场景虽然都通过，但难度不同。
其中最大俯仰响应来自 `Push -4 N`，本 campaign 的最大值为 13.70°，所以它比大多数
摩擦、延迟和噪声场景更接近姿态边界。

右侧点是 observed pass rate，横线是 Wilson 95% confidence interval。这里要特别
解释：30/30 的观测成功率确实是 100%，但样本只有 30 次，所以真实成功概率的
95% 置信下界约为 88.65%。这不是说系统只成功了 88.65%，而是说有限样本仍有
不确定性。项目的单场景验收条件使用 observed success rate ≥ 90%；图中的 Wilson
区间是额外的统计诚实性表达，不能把它偷换成项目原本的通过条件。

因此这页可以支持的表述是：“最终 C2 控制器在预先定义的 32 场景、960 次 MuJoCo
仿真中全部通过，而且关键稳定性指标没有越界。”不能说成“已经证明真实机器人在
所有扰动下都具有鲁棒性”，因为接触建模误差、执行器饱和、结构柔顺性、通信抖动、
地面材料和真实推力重复性仍需要硬件验证。

### How to read every visual element

- **One row:** one named scenario, never a category average.
- **Orange filled circle:** a scenario marked critical in the recorded campaign.
- **Blue open square:** an intermediate scenario. Colour and fill/shape encode
  the same distinction for grayscale safety.
- **Left dot and range:** mean and observed minimum-to-maximum peak pitch error.
- **15° vertical dashed line:** per-trial pitch gate.
- **Right dot:** recorded success count divided by 30 trials.
- **Right horizontal interval:** Wilson 95% confidence interval for that finite
  sample; it is not the range of pass rates across seeds.
- **90% dashed line:** the recorded per-scenario observed-rate gate.
- **95% dotted line:** the campaign target reference, not a lower-confidence-bound
  gate.

## What the figures support about the three KPIs

### Stability

The figures provide stability evidence because the recorded trial gate includes
bounded pitch and additional fall/contact-integrity checks. In the underlying
campaign a trial also requires maximum unintended flight ≤ 0.02 s, minimum
height fraction ≥ 0.65, contact-event match ≥ 0.90, valid event order, maximum
controller correction `|alpha| ≤ 5°`, and zero command saturation. Step cases
must additionally encounter the obstacle. The plotted pitch gate is therefore
the most presentation-friendly part of a wider acceptance contract, not the
only definition of stability.

### Robustness

The figures provide robustness evidence through breadth and repetition:
directional pushes, physics variation, terrain, timing, actuator variation,
noise, and combined conditions are each repeated over 30 seeds. The correct
claim is **robustness within the tested simulation envelope**.

### Agility

These two figures do not directly prove agility. Agility needs separate evidence
about commanded motion range or speed, response time, gait-transition time,
tracking bandwidth, or task completion under faster commands. Do not rename
this robustness campaign as an agility result.

## Recommended placement in the current presentation

- Replace the existing slide 24 robustness graphic with
  `robustness_bidirectional_push_profile.png`.
- Replace the existing slide 25 all-green success bars with
  `robustness_32_scenario_profile.png`.
- Use the PNG files for PowerPoint and retain the PDF files for the thesis or
  later vector editing.

## Literature design references

The redesigned figures follow the communication logic used in comparable
legged-robot studies: show the disturbance envelope and recovery response, not
only one aggregate bar; separate success frequency from response severity; and
make test conditions traceable.

1. Gu, Boyd, and Zhao, *Reactive Locomotion Decision-Making and Robust Motion
   Planning for Real-Time Perturbation Recovery* — perturbation direction,
   magnitude, phase, and recovery success are shown as structured maps and
   profiles: https://arxiv.org/pdf/2110.03037
2. Haarnoja et al., *Learning to Walk via Deep Reinforcement Learning* —
   perturbation response is presented as time/trajectory behaviour and recovery,
   rather than only a single pass-rate bar:
   https://www.roboticsproceedings.org/rss15/p11.pdf
3. Song et al., *Bipedal Stepping Controller Design Considering Model
   Uncertainty: A Data-Driven Perspective* — success-rate summaries are paired
   with slope/disturbance recovery evidence:
   https://www.mdpi.com/2313-7673/9/11/681
