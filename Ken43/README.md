# Ken43: Frame-Accurate Oki Microgame for Opponent Adaptation

Ken43 是一个固定 **Ken vs Ken、版边、压起身** 场景的 Street Fighter 6 简化仿真环境。项目最终目标不是训练一个只会背连段的 Ken，而是研究：

> 在重复格斗博弈中，显式利用对手模式的策略，是否优于只追求稳健均衡的无记忆策略？

形式上，我们关心的是：

\[
\pi(a_t \mid s_t, h_t)
\quad\text{是否系统性优于}\quad
\pi(a_t \mid s_t)
\]

其中 \(s_t\) 是当前游戏状态，\(h_t\) 是本场 match 内已经观察到的对手行为。长期目标是让智能体不仅能够记住和适应，还会主动执行具有信息价值的试探动作。

## 当前状态

目前已经完成的是可验证的规则与状态基础，还不是可直接进行 PPO 训练的完整环境。

已实现：

- 60 FPS、逐帧动作时钟和基础状态推进；
- startup、active、recovery、hitstun、blockstun 与晚 active 接触；
- cancel gap、true blockstring、Quick Dash、Jinrai branch 与安全跳计算；
- 起身/硬直后投无敌及投捕获失败规则；
- posture-dependent effective range；
- block 后 spacing、pushback、完整/取消 recovery 的不同 \(D_{min}\)；
- 主动放帧误差 \(\epsilon \sim Uniform\{-2,-1,0,1,2\}\)；
- 最小 `FrameState`、动作 mask、帧推进接口；
- 24 个时序、射程和 spacing 回归测试。

尚未实现：

- 完整的双方同步动作 resolver；
- hit/block/whiff、伤害、Drive、DI、Parry、reversal 等端到端结算；
- round、KO、best-of-3 match 生命周期；
- Gymnasium/PettingZoo 标准接口和向量化环境；
- 固定 bot、self-play population、PPO 训练和 checkpoint 管理；
- opponent-history encoder 与 probing 实验。

因此，当前代码适合验证规则和构建训练环境，不应被描述为已经可以直接训练的 RL benchmark。

## 仿真边界

V0 有意保持狭窄和可解释：

- 仅 Ken vs Ken；
- 固定版边压起身 microgame；
- 只开放与压起身攻防相关的动作；
- 未建模动作通过 action mask 禁用；
- 不模拟 juggle、完整弹道飞行、Super、hitstop 和 Perfect Parry 时停；
- jHP 使用已校准的合法跳跃轨迹，不重建完整空中 hitbox；
- 未测量的蹲姿射程会在 JSON 中明确标为暂用站姿射程，而不是伪装成实测值。

权威数据源是：

```text
ken_oki_microgame_v0_4b_verified_spacing.json
```

该文件同时记录帧数、姿态射程、移动、spacing、经验校准和数据质量说明。计算器不得读取 `validation_cases.expected` 作为答案，测试结果必须从规则字段重新推导。

## 核心时序语义

动作帧使用 1-based 编号。startup 为 \(N\) 表示第 \(N\) 帧首次 active。

hitstun/blockstun 包含接触当帧，因此地面技实际帧差为：

\[
Adv(k)=Stun-\left(RemainingActive(k)+Recovery+1\right)
\]

同一招的 stun 固定，越晚的 active 接触，raw advantage 每帧增加 1F。

取消后，父动作剩余 active/recovery 被丢弃。新版数据区分：

- contact cancel：子动作第 1F 从下一 global frame 开始；
- 显式 branch：子动作第 1F 通常从 branch input 的 global frame 开始；
- 若 JSON 提供 empirical timing override，则以实测 override 为准。

定义上一段最后一个 stun frame 为 \(t_b\)，下一攻击第一 active 为 \(t_a\)：

\[
Gap=t_a-(t_b+1)
\]

- `gap < 0`：至少重叠 1F，属于 true blockstring/真正连段；
- `gap == 0`：没有自由帧，但也没有 stun overlap，不属于 true blockstring；
- `gap > 0`：防守方存在对应数量的自由帧。

距离结算遵循严格顺序：

```text
当前距离做 range check
→ hit / block / whiff
→ 仅在接触后应用 spacing transition
→ 得到下一动作开始前的距离
```

典型 block spacing 为：

\[
d'=\max(D_{min},d+\Delta)
\]

whiff 不应用 block pushback。

## 预期环境层级

完整 benchmark 计划使用四层时间结构：

```text
frame
  -> exchange：一次压起身攻防序列
  -> game：多个 exchange，HP/资源持续，直到 KO 或达到上限
  -> match：best-of-3，先赢两局者获胜
```

重置边界：

- exchange 之间：回到规定的版边击倒局面，HP 和需要持续的资源不重置；
- game 之间：HP、位置和指定资源按实验规则重置；
- match 内：opponent memory 不重置；
- match 结束：策略隐藏状态与对手历史全部重置。

底层仿真仍然逐帧推进，但策略优先只在有意义的 decision point 决策，例如起身选择、可行动帧、取消窗口和派生窗口。这样可以保持帧精度，同时避免 PPO 被大量无意义的等待动作淹没。

## RL 定义

基础 Markov observation 至少包括：

- 双方动作、动作帧和可行动状态；
- 距离、站/蹲/空中姿态；
- HP、Drive；
- blockstun、hitstun、knockdown、invulnerability；
- 当前合法动作 mask；
- 当前 game 和 match 比分。

最初的奖励保持接近结果：

\[
r_t=\Delta HP_{opponent}-\Delta HP_{self}
\]

再加入 game 和 match 胜负奖励。不能预先规定“连段、投、压制一定是好行为”，以免把预期打法直接写进 reward。

由于跨局信息价值跨度较长，训练时需要谨慎处理折扣。优先考虑按 decision point 折扣，或使用接近 1 的 match-level discount，避免第一局获得的信息在第二局前已经被折扣掉。

## Opponent History

history-aware 策略维护本场 match 的对手历史：

\[
z_t=f_\phi(h_t),\qquad
\pi(a_t\mid s_t,z_t)
\]

历史不直接存储全部原始帧，而是编码结构化事件：

```text
context / wake-up situation
opponent action and timing
hit / block / whiff
throw / tech / shimmy response
mash after blockstun
reversal / DI / parry / backdash
HP, Drive and previous outcome context
```

第一版使用 GRU/LSTM。Transformer history encoder 只在循环基线验证有效后考虑。

## 实验计划

### Phase 0: Environment Sanity

目标：排除 simulator exploit，建立确定性规则基线。

- 扩展完整动作 resolver；
- 验证 simultaneous interaction、counter-hit、throw/strike、DI armor 和取消窗口；
- 检查 range check 与 spacing update 顺序；
- 固定随机种子，验证相同输入轨迹产生相同结果；
- 建立每个核心机制的人工可解释测试。

通过标准：规则回归全部通过，随机 rollout 不出现非法状态，已知命题与训练模式观测一致。

### Phase 1: Fixed Bots

实现一组参数化对手：

- always block；
- high-frequency mash；
- high-frequency throw tech；
- high-frequency reversal；
- mixed policy。

训练 memoryless PPO，确认它能针对明显风格学出合理 counter。

研究问题：环境是否可学习，而不是只能被脚本利用？

### Phase 2: Single-Game Self-Play

两边共同训练，观察是否自然出现：

- meaty；
- throw / shimmy；
- frame trap；
- reversal bait；
- spacing trap；
- 非退化混合策略。

研究问题：环境中是否存在合理博弈结构，还是存在单一 dominant strategy？

### Phase 3: Population Self-Play

保存历史 checkpoint 和不同风格策略：

\[
\Pi=\{\pi^{(1)},\pi^{(2)},\ldots,\pi^{(k)}\}
\]

训练时从 population 采样对手，避免只适应当前镜像策略。population 同时作为后续 opponent-modeling 的训练分布。

### Phase 4: Best-of-3 Match

将 episode 提升为抢二 match：

```text
Game 1
-> preserve opponent memory
Game 2
-> preserve opponent memory
Game 3 if needed
-> terminal match reward
```

这一阶段只验证 match 生命周期、跨局隐藏状态和长期奖励传播，不立即加入复杂 history encoder。

### Phase 5: History-Conditioned Policy

加入事件历史编码器，比较：

- Markov MLP：只看当前状态；
- recurrent/reset-every-game：网络有记忆，但每局清空；
- recurrent/match-memory：只在 match 结束时清空；
- shuffled-history：历史被打乱的负对照；
- oracle-type：直接提供 opponent type 的性能上界。

核心消融是：

\[
\text{memoryless}\quad vs.\quad\text{match-history-conditioned}
\]

### Phase 6: Exploitable Opponent Population

构造具有持久 latent tendency 的对手，例如：

- 不同 \(p(mash)\)、\(p(tech)\)、\(p(reversal)\)；
- 连续被投后提高 tech 概率；
- 低血量时提高 DI/reversal 概率；
- 被 shimmy 后转为 crouch block；
- 基于前一次结果切换策略。

训练集和测试集应按 opponent parameter/type 分离，避免模型只是记住某个 checkpoint 的表面特征。

主要比较：

\[
R_{history}-R_{memoryless}
\]

以及该差值是否随对手可预测结构增强而增加。

### Phase 7: Active Probing

构造至少两个隐藏对手类型：

- 普通动作下的短期最优反应相同；
- 对某个 probe action 的反应不同；
- probe 有明确的即时机会成本；
- 识别类型后，后续最优策略不同；
- match 剩余收益足以覆盖 probe 成本。

若 history-aware agent 会在前期主动 probe，并根据观察到的反应切换后续策略，而 memoryless/shuffled-history agent 不会，则可视为 information-seeking behavior 的证据。

第一版不向 reward 显式加入 information gain。只有在纯长期回报无法学习时，才把辅助 opponent-type prediction 或显式 belief model 作为独立实验，而不是悄悄改变主实验定义。

## 评估指标

基础性能：

- game win rate；
- best-of-3 match win rate；
- 平均 damage return；
- 对 population 各成员的 matchup matrix；
- 对未见 opponent 参数的泛化表现。

适应能力：

- 前后 game 的策略变化；
- 识别对手倾向所需的 exchange 数；
- history-conditioned 相对 memoryless 的收益；
- shuffled history 后收益是否消失；
- latent state 对 opponent tendency 的可分性。

主动试探：

- probe frequency；
- probe 的即时成本；
- probe 后的策略分化；
- probe 后剩余 match 的增量收益；
- counterfactual 中禁止 probe 后的性能下降。

所有主要结果应报告多个随机种子、置信区间和完整 opponent matchup，而不能只报告单个总体胜率。

## 目录

```text
Ken43/
  ken_oki_microgame_v0_4b_verified_spacing.json
    帧数、动作、姿态射程、spacing 和经验校准数据
  ken43/
    data.py
      JSON loader 与 FrameData
    frame_math.py
      active、advantage、cancel、branch、safe-jump 和执行误差计算
    spacing.py
      posture range、block spacing、backwalk 和连续距离演化
    env.py
      当前最小 FrameState 与逐帧环境骨架
  tests/
    test_validation_cases.py
      时序与系统规则回归
    test_spacing_validation_cases.py
      12 项射程与 spacing 回归
  run_validations.py
    输出时序命题
  run_oki_scenarios.py
    输出放帧误差场景的精确成功率
  run_spacing_validations.py
    输出射程与 spacing 的完整诊断字段
```

## 验证命令

从仓库根目录 `D:\github\SokuRL` 运行，并始终使用项目 `.venv`：

```powershell
.\.venv\python.exe -m pytest Ken43\tests -q
.\.venv\python.exe Ken43\run_validations.py
.\.venv\python.exe Ken43\run_oki_scenarios.py
.\.venv\python.exe Ken43\run_spacing_validations.py
```

最近一次验证结果为 `24 passed`。

## 研究成功标准

项目最终希望依次回答三个问题：

1. **Can the agent remember?** 历史输入是否在严格对照下提高表现？
2. **Can the agent adapt?** 面对不同 latent opponent type，策略是否产生可重复的条件变化？
3. **Can the agent probe?** 策略是否愿意承担局部成本来获得信息，并在后续 match 中收回收益？

只有第三点成立时，Ken43 才真正从“有记忆的 PPO bot”成为 active opponent modeling benchmark。
