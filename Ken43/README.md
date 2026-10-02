# Ken43: Frame-Accurate Oki Microgame for Opponent Adaptation

> 当前这一轮研究已经停止训练。项目的设计、调试过程、失败结果和停止原因见
> [`RETROSPECTIVE.md`](RETROSPECTIVE.md)。README 以下内容保留为环境与训练接口说明。

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

当前训练入口是 Drive-only 角落 interaction：不观察或奖励 HP，一次 episode 在破防、投、DI 上墙、拆投、主动脱压或换边时结束。超过 2MK 射程只标记为 `disengaged`，进攻方自身 pushback 造成的拉开不会判防守方获胜。前 360F 没有时间成本；361–600F 默认施加累计 `-0.5` 的二次增长时间压力，真正 timeout 再对双方分别施加 `-0.5`。旧 checkpoint 与当前 observation/action schema 不兼容，只作为历史产物保留。

PettingZoo 环境仍把 600F hard cap 报告为 truncation。供 PPO 使用的 `Ken43GymEnv` 默认把这个带惩罚的 timeout 作为学习终点返回，防止 SB3 的 TimeLimit bootstrap 用 `gamma * V(terminal_state)` 抵消惩罚；这不会给任何一方胜利 bonus。实验需要旧式 bootstrap 时可设置 `bootstrap_timeouts=True` 或训练参数 `--bootstrap-timeouts`。

已实现：

- 60 FPS、逐帧动作时钟和基础状态推进；
- startup、active、recovery、hitstun、blockstun 与晚 active 接触；
- cancel gap、true blockstring、Quick Dash、Jinrai branch 与安全跳计算；
- 起身/硬直后投无敌及投捕获失败规则；
- 由 `attack_reach + defender_hurtbox_half_width` 计算的姿态相关射程；
- block 后 spacing、pushback、完整/取消 recovery 的不同 \(D_{min}\)；
- 主动放帧误差 \(\epsilon \sim Uniform\{-2,-1,0,1,2\}\)；
- 对称双人 `CombatState`、动作 mask 和统一 `resolve_frame()`；
- 同快照生成双方 contact、统一提交 trade 的逐帧事件结算；
- strike/throw、guard、invulnerability、stun、damage、knockdown；
- DI 的最小闭环：反 DI、多段破甲和投绕过护甲；
- 轻旋风 2 hit、中升龙 2 hit、重升龙 3 hit、KK 升龙 6 hit 的聚合多段语义；
- contact cancel、显式 branch、action end 和经验时序 override；
- 固定第 43F 起身、起身首帧投无敌和首帧 reversal；
- endpoint-compatible aggregate spacing，以及供未来实测数据使用的可选 `MovementProfile`；
- 浮点 Drive cost/gain/loss、CH/PC reward multiplier 和可配置延迟恢复；
- DI close-range domain、armor conversion、corner wall splat；
- Sacrifice Jump 的 1–4F/起跳/威胁相关滞空/3F 落地生命周期；
- PettingZoo `ParallelEnv`、32 维无 HP 观察、参数化 `MultiDiscrete` 动作和动态 action mask；
- 九个 scripted policy 与角色平衡 pairwise payoff matrix；
- Gymnasium 单智能体 facade、5 个 fixed bot 和 MaskablePPO 启动脚本；
- fixed-bot checkpoint 的确定性评估脚本；
- 97 个时序、射程、combat-effects、诊断日志和训练接口回归测试。

尚未实现：

- round、HP、KO、best-of-3 match 生命周期；
- 向量化环境、self-play population 和 checkpoint population 管理；
- recurrent PPO 与正式的大规模训练配置；
- opponent-history encoder 与 probing 实验。

因此，当前代码可以开始 Phase 1 fixed-bot 训练与学习性验证，但还不是完整的 population self-play / opponent-modeling benchmark。

## 仿真边界

V0 有意保持狭窄和可解释：

- 仅 Ken vs Ken；
- 默认版边、中心距离 70、+43F 压起身，并支持 +26/+38/+43 配置；
- 只开放与压起身攻防相关的动作；
- 未建模动作通过 action mask 禁用；
- 迅雷、Quick Dash 和 target combo 派生不接受独立 `action` 输入，只能通过合法 branch/cancel window 进入；
- 不模拟 juggle、完整弹道飞行、Super、hitstop 和 Perfect Parry 时停；
- jHP 使用已校准的合法跳跃轨迹，不重建完整空中 hitbox；
- Ken 站姿/蹲姿横向 hurtbox 半宽分别为 39.8/47.0，所有已测 normal 共用几何射程规则。

权威数据源是：

```text
ken_oki_microgame_v0_4b_verified_spacing.json
```

该文件同时记录帧数、姿态射程、移动、spacing、经验校准和数据质量说明。计算器不得读取 `validation_cases.expected` 作为答案，测试结果必须从规则字段重新推导。

### Deferred by Design

以下机制不是 V0 的遗留缺口，而是当前研究范围内明确推迟：

- projectile 对象与完整飞行生命周期；
- Parry、Drive Rush、Drive Reversal 和复杂 Burnout；
- 通用 knockdown duration 与不同击倒技的起身优势；
- 逐动作、逐帧 animation movement trajectory；
- 多段技逐 hit 的 hitbox、stun 与空间轨迹。

因此 `drive_parry`、Drive Rush、Drive Reversal 和未保留的 OD 技不进入训练动作目录。Drive cost、命中回复、防御/PC 消耗、投命中回复、拆投回复和延迟自然恢复已经启用。combat 数据含 `null` 的普通路线继续由 mask 阻止；Jinrai 派生的缺失 Drive 精度仍需后续数据补全。

多段技采用聚合规则：第一段在允许打满的距离成功接触后，整招按固定 hit count 参与 DI 护甲计算，伤害使用 JSON 中的整招总伤害，最终一段负责 knockdown/end state。KK 升龙目前只有 6 hit 语义，没有独立帧表，所以对应 Quick Dash branch 仍会被拒绝，不能作为可执行动作。

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

## Unified Frame Resolver

逐帧仿真的主要入口是：

```python
next_state, events = resolver.resolve_frame(
    state,
    (player_0_input, player_1_input),
)
```

`CombatState.players[0/1]` 是对称的，不永久绑定“进攻方/防守方”。每个 global frame 按固定阶段执行：

```text
推进已有动作帧
→ 启动 queued action / 读取双方输入
→ 处理本帧 branch
→ 应用显式 movement（V0 默认无逐帧动画位移）
→ 从同一个 pre-commit snapshot 生成 contact candidates
→ 解决 strike/throw、invulnerability、armor 和 trade
→ 统一提交 damage / stun / knockdown / spacing
→ 处理 contact cancel
→ 处理 action end
→ 递减 stun / throw invulnerability
```

resolver 返回结构化 `FrameEvent`，包括 `ActionStarted`、`Hit`、`Block`、`RangeMiss`、`ArmorAbsorb`、`ArmorBreak`、`CounterDI`、`ThrowWhiff`、`CancelQueued`、`BranchStarted`、`Movement` 和 `ActionEnded`。训练 reward、调试 trace 和 replay 应从事件流派生，不在 resolver 内写入策略偏好。

当前 JSON 大多只有动作结束时的 spacing endpoint，不能唯一反推出动画每帧位移。V0 以 aggregate spacing 为标准语义；resolver 仍保留两种移动来源，方便未来接入确有必要的实测轨迹：

- `per_frame`：来自已知或注入的 `MovementProfile`，在 contact 判定前应用；
- `aggregate_*_fallback`：缺少逐帧数据时使用现有 endpoint，并在事件中明确标记。

对于 5HP，`FrameInput.recovery_mode` 默认是 `hold2`，接触和空挥均直接使用 hold2 endpoint；显式选择 `non_hold2` 才使用完整 recovery endpoint。其他动作继续使用各自唯一的 aggregate endpoint。只有当某个 startup/active/recovery 中途位移被实测证明会改变当帧射程胜负时，才需要为该动作增加 `MovementProfile`。

## Training Interface

`Ken43ParallelEnv` 是当前训练入口。双方在同一 global frame 提交参数化动作，环境返回 PettingZoo Parallel API 的 observation、reward、termination、truncation 和 info：

```python
import numpy as np
from Ken43.ken43 import Ken43ParallelEnv

env = Ken43ParallelEnv(max_frames=600, drive_reward_weight=1.0)
observations, infos = env.reset(options={"distance": 70.0})

while env.agents:
    actions = {agent: env.command_index["noop"] for agent in env.agents}
    observations, rewards, terminated, truncated, infos = env.step(actions)
```

每个 agent 的 observation 是：

```text
observation: float32[32]
action_space: Discrete(368)
action_mask: MultiBinary(368)
```

每个离散 action 都是一条完整命令，例如 `immediate:5LP`、`schedule:5:5LP` 或 `branch:3:kazekama`。这种扁平表示让 action mask 能表达联合合法性，不会组合出 `schedule:guard` 或 `schedule:0:5LP`。schedule 只接受 1–10F，并在提交时只采样一次 `Uniform{-2,-1,0,1,2}` 误差。Jinrai 派生的 delay 从首次合法 branch window 相对计算。react/confirm 已删除。5MK、5HK、2HK 均可 immediate 或 schedule，用作稳定 frame-kill。

逐帧 reward 为零和形式，只使用 Drive 削减和事件 bonus，不使用 HP：

\[
r_i=\lambda_D\left(\Delta D^{opp}-\Delta D^{self}\right)+event\_bonus
\]

默认 \(\lambda_D=1\)。Drive 项按当帧最终净变化结算：Drive 损失为正，Drive regen 为负，因此低频 poke 后让对手回满会抵消此前收益。CH/PC 对当次 Drive reward 分别乘 1.1/1.2；mash interrupt、throw tech、punish/PC、counter-DI、换边和 ODDP pressure escape 只在成功时给予 event bonus。普通 block、ODDP/DI/mash attempt、combo、cancel、spacing 和特定招式没有额外奖励。

Regen 只在 idle、guard、backwalk 或 forward walk 且 cooldown 已结束时发生；hit、block 和 Drive expenditure 会重置 cooldown。连续 forward walk 达到 11F 后默认使用 1.5 倍恢复率。持续 pressure 因反复重置 cooldown 而能阻止恢复。

Phase 1 fixed-bot PPO 入口是 `train_fixed_bot.py`，使用 MaskablePPO 消费动态 action mask。项目 `.venv` 已安装 `stable-baselines3==2.9.0`、`sb3-contrib==2.9.0` 和 CUDA 版 `torch==2.9.1+cu128`：

```powershell
.\.venv\python.exe -m pip install -e .[ppo]
.\.venv\python.exe Ken43\run_pairwise_payoffs.py --episodes 20
.\.venv\python.exe Ken43\train_fixed_bot.py --bot heuristic --timesteps 100000
```

长训练可以用 `--episodes` 精确限制完成的 episode 数；`--timesteps` 仍作为总步数安全上限。训练会按 episode 周期保存中间 checkpoint：

```powershell
.\.venv\python.exe Ken43\train_fixed_bot.py --bot heuristic --episodes 1500 --timesteps 900000 --device cuda --num-envs 4 --checkpoint-every-episodes 500
```

`--role attacker`（默认）训练 `player_0`，`--role defender` 训练 `player_1`。防守 curriculum 的第一条并行入口使用状态化进攻宏：

```powershell
.\.venv\python.exe Ken43\train_fixed_bot.py --role defender --bot heuristic_offense --episodes 1500 --timesteps 900000 --device cuda --num-envs 4 --checkpoint-every-episodes 500
```

`HeuristicOffenseBot` 每次选择并完成一个 `safe_jump`、`strike_throw`、`stagger` 或 `commitment` 宏；具体序列包括 32–34F jHP、投/打、三条 light chain、MP pressure、Jinrai cancel/派生和 DI conditioning。宏执行期间不会逐帧重新抽取无关动作。

当前 PPO 默认使用 `gamma=0.999`、`gae_lambda=0.97`，以覆盖 600F interaction 的长时信用分配。`--num-envs 4` 使用 Windows spawn 子进程并行采样；在受限 sandbox 中运行时需要允许创建本地 multiprocessing pipe。

scripted policies 为 AlwaysBlock、AlwaysThrow、AlwaysMash、AlwaysODDP、AlwaysDI、AlwaysJinrai、BackdashHeavy、Random 和 Mixed。训练 CLI 推荐显式使用 `--bot heuristic`；`--bot mixed` 与 `MixedBot` 仅作为旧命令的兼容名称，实际都使用带短期记忆的 `HeuristicDefenderBot`，不再每个 decision 独立随机抽动作。它在起身、DI、light pressure、中重攻击接必杀、后走和距离外局面分别进入稳定决策分支，并保存最近 5 个防守事件、连续轻攻击次数、normal-to-special 次数、counter-DI 次数和行为 bias。成功 family 会按 67.5% persistence 在后续同类局面复用。启发式 bot 可读取 resolver 上下文，但 learned policy 的 32 维 observation 没有增加 action identity 或内部状态泄漏。

旧的 20 局/有序组合矩阵保存在 `Ken43/results/pairwise_payoffs.json`；它是在随机 MixedBot 语义下生成的历史结果，不应用来代表当前 heuristic 对手。`RandomBot` 仍保留作随机 rollout 和 API 压力测试，但不再是 `--bot mixed` 的主要训练行为。

当前 600F、新 disengage/regen/reward 规则的 20 局/有序组合矩阵保存在 `Ken43/results/pairwise_payoffs_600f_v2.json`，没有发现 universal dominant policy。405 局诊断保存在 `Ken43/results/v1_diagnostics_600f.json`：平均长度 352.73F，173 局（42.72%）达到 hard cap，其中 `hard_stall=5`、`soft_loop=1`、`live_combat=167`。timeout 最近 120F 的平均 fallback-guard 占比为 94%，说明当前超时主要集中于 scripted bot 的保守 guard 行为，而不是无资源变化的纯停滞。

每步 `info` 提供 `disengaged`、`disengage_cause` 和 `diagnostic_events`；episode 结束时额外提供 `episode_diagnostics`。摘要包含终止原因、动作频率、schedule 目标/噪声/实际 delay、Jinrai 强度/派生/delay、OD DP 与 DI outcome、frame-kill、Drive 来源、reward 分解、三个 `frames_since_*` 指标，以及最近 120F trace。timeout 会按资源净变化、接触、动作、位移、regen 与 guard/light 比例分类为 `hard_stall`、`soft_loop` 或 `live_combat`。批量诊断会把每局完整摘要写入压缩 JSONL：`Ken43/results/v1_episode_diagnostics_600f.jsonl.gz`。

首个 always-block checkpoint 使用旧 observation/action schema；其确定性 100-episode 历史评估回报为 540，策略主要执行 `5HP(non-hold2) -> M Shoryuken cancel` 来削减对手 Drive，并在其余时间站防。该 checkpoint 只证明旧训练链路可工作，不能与当前参数化动作空间的结果直接比较。

当前 Drive-only 动作空间已改为扁平 `Discrete(368)`。所有旧 MultiDiscrete checkpoint（包括 500/1000/1500 局模型）与新 action head 不兼容，只保留作历史评估，禁止恢复训练。正式 self-play 前仍应先增加多个 seed 的 payoff 置信区间。

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

- 验证固定第 43F 起身、首帧投无敌、wakeup reversal、safe jump 和 meaty 使用同一时间轴；
- 验证 DI 可被反 DI、足够 hit count 的多段技和投有效处理；
- 验证简化多段技的 hit count、聚合伤害与最终 knockdown；
- 验证 simultaneous interaction、throw/strike、invulnerability、spacing 和取消窗口；
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
  ken43_combat_effects_v0.json
    Drive、CH/PC、scaling、多段与 DI reward profile
  ken43/
    data.py
      JSON loader 与 FrameData
    frame_math.py
      active、advantage、cancel、branch、safe-jump 和执行误差计算
    spacing.py
      posture range、block spacing、backwalk 和连续距离演化
    resolver.py
      对称 CombatState、FrameInput/FrameEvent、MovementProfile 与 resolve_frame
    parallel_env.py
      PettingZoo ParallelEnv、数值 observation、命令目录与动态 mask
    gym_env.py / bots.py
      单智能体 Gymnasium facade 与 fixed-bot baselines
    env.py
      兼容旧 FrameState，并将 step/resolve_frame 接到统一 resolver
  tests/
    test_validation_cases.py
      时序与系统规则回归
    test_spacing_validation_cases.py
      12 项射程与 spacing 回归
    test_frame_resolver.py
      同步接触、trade、投、无敌、armor、cancel、branch 和 movement 回归
  run_validations.py
    输出时序命题
  run_oki_scenarios.py
    输出放帧误差场景的精确成功率
  run_spacing_validations.py
    输出射程与 spacing 的完整诊断字段
  run_training_smoke.py
    运行固定种子的 masked random rollout
  train_fixed_bot.py
    对 fixed bot 训练 MaskablePPO 的 Phase 1 入口
```

## 验证命令

从仓库根目录 `D:\github\SokuRL` 运行，并始终使用项目 `.venv`：

```powershell
.\.venv\python.exe -m pytest Ken43\tests -q
.\.venv\python.exe Ken43\run_validations.py
.\.venv\python.exe Ken43\run_oki_scenarios.py
.\.venv\python.exe Ken43\run_spacing_validations.py
.\.venv\python.exe Ken43\run_training_smoke.py --episodes 10 --seed 7
```

最近一次验证结果为 `77 passed`，包括 Gymnasium checker、PettingZoo `parallel_api_test` 与随机 rollout invariant。

## 研究成功标准

项目最终希望依次回答三个问题：

1. **Can the agent remember?** 历史输入是否在严格对照下提高表现？
2. **Can the agent adapt?** 面对不同 latent opponent type，策略是否产生可重复的条件变化？
3. **Can the agent probe?** 策略是否愿意承担局部成本来获得信息，并在后续 match 中收回收益？

只有第三点成立时，Ken43 才真正从“有记忆的 PPO bot”成为 active opponent modeling benchmark。
