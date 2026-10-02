# Ken43 项目复盘

## 最初的想法

Ken43 起初想回答一个比“能不能训练出格斗游戏 AI”更窄的问题：在重复发生的角落攻防里，智能体能否观察对手习惯，逐渐调整打法，并在必要时主动试探。

我们关心的不是 Ken 会不会背一套伤害最高的连段，而是下面三件事：

1. 它能否记住对手此前的防守选择。
2. 它能否针对爱抢招、爱拆投、爱升龙或偏好防守的对手改变决策。
3. 它是否愿意牺牲一点眼前收益，用一次投、shimmy 或延迟攻击换取信息，再在后续回合利用这些信息。

完整 SF6 对这个问题太大。角色、系统资源、浮空连段、弹道、位置、数百种动作和视觉输入会把“对手建模”淹没在控制与感知问题里。Ken43 因此选择了一个可解释的 microgame：Ken 对 Ken，防守方位于版边，从固定压起身局面开始，只保留会影响投打、抢招、确反、跳跃、迅雷、升龙和 DI 的机制。

这个缩小不是为了复刻整局 SF6，而是为了得到一个每次决策都能解释的重复博弈。如果策略突然增加投、延迟攻击或 reversal bait，我们希望能沿着事件记录找到原因。

## 为什么这样建模

### 帧精确，而不是结果表驱动

第一版工作从帧数计算器开始。它能回答某个固定命题，例如晚 active 命中后的优势、取消 gap、安全跳是否接触，但不能处理双方同时行动。

后来加入统一的 `resolve_frame()`，让双方在同一个 global frame 提交输入，再按固定顺序推进动作、检查 active/armor/invulnerability、结算接触、更新硬直和距离。这样才可以表示抢招、投无敌、反 DI、空挥和同时接触。

我们保留了几个重要语义：

- startup、active、recovery 使用 1-based 动作帧；
- 晚 active 会增加 raw advantage，但不会改变取消连段原有的时序；
- `gap < 0` 才是真联防，`gap = 0` 只是没有自由帧；
- 取消与派生拥有各自明确的子动作起始帧；
- range check 发生在 spacing transition 之前；
- 超出射程可以出招，但会 whiff。

这些规则后来证明很重要。RL 会利用任何一帧的错误，手工验算看似无害的偏差在训练中可能变成稳定 exploit。

### 距离使用测量值，但不重建完整动画

环境使用姿态相关 hurtbox、effective range 和接触后的 aggregate spacing。距离更新大体采用：

\[
d'=\max(D_{min},d+\Delta)
\]

我们没有拟合每个动作的逐帧动画位移，因为当前研究只需要判断这一招能否接触、接触后距离是多少、下一招是否还在射程。只有中途位移会改变双方当帧胜负时，逐帧 trajectory 才值得加入。

这项取舍基本成立。后来的主要问题并不来自 aggregate spacing，而来自训练分布、终止规则和少数 resolver 错误。

### 特殊技指令被抽象成一次决策

236K、623P 等指令不模拟方向输入过程。agent 提交动作后，下一帧进入动作 frame 1。需要主动放帧时，策略使用 `schedule(delay, action)`；环境只在提交时采样一次 ±2F 执行误差，避免 agent 每帧 noop 来获得不现实的精确控制。

空挥 5MK、5HK、2HK 等 frame-kill 被保留，因为它们本来就是格斗游戏 timing 的一部分。动作空间后来改为 368 个完整命令的 flat discrete space，使 action mask 可以直接表达一条命令是否合法。

### 从 HP 转向 Drive-only interaction

项目中途删除 HP，把 Drive 同时当作防御资源和 interaction 的主要目标。一次 episode 不再代表完整 round，而是一段角落攻防。投中、DI 上墙、拆投、换边、主动脱压、Drive break 或 600F 上限都可以结束 interaction。

reward 主要使用双方净 Drive 变化：

\[
r_i=\Delta D_{opp}-\Delta D_{self}+event\ bonus
\]

普通 block、combo 数、cancel 数和特定招式没有额外奖励。后期只保留少量明确事件 bonus，并加入 Drive regen、regen cooldown、360F 后逐渐增加的时间成本和 timeout penalty。

这个版本比早期 HP/Drive 混合 reward 更容易诊断，但它也改变了问题本身。安全削 Drive 的价值被放大，高风险动作没有掉血代价；防守方甚至可以通过反攻打空进攻方 Drive 来获得“防守成功”。最终它更像一个 Drive-only pressure benchmark，而不是完整 SF6 攻防。

## 调试过程

### 从规则验算到同步 resolver

早期大量工作用于封闭基础语义：取消发生在哪一帧、blockstun 是否包含接触帧、true blockstring 如何定义、迅雷派生如何使用相对帧、jHP 在安全跳轨迹中的接触时点，以及站姿和蹲姿为何会让龙尾命中不同 active。

这一阶段的价值很明确。固定命题可以作为回归测试，帮助我们在改 resolver 时发现 off-by-one，而不是靠训练曲线猜规则是否正确。

### 动作 mask 曾经污染训练

旧动作空间使用 factorized `MultiDiscrete`。各维 mask 分别合法，不代表组合后的完整命令合法，于是会出现 scheduled guard 等无意义组合。环境把这些非法组合转换为 noop，PPO 随后开始适应这种接口错误。

我们最终把动作空间改成完整命令的 flat discrete 表。旧 checkpoint 因 action head 不兼容而全部作废。这次修改说明，训练链路“能跑”不等于 agent 实际拿到了设计者以为的动作空间。

### timeout 惩罚没有直接解决等待

最初只在 600F 结束时处罚 timeout。长 horizon 下，这个终局信号经过折扣后很弱；SB3 还会对 truncation 做 value bootstrap，部分抵消惩罚。策略可以先获得 Drive 收益，再一直防守到结束，最终仍有正回报。

后来我们关闭该 bootstrap，提高 `gamma` 和 `GAE lambda`，并在 360F 后加入二次增长的时间成本。超时有所减少，但问题没有消失。对手分布不变时，时间惩罚只会改变“等待有多贵”，不会告诉策略如何处理从未学过的 AlwaysMash 或 AlwaysThrow。最终模型面对这两类对手仍然 100% timeout。

### 随机对手产生了噪声，不是有意义的博弈

早期 MixedBot 每次决策都重新随机动作。它会在很短时间内混用投、OD DP、DI、5LP 和迅雷，行为缺少持续意图。策略学到的是如何在高噪声动作流里找平均收益，而不是如何识别一种对手习惯。

我们随后把随机对手改成状态化 heuristic decision tree。进攻方拥有 safe jump、strike/throw、stagger 和 commitment 宏；防守方拥有 block、tech、mash、OD DP、DI、anti-air、确反和换边逻辑。这个变化提高了可解释性，但也埋下了最终停止训练的原因：单一决策树开始同时充当训练分布、课程和考试标准。

### 防守方训练反复迁移到决策树的新盲点

防守方最初很快学会以 jHP 和跳跃换边，确定性评估接近 99% 成功。修复 movement 与受击交互后，它转向 jHP 和 DI。加入更多自适应反应后，防守方又开始高频反攻并打空进攻方 Drive。

当进攻 heuristic 补上对所有跳跃的反应后，防守成功率从 99.3% 降到 7.7%，83.7% 的 episode 超时。原来有效的跳跃路线一旦被封住，策略几乎没有备用方案。随后加入 repeated-poke 反制并重新训练，成功率又恢复到 85.7%，但行为转向 DI、counter-DI 和另一套换边路线。

这段过程没有显示出逐步形成的稳健防守。每次我们补一个对手分支，最优策略就整体换轨。

### 进攻方训练形成 exploit-patch 循环

进攻方也经历了同样过程。加入负帧与距离相关的确反、抢招、升龙对空和中断后逃角后，旧策略性能下降。Round 5 收敛到 `jHP/5HP -> DI`；我们强化防守方的 DI 反应后，冻结策略的成功率从 83.3% 降到 63.3%，失败率从 6.7% 升到 36.7%。

Round 6 很快找到替代路线 `jHP -> 2MP -> DI`。300 局评估中有 209 局以 DI 上墙结束。继续追踪事件后发现，这一次不只是 heuristic 漏洞，而是 resolver 错误：防守方启动 counter-DI 时，事件被记给先手 DI，随后系统仍继续结算先手命中和 wall splat。

修正 counter-DI 归属、中断先手动作并跳过原命中结算后，冻结 Round 6 策略从 78% 成功直接降到 0%。这说明 RL 对机制错误非常敏感，也说明高胜率本身不能证明策略合理。

Round 7 在正确的 counter-DI 规则下重新训练。它不再滥用 DI，而是集中使用 `jHP -> 2HP -> H Jinrai -> Senka/Gorai`。对训练 heuristic 的确定性成功率达到 93.5%，随机采样成功率为 79.5%。单看这组数字，训练似乎成功了。

## 最终结果

Round 7 的 held-out opponent 测试否定了“通用进攻已经学成”的解释。

| 对手 | 100 局确定性结果 |
| --- | --- |
| AlwaysBlock | 100 次打空防守方 Drive |
| 训练用 Heuristic/Mixed | 94 次打空防守方 Drive |
| AlwaysMash | 100 次 timeout |
| AlwaysThrow | 100 次 timeout |
| AlwaysODDP | 100 次被脱压 |

动作分布也很集中。200 局中出现 600 次 jHP、779 次 2HP、659 次 H Jinrai 和 394 次 Senka，几乎没有形成有意义的投打混合。随机采样时仍以同一路线为主。

所以这一轮得到的不是稳健的角落进攻策略，而是训练 heuristic 的高质量 best response。对手一换，策略要么停止进攻，要么立即输给固定 reversal。

## 为什么停止当前路线

继续给 heuristic decision tree 增加分支，会让我们进入没有终点的人工补洞：策略找到漏洞，我们加规则封住，再训练出下一条漏洞。这样可以不断提高对当前 bot 的胜率，却无法说明能力来自 RL，还是来自我们逐步写进对手树里的知识。

固定 +43、距离 70、Ken mirror 和固定资源也让策略容易学习开局脚本。Drive-only reward 与立即终止进一步放大了少数安全路线。只调整 timeout penalty 或动作概率无法解决这些问题。

因此我们停止了当前训练路线，没有继续启动下一轮。这里放弃的是“单一 heuristic 对手 + 固定场景 + PPO best response”这套训练方案，不是整个 Ken43 仿真器。

## 仍然保留的成果

Ken43 留下了几项可复用结果：

- 一套帧精确、双方同步的 resolver；
- 已校准的帧数、距离、spacing、取消和派生数据；
- Drive、投、跳跃、升龙、迅雷和 DI 的最小交互闭环；
- 可以定位动作、接触、资源和 terminal 原因的事件日志；
- 一组覆盖 timing、range、spacing 和 combat effect 的回归测试；
- 多个失败策略，可作为未来 population 或 exploit 测试样本。

这些内容足以把环境保留为 mechanics baseline，也能用于验证 RL 是否会发现 resolver 缺陷。

## 如果以后重新开始

下一次训练不应继续扩展同一棵 heuristic tree。更合理的起点是 opponent population：参数化的 block、mash、throw、OD DP、DI 对手，历史 checkpoint，以及逐步加入的 self-play snapshot。训练和评估必须分开，至少保留一组从未进入训练分布的 held-out opponents。

初始优势应在 26、38、43 之间变化，距离和 Drive 也应在合法范围内随机。模型晋级看跨对手平均表现和最差分位，而不是对训练 bot 的单一胜率。

等 memoryless policy 能在这套分布上稳定泛化，再加入 best-of-3、history encoder 和主动 probing。否则 recurrent policy 只会更有效地识别人工决策树，而不会回答最初关于对手建模的问题。

Ken43 最终证明了两件事。帧精确 microgame 足以让 RL 找到复杂的高收益路线，也足以让它暴露很难通过手工测试发现的规则错误。但要研究适应和试探，决定结果的不能是一棵由我们不断修补的对手树。
