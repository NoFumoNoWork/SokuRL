# Ken43 环境人工设计依赖性详细评估

## 1. 结论摘要

当前 Ken43 已经可以作为一个可重复、可诊断的帧精确 microgame 测试床，
但尚不能把训练结果解释为“智能体学会了通用的角落攻防”。现阶段得到的策略主要是：

1. 在人工规定的帧数、距离、终止条件和 Drive-only 目标内寻找最优路线；
2. 对单一 heuristic opponent 的决策树学习窄化 best response；
3. 在 resolver 存在错误时迅速利用错误；
4. 当我们修改 opponent tree 或修正规则后，再迁移到下一条固定路线。

因此，环境对人工设计的依赖可以分为两部分：

- **必要且合理的依赖**：帧数、距离、取消、硬直、投无敌、DI armor、counter-DI、
  action mask 等仿真规则。这些本来就必须由人定义并通过测试保证正确。
- **过强且影响研究结论的依赖**：单一 heuristic opponent、固定 +43/距离 70、
  Drive-only 终止与奖励、人工设定的响应概率、基于 resolver 内部状态的决策树，
  以及针对每个新 exploit 继续添加分支的训练方式。

总体判断：**仿真器层可继续保留，当前训练层需要重构。**

## 2. 训练过程中的证据

### 2.1 早期训练：奖励和动作接口本身就在塑造策略

最初训练暴露了两个与 PPO 无关、但会直接改变学习目标的问题：

- 旧的 factorized `MultiDiscrete` mask 无法表达联合合法性，非法组合会被转换成
  `noop`。这使策略学习到的部分行为其实是在适应接口错误。改成 368 个完整命令的
  flat discrete action space 后，旧 checkpoint 不再兼容。
- 单独在第 600F 施加 timeout penalty 的长期信用很弱。旧参数下，策略可以先获得
  Drive reward，再超时保留正回报。之后加入 360F 后开始的二次时间成本、终局惩罚、
  `gamma=0.999`、`GAE lambda=0.97`，并关闭 timeout value bootstrap。

这些修改是必要的工程修正，但也说明“策略为什么行动”高度依赖 reward horizon 和
action encoding。时间惩罚减少了部分纯等待，却没有自动产生通用进攻：最终策略对
AlwaysMash 和 AlwaysThrow 仍然可以 100% 超时。

### 2.2 防守方训练：每次补 opponent tree，优势路线都会整体迁移

防守方以人工设计的 `HeuristicOffenseBot` 为对手。该 bot 包含 safe jump、strike/throw、
stagger、commitment 等宏动作，并逐步加入对 jump、DI、poke、负帧和确反的响应。

| 阶段 | 确定性结果 | 主要行为/终局 | 说明 |
| --- | ---: | --- | --- |
| 初始 1500 | 99% 防守成功 | 99/100 side switch，主要使用 jHP | 学到单一换边路线 |
| movement fix 后 3000 | 87.7% 成功 | 219 side switch；大量 jHP、DI | 修正移动/受击交互后仍依赖跳与 DI |
| adaptive reactions 后 3000 | 99.3% 成功 | 289/300 attacker Drive break；3898 次 jHP | 防守方转为高频反攻并打空进攻方 Drive |
| offense 对所有 jump 做反应后 | 7.7% 成功，83.7% timeout | 4207 次 5MP，251/300 timeout | 原有跳跃 exploit 被封后策略几乎失效 |
| offense 加 repeated-poke 反制后 | 85.7% 成功 | 113 side switch、92 counter-DI、44 attacker Drive break | 新策略转向 DI、反 DI 和新的换边路线 |

最关键的证据是：仅仅把进攻 heuristic 的跳跃反应补全，冻结评估分布就让防守方从
99.3% 成功降到 7.7%。这不是小幅泛化下降，而是策略目标整体改变。后续恢复到 85.7%
也不是学成了通用防守，而是再次找到新树上的高收益分支。

此外，“attacker Drive break”被计为防守成功，说明 Drive-only microgame 中的防守方
经常不是通过 block、tech、punish 或逃角获胜，而是主动反攻并打空对手 Drive。
这在当前 benchmark 定义下合法，但不能等同于完整 SF6 中的防守能力。

### 2.3 进攻方训练：人工补洞形成手工 adversarial curriculum

进攻方改为对 `HeuristicDefenderBot` 训练后，出现了更清晰的 exploit-patch 循环：

1. 早期策略偏向 jHP、投和特定 pressure route。
2. 加入负帧/距离确反、抢招、升龙对空和中断后换边，旧策略表现下降。
3. Round 5 收敛为 `jHP/5HP -> DI`。调整 defender 的 DI 响应和 jHP 后行为后，
   冻结策略从 83.3% 成功、6.7% 失败变为 63.3% 成功、36.7% 失败。
4. Round 6 又收敛为 `jHP -> 2MP -> DI`，300 局有 209 次 DI wall splat。
5. 进一步检查发现这次不只是 heuristic 漏洞，而是 resolver 把 counter-DI 记给了
   先手 DI，并继续结算先手命中和 wall splat。修正后，冻结 Round 6 策略从 78%
   成功直接降到 0%；50 局中 39 次 counter-DI escape、11 次 OD-DP escape。
6. Round 7 在修正规则后改为 `jHP -> 2HP -> H Jinrai -> Senka/Gorai`，对训练
   heuristic 的确定性成功率达到 93.5%，随机采样成功率 79.5%。

这个过程说明 RL 确实能快速发现环境中的因果结构，但目前发现的首先是“当前环境和
当前对手的最窄高收益通路”，而不是自动获得投打、shimmy、frame trap、bait 等完整
混合策略。

### 2.4 Round 7 的 held-out opponent 测试

最终 checkpoint 对训练 opponent 很强，但迁移结果如下：

| 对手 | 100 局确定性结果 |
| --- | --- |
| AlwaysBlock | 100 次 defender Drive break |
| Heuristic/Mixed | 94 次 defender Drive break |
| AlwaysMash | 100 次 timeout |
| AlwaysThrow | 100 次 timeout |
| AlwaysODDP | 100 次 pressure escape |

动作也高度集中：200 局确定性评估中出现 600 次 jHP、779 次 2HP、659 次 H Jinrai、
394 次 Senka，几乎没有有意义的 throw/DI mixup。随机采样评估仍保持同一主路线。

因此 93.5% 的训练对手胜率主要测量的是“是否找到该决策树的 best response”，不能
作为通用角落进攻能力的证据。

## 3. 人工依赖的来源

### 3.1 核心规则依赖：高，但合理

必须手工维护：

- frame data、active 区间、hit/block stun；
- range、posture hurtbox、aggregate spacing；
- cancel/branch、throw invulnerability、armor、counter-DI；
- schedule noise、识别延迟和合法动作 mask。

这部分不是问题。问题在于任何错误都会成为高强度训练中的最优 exploit，所以应先由
规则测试和 scenario test 封闭，而不是靠 heuristic opponent 补偿。

### 3.2 场景分布依赖：非常高

当前长期固定：

- `initial_advantage=43`；
- `initial_distance=70`；
- Ken mirror；
- 防守方在角落；
- 固定 Drive 初值；
- interaction 而非完整 round。

这允许策略把开局压缩成确定性的 frame script。即使 `schedule` 有 ±2F 误差，策略仍可
围绕一个 setup 学习固定动作链。它尚未证明能迁移到 +26、+38、不同距离或不同资源。

### 3.3 对手依赖：非常高，是当前最大问题

两个 heuristic bot 都直接读取 `env.core.combat_state`、movement 和 action state，再用
人工延迟模拟可观察性。它们不是通过与 PPO 相同的 observation interface 决策，因此
仍带有特权状态和人工知识。

其宏动作概率、anti-air 概率、重复 poke 后 DI、jHP 后 guard、确反选择等，实际上定义了
训练分布。继续添加分支等价于手工设计一个 adversarial curriculum；每次修改都在改变
任务，而不是只提高同一任务的训练质量。

### 3.4 Reward 和 terminal 依赖：中高

当前主要 reward 是净 Drive 变化，加上 side switch、counter-DI、pressure escape 等
event bonus，以及后段时间成本和 timeout penalty。

优点是 reward 大体仍与目标一致，没有直接奖励某个招式或 combo。问题是：

- 删除 HP 后，高风险招式的“掉血代价”不存在；
- Drive 同时承担资源与生命目标，持续安全削 Drive 的价值被放大；
- 防守方可以通过打空攻击方 Drive 获得“防守成功”；
- interaction 在 throw、ODDP escape、side switch、counter-DI 等事件立即结束，后续局面
  价值由人工 terminal bonus 代替；
- timeout shaping 只能惩罚等待，不能提供缺失的 opponent diversity。

所以目前更准确的名称是“Drive-only corner interaction benchmark”，而不是完整角落攻防。

### 3.5 动作与观察接口依赖：中等

`immediate/schedule/react`、完整命令 flat action、mask 和 delayed recognition 都是合理抽象，
但也决定了 agent 能学到什么。早期 factorized mask 的失败已经证明接口错误会被策略吸收。

下一版应保证 scripted bot、PPO 和未来 recurrent policy 尽量使用同一 observation/action
contract；任何读取 resolver 内部字段的 bot 都只应用于测试，不应用作唯一训练对手。

## 4. 当前能与不能支持的研究结论

当前结果可以支持：

- resolver 中存在可被 RL 自动发现的机制和漏洞；
- PPO 能对确定的 opponent distribution 学出高收益 best response；
- 帧数、距离、取消、Drive 和 terminal 足以形成非平凡优化问题；
- event-level logging 能解释策略为何成功或失败。

当前结果不能支持：

- agent 已学会通用 Ken 角落攻防；
- agent 自然形成了稳健投打或接近均衡的混合策略；
- agent 能迁移到未训练 opponent、不同起攻优势或不同距离；
- agent 已具备 opponent modeling、跨局适应或主动 probing；
- heuristic-aware 胜率可以代表对人或完整 SF6 的表现。

## 5. 建议的重构顺序

### Phase A：冻结仿真器，停止追加策略补丁

1. 保留现有 resolver 和所有规则测试。
2. 把 heuristic bot 降级为 regression/curriculum member，不再作为唯一训练对手。
3. 为 DI、jump、throw、cancel、side switch 增加双向 scenario tests，确保事件 actor、target、
   terminal reason 和 reward 一致。

### Phase B：建立 opponent population

训练时混合抽样：

- AlwaysBlock、参数化 Mash/Throw/ODDP/DI；
- 不同概率和条件反应的有限状态 bot；
- 历史 attacker/defender checkpoints；
- 当前策略的 self-play snapshot。

至少保留一组完全不参加训练的 held-out bots。模型晋级应看最差分位或跨对手平均表现，
而不是训练 opponent 胜率。

### Phase C：随机化初始状态

第一步随机化：

- `initial_advantage in {26, 38, 43}`；
- 小范围合法 initial distance；
- 初始 Drive；
- defender posture/guard tendency。

不要用动作多样性 reward 强迫“看起来丰富”。如果一个动作在所有随机场景和 opponent
上都最优，它集中使用是合理的；应通过任务分布而不是奖励审美来产生多样性。

### Phase D：最小 reward ablation

比较：

1. 仅净 Drive delta + terminal；
2. 加 timeout anti-stall；
3. 再加少量明确事件 bonus。

用相同 seeds 和 opponent population 比较，确认 side-switch、counter-DI 等 bonus 是否
改变策略排序。若研究目标最终包含完整攻防，需重新决定是否恢复 HP 或至少加入风险代理。

### Phase E：再做 opponent modeling

只有 memoryless population baseline 能跨 opponent 泛化后，才适合进入 BO3、history encoder
和 active probing。否则 recurrent policy 很可能只是更好地识别某棵人工决策树，而不是
学习可发表的 opponent adaptation。

## 6. 最终判断

当前 Ken43 的**机制层人工设计依赖是必要的，训练层人工设计依赖则过强**。最大的风险
不是 reward 数值略有偏差，而是单一 heuristic tree 同时承担了课程、对手分布和评估标准。

建议保留当前代码作为 mechanics baseline，冻结本轮 checkpoint 作为“heuristic best-response
对照组”，然后从 opponent population + held-out evaluation 重新开始训练设计。这样后续才能
区分：策略是真的学会攻防，还是只学会了我们写进决策树里的漏洞与习惯。
