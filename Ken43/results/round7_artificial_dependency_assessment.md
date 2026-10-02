# Round 7 Artificial-Dependency Assessment

## Result

Round 7 trained the attacker for 1,500 episodes against `HeuristicDefenderBot`
after correcting counter-DI ownership. Training stopped normally and no later round
was launched.

Against the training opponent, the final checkpoint looked strong:

| Evaluation | Success | Failure | Timeout |
| --- | ---: | ---: | ---: |
| Deterministic, 200 episodes | 93.5% | 5.0% | 1.5% |
| Stochastic, 200 episodes | 79.5% | 13.5% | 7.0% |

The deterministic policy was highly concentrated: 600 `jHP`, 779 `2HP`, 659 H
Jinrai roots, 394 Senka branches, and only 48 Gorai branches. It used no meaningful
throw or DI mixup. Most wins were Drive breaks (187/200).

## Transfer Test

| Defender | Main result over 100 deterministic episodes |
| --- | --- |
| AlwaysBlock | 100 Drive breaks |
| Heuristic/Mixed | 94 Drive breaks |
| AlwaysMash | 100 timeouts |
| AlwaysThrow | 100 timeouts |
| AlwaysODDP | 100 pressure escapes |

The policy therefore learned a narrow best response to the training tree rather
than a generally competent offensive policy. The repeated cycle of patching one
heuristic exploit and observing a new fixed route is further evidence that the
opponent tree is currently supplying too much of the task definition.

## Dependency Classification

### Necessary simulator design

- Frame data, range, spacing, stun, cancel, armor, throw invulnerability, and
  counter-DI rules.
- Legal-action masks and command timing semantics.
- Explicit scope decisions such as aggregate spacing and the omitted full Drive
  system, provided these remain documented limitations.

These define the game being simulated and should remain hand-authored and tested.

### Excessive training design

- A single stateful heuristic opponent is the sole source of defensive behavior.
- Hand-tuned response probabilities determine which offensive route becomes best.
- Fixed `+43`, distance 70, corner state, and Ken mirror permit deterministic
  frame scripts.
- Drive-only survival makes safe block pressure disproportionately valuable and
  removes the HP tradeoff of risky pressure.
- Immediate interaction termination after OD-DP escape prevents learning the
  later punish/reset consequences of failed or blocked reversals.
- Repeated heuristic patches act as manual curriculum updates and move the target
  distribution after every exploit is found.

### Reward dependence

The final route is consistent with the stated Drive-only objective: repeated safe
pressure beats blocking and ends in Drive break. The larger problem is not an
obvious reward-sign error; it is that one opponent and one initial state expose a
single low-variance way to collect that reward. Timeout shaping suppresses some
stalling but cannot create missing strategic diversity.

## Recommendation

Do not continue expanding the heuristic tree as the primary trainer. Keep scripted
bots as regression tests and curriculum members, then rebuild training around:

1. An opponent population containing simple archetypes, parameterized mixtures,
   past checkpoints, and eventually self-play policies.
2. Randomized initial advantage and spacing, at least `{26, 38, 43}` and a small
   valid distance range.
3. Held-out-opponent gates as the main promotion criterion; training-opponent win
   rate should be secondary.
4. A minimal zero-sum Drive/terminal reward, with timeout shaping retained only as
   an anti-stall constraint.
5. Separate correctness tests for resolver rules from behavioral assumptions in
   heuristic policies.

The current environment is useful as a deterministic mechanics testbed, but its
current RL result should be interpreted as heuristic exploitation, not learned
general corner offense.
