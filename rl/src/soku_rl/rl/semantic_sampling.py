"""Finite, repeat-capped overlapping event sampling, not action-class quotas."""
import numpy as np


STRATA = ('movement_neutral', 'attack_skill', 'cancel_link', 'defense_reaction')
RATIOS = np.array([.30, .30, .25, .15])


def sampling_membership(data):
    roles = np.asarray(data['supervision_strata'], bool)
    eligible = np.asarray(data.get('training_eligible', np.ones(len(roles), bool)))
    if eligible.shape != (len(roles),) or eligible.dtype != np.bool_:
        raise ValueError('boolean training eligibility must align with event rows')
    return roles & eligible[:, None]


def choice_weight_mask(data):
    known = np.asarray(data['target_action_mask'], bool)
    eligible = np.asarray(data.get('training_eligible', np.ones(len(known), bool)))
    if eligible.shape != known.shape or eligible.dtype != np.bool_:
        raise ValueError('boolean training eligibility must align with action labels')
    return known & eligible


def stratified_order(strata, replays, chains, rng, draws, *, row_cap=2,
                     replay_cap=256, chain_cap=32):
    strata = np.asarray(strata, bool)
    replays = np.asarray(replays)
    chains = np.asarray(chains)
    if (strata.ndim != 2 or strata.shape[1] != 4 or len(replays) != len(strata)
            or len(chains) != len(strata) or min(draws, row_cap, replay_cap, chain_cap) < 1):
        raise ValueError('explicit positive sampling limits and four event strata required')
    usage = np.zeros(len(strata), int)
    rep_values, rep_index = np.unique(replays, return_inverse=True)
    chain_keys = np.stack((replays, chains), -1)
    _, chain_index = np.unique(chain_keys, axis=0, return_inverse=True)
    rep_usage = np.zeros(len(rep_values), int)
    chain_usage = np.zeros(chain_index.max(initial=-1) + 1, int)
    assigned = np.zeros(4, int)
    selected, roles = [], []
    pools = [np.flatnonzero(strata[:, k]) for k in range(4)]
    depleted = set()
    while len(selected) < draws:
        viable = []
        eligible = ((usage < row_cap) & (rep_usage[rep_index] < replay_cap) &
                    ((chains < 0) | (chain_usage[chain_index] < chain_cap)))
        for pool in pools:
            viable.append(pool[eligible[pool]])
        available = np.array([len(p) > 0 for p in viable])
        if not available.any():
            break
        depleted.update(np.flatnonzero(~available).tolist())
        effective = RATIOS * available
        effective /= effective.sum()
        # Deficit scheduling gives stable quotas; depleted strata redistribute.
        deficit = (len(selected) + 1) * effective - assigned
        deficit[~available] = -np.inf
        k = int(deficit.argmax())
        # Uniform over REP first, then chain group, then row prevents long cases
        # or highly repeated chain rows from monopolizing a role's draws.
        pool = np.asarray(viable[k])
        reps = np.unique(replays[pool])
        rep = int(rng.choice(reps))
        pool = pool[replays[pool] == rep]
        groups = np.unique(chains[pool])
        group = rng.choice(groups)
        i = int(rng.choice(pool[chains[pool] == group]))
        selected.append(i)
        roles.append(k)
        assigned[k] += 1
        usage[i] += 1
        rep_usage[rep_index[i]] += 1
        if chains[i] >= 0:
            chain_usage[chain_index[i]] += 1
    report = dict(requested=draws, actual=len(selected),
        status='PASS' if len(selected) == draws else 'BLOCKED_CAPACITY',
        requested_ratios=RATIOS.tolist(), role_draws=dict(zip(STRATA, map(int, assigned))),
        eligible_rows=dict(zip(STRATA, map(int, strata.sum(0)))),
        depleted_strata=[STRATA[k] for k in sorted(depleted)], unique_rows=int((usage > 0).sum()),
        max_row_draws=int(usage.max(initial=0)), max_replay_draws=int(rep_usage.max(initial=0)),
        max_chain_draws=int(chain_usage.max(initial=0)),
        caps=dict(row=row_cap, replay=replay_cap, chain=chain_cap))
    return np.asarray(selected, np.int64), np.asarray(roles, np.int64), report
