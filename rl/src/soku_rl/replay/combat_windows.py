"""Sample selection over existing policy-visible data, never a native reader."""
import numpy as np


SELECTION_REASONS = ('attack_choice', 'object_mobility', 'object_context',
                     'observed_link', 'attack_continuation', 'defense_context')


def post_opening_selection(data, alignment, catalog, min_frame=300):
    """Keep full histories; select decisions using original pre-action time.

    Object presence is the existing visible token proxy, not proof of damaging,
    enemy-owned or grazeable projectiles. No contact outcome gates an attack.
    """
    if type(min_frame) is not int or min_frame < 0:
        raise ValueError('nonnegative original observation-frame threshold required')
    n = len(data['state'])
    if data['state'].shape != (n, 185) or len(alignment['samples']) != n:
        raise ValueError('aligned public185 state required')
    frames = np.array([s['observation_frame'] for s in alignment['samples']], np.int64)
    if (frames < 0).any() or (np.diff(frames) <= 0).any():
        raise ValueError('strict original observation chronology required')
    known = np.asarray(data['target_action_mask'], bool)
    labels = np.asarray(data['target_action'])
    if ((labels[known] < 0) | (labels[known] >= len(catalog))).any():
        raise ValueError('known command outside catalog')
    active = np.asarray(data['active_action'])
    if ((active < 0) | (active > len(catalog))).any():
        raise ValueError('active command outside catalog/null contract')
    attack_catalog = np.array([not c['continuous'] for c in catalog], bool)
    attack = known & attack_catalog[labels.clip(0, len(catalog)-1)]
    mobility = known & ~attack
    active_attack = ((active < len(catalog)) &
                     attack_catalog[active.clip(0, len(catalog)-1)])
    tokens = data['state'][:, 20:180].reshape(n, 16, 10)
    present = tokens[:, :, 0] > 0
    owner = tokens[:, :, 9]
    if not np.isfinite(tokens).all() or ((owner[present] != 0) & (owner[present] != 1)).any():
        raise ValueError('invalid existing visible object token')
    own = (present & (owner == 0)).sum(-1)
    enemy = (present & (owner == 1)).sum(-1)
    objects = present.any(-1)
    usable = known | data['target_termination_mask'] | data['target_sequence_mask'].any(-1)
    reasons = np.stack((attack, objects & mobility, objects & ~known,
        data['supervision_strata'][:, 2], active_attack & data['target_termination_mask'],
        data['supervision_strata'][:, 3]), -1).astype(bool)
    after = frames >= min_frame
    reasons &= (after & usable)[:, None]
    eligible = reasons.any(-1)
    if not np.all(eligible[after & attack]):
        raise AssertionError('all reliable post-opening attacks must remain eligible')
    metadata = dict(training_eligible=eligible, selection_reasons=reasons,
        observation_frame=frames, visible_object_context=objects,
        visible_own_object_count=own.astype(np.int16), visible_enemy_object_count=enemy.astype(np.int16))
    names = np.array([c['name'] for c in catalog])
    neutral = known & (names[labels.clip(0, len(catalog)-1)] == '5move')
    stats = dict(source_rows=n, after_opening_rows=int(after.sum()), eligible_rows=int(eligible.sum()),
        excluded_opening_rows=int((~after).sum()),
        eligible_attack_labels=int((eligible & attack).sum()),
        eligible_continuous_labels=int((eligible & mobility).sum()),
        object_continuous_labels=int((eligible & objects & mobility).sum()),
        object_direction_or_D_labels=int((eligible & objects & mobility & ~neutral).sum()),
        object_neutral_or_release_labels=int((eligible & objects & neutral).sum()),
        object_mobility_without_active_attack=int((eligible & objects & mobility & ~active_attack).sum()),
        enemy_object_mobility_without_active_attack=int((eligible & (enemy > 0) & mobility & ~active_attack).sum()),
        reasons={k:int(reasons[:, i].sum()) for i,k in enumerate(SELECTION_REASONS)})
    return metadata, stats
