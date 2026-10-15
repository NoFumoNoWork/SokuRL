import pytest
from bridge_shared import RawFrameState
from soku_rl.env.combat_events import CombatEventTracker, require_resolved_contact


def test_snapshot_changes_are_not_contact_or_punish():
    raw = RawFrameState()
    raw.p1.hp = raw.p2.hp = 10000
    tracker = CombatEventTracker()
    assert tracker.observe(raw) == ()
    raw.frameId = 1
    raw.p2.hp = 9800
    raw.p1.comboCount, raw.p1.comboRate = 2, .85
    events = tracker.observe(raw)
    hp = next(event for event in events if event['kind'] == 'HP_CHANGE')
    assert (hp['before'], hp['after']) == (10000, 9800)
    assert hp['hit_or_block'] == 'UNKNOWN' and hp['punish_counter'] is None
    assert hp['damage_rate'] is None and hp['attacker_instance'] is None
    with pytest.raises(NotImplementedError):
        require_resolved_contact(hp)
    with pytest.raises(RuntimeError, match='consecutive'):
        tracker.observe(raw)
    raw.frameId = 0
    raw.p2.hp = 10000
    assert tracker.observe(raw) == ()


def test_action_change_does_not_mean_independent_attack():
    raw = RawFrameState()
    tracker = CombatEventTracker()
    tracker.observe(raw)
    raw.frameId, raw.p1.actionId = 1, 300
    events = tracker.observe(raw)
    assert len(events) == 1 and events[0]['kind'] == 'ACTION_STATE_CHANGE'
    assert events[0]['attacker_instance'] is None
