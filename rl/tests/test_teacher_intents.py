"""Intent lifecycle contracts; synthetic adapters are not original AI gameplay."""
from pathlib import Path

import pytest

from soku_rl.policy.god.query import QueryTeacher
from soku_rl.policy.hierarchical_control import Opportunity, command_catalog
from test_kickstarting import fixture_teacher, observation, spec


def boundary(*, source='PROVIDED_ENGINE_LEGAL_SET', forced=False, banned=None):
    return Opportunity(True, tuple(row['name'] != banned for row in command_catalog()),
                       'offline supplied legal set, NOT native validation', source, forced)


def actor(tmp_path, source):
    contract, teacher = fixture_teacher(tmp_path, source)
    return contract, QueryTeacher(contract, teacher.spec | {'mode': 'intent'}).spawn(17)


def warm(contract, teacher):
    assert teacher.query(observation(contract)).status == 'REACTION_WARMUP'
    assert teacher.query(observation(contract)).status == 'REACTION_WARMUP'


def test_reject_discards_coroutine_not_tactical_memory(tmp_path):
    contract, teacher = actor(tmp_path, '''
loaded = 1
local decisions = 0
function teacher_decide()
 decisions = decisions + 1
 if decisions > 2 then command('5A') else command('236B') end
 error('STALE_CONTINUATION_EXECUTED')
end
''')
    warm(contract, teacher)
    first = teacher.query(observation(contract), opportunity=boundary(), boundary_id=2)
    assert first.status == 'INTENT_PROPOSED' and first.intent == '236B'
    assert first.action is None and first.command is None and not first.input_legal
    assert 'fixture.ai:' in first.source
    teacher.intents.reject(first.intent_id)
    obs = observation(contract)
    obs[-6] = 1  # actual student chose A, not the suggested motion
    assert teacher.query(obs).status == 'WAIT_TRUSTED_DECISION_BOUNDARY'
    second = teacher.query(obs, opportunity=boundary(), boundary_id=4)
    assert second.intent == '236B' and second.intent_id == 2
    assert teacher.intents.invalidations == 1
    third = teacher.query(obs, opportunity=boundary(), boundary_id=5)
    assert third.intent == '5A' and teacher.env[b'loaded'] == 1
    assert teacher.blocked is None  # no coroutine ever reached the error
    with pytest.raises(ValueError, match='currently pending'):
        teacher.intents.reject(first.intent_id)


def test_boundary_mask_staleness_forced_and_episode_isolation(tmp_path):
    contract, teacher = actor(tmp_path, '''
count = 0
function teacher_decide() count=count+1; command('236B') end
''')
    warm(contract, teacher)
    for at, opp, expected in (
        (2, boundary(source='MASKED_CANDIDATES_NATIVE_AUTHORITY'), 'WAIT_TRUSTED_DECISION_BOUNDARY'),
        (3, boundary(forced=True), 'WAIT_TRUSTED_DECISION_BOUNDARY'),
        (3, boundary(), 'STALE_DECISION_BOUNDARY'),
        (5, boundary(banned='236B'), 'MASKED_INTENT'),
        (6, boundary(), 'INTENT_PROPOSED'),
    ):
        result = teacher.query(observation(contract), opportunity=opp, boundary_id=at)
        assert result.status == expected
    assert teacher.env[b'count'] == 2
    other = teacher.teacher.spawn(17)
    warm(contract, other)
    result = other.query(observation(contract), opportunity=boundary(), boundary_id=2)
    assert result.intent_id == 1 and other.env[b'count'] == 1
    assert teacher.env[b'count'] == 2


def test_replans_using_updated_delayed_world_not_previous_choice(tmp_path):
    contract, teacher = actor(tmp_path, '''
function teacher_decide()
 if my_act==400 then command('236B') else command('5A') end
end
''')
    for action in (400, 0):
        teacher.query(observation(contract, action))
    first = teacher.query(observation(contract), opportunity=boundary(), boundary_id=2)
    assert first.intent == '236B'
    teacher.intents.reject(first.intent_id)
    second = teacher.query(observation(contract), opportunity=boundary(), boundary_id=3)
    assert second.intent == '5A'


@pytest.mark.parametrize('source,expected', [
    ("function teacher_decide() command('623Z') end", 'UNMAPPABLE_INTENT'),
    ("function teacher_decide() command('5AB') end", 'UNMAPPABLE_INTENT'),
    ("function teacher_decide() return nil end", 'NO_INTENT'),
    ("function teacher_decide() key_on(ACT_B) end", 'UNSUPPORTED_INTENT_SEGMENT'),
    ("function teacher_decide() wait(3) end", 'UNSUPPORTED_INTENT_SEGMENT'),
    ("function teacher_decide() local a=enemy_keys end", 'BLOCKED_INTENT_SOURCE'),
    ("function teacher_decide() while true do end end", 'BLOCKED_INTENT_SOURCE'),
    ("function main() error('MAIN_MUST_NOT_RUN') end", 'BLOCKED_INTENT_ADAPTER'),
])
def test_no_fake_labels_or_hidden_coroutines(tmp_path, source, expected):
    contract, teacher = actor(tmp_path, source)
    warm(contract, teacher)
    result = teacher.query(observation(contract), opportunity=boundary(), boundary_id=2)
    assert result.status == expected
    assert result.action is None and not result.input_legal


def test_original_full_ai_requires_reviewed_entry_not_automatic_main_restart():
    root = Path(__file__).resolve().parents[2]
    package = root / 'third_party/th123_ai/package/th123ai/script'
    if not package.is_dir():
        pytest.skip('external original AI package unavailable')
    from test_kickstarting import interface
    contract = interface()
    teacher = QueryTeacher(contract, spec(package) | {
        'script': '01_marisa_main.ai', 'mode': 'intent', 'instruction_limit': 1000000}).spawn(17)
    warm(contract, teacher)
    result = teacher.query(observation(contract), opportunity=boundary(), boundary_id=2)
    assert result.status == 'BLOCKED_INTENT_ADAPTER'
    assert result.action is None and 'main is not restarted' in result.detail


def test_intent_does_not_silently_become_a_90_action_ppo_label(tmp_path):
    from test_kickstarting import make_model
    from soku_rl.rl.kickstarting import KickstartingPPO, attach_kickstarting
    contract, teacher = actor(tmp_path, "function teacher_decide() command('236B') end")
    model = make_model(contract, KickstartingPPO)
    with pytest.raises(RuntimeError, match='decision-boundary provider'):
        attach_kickstarting(model, contract,
            dict(teacher=teacher.teacher.spec, lambda_initial=.1, decay_transitions=100), False)
    assert not hasattr(model, '_expert')
