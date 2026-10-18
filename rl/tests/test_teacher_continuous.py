"""Source-backed offline recovery tests. All receipts/opportunities SYNTHETIC."""
from pathlib import Path
from types import SimpleNamespace
import json

import gymnasium as gym
import numpy as np
import pytest

from soku_rl.env.encoding import encode_action, decode_action
from soku_rl.env.wrappers.learning import LearningConfig, LearningInterface
from soku_rl.env.observation.shared_native import SCHEMA, assemble_shared
from soku_rl.policy.god.package import ScriptPackage
from soku_rl.policy.god.continuous import ContinuousSourceTeacher, ExecutionReceipt, PlanState
from test_teacher_shared_native import native_example, public_example


def teacher(script='01_marisa_main.ai'):
    root = Path(__file__).resolve().parents[2]
    api = root/'third_party/th123_ai/source/th123_ai/api.ai'
    if not api.is_file():
        pytest.skip('external pinned source unavailable')
    package = ScriptPackage(root/'third_party/th123_ai/package/th123ai/script', api)
    interface = LearningInterface(SimpleNamespace(combat_reward=None,
        observation_mode='state', space=lambda: gym.spaces.Box(-1, 1, (1,), np.float32)),
        LearningConfig('combat', False, 0, 0))
    return ContinuousSourceTeacher(package, script, 17, interface,
        initial_action=interface.action(encode_action((0,)*8)))


def packet(frame, script='01_marisa_main.ai', *, hitstop=0, facing=1):
    state = native_example(frame)
    state.players[0].update(char=1 if script.startswith('01_') else 0,
        hitstop=hitstop, dir=facing)
    state.players[1]['char'] = 1-state.players[0]['char']
    return assemble_shared(public_example(state), state, public_frame=frame)


MASK = tuple(i % 10 not in (5, 6) for i in range(90))  # cards disabled


@pytest.mark.parametrize('script', ['01_marisa_main.ai', '00_reimu_main.ai'])
def test_original_calculation_survives_repeated_rejection(script, record_property):
    actor = teacher(script)
    previous = None
    for frame in range(12):
        if previous is not None:
            receipt = ExecutionReceipt(frame-1, (previous['training_label']+10) % 90,
                1, frame-1, True)
        else:
            receipt = None
        result = actor.query(packet(frame, script), schema=SCHEMA, receipt=receipt,
            trusted_boundary=True, legal_actions=MASK)
        assert result['training_label'] is not None, result
        previous = result
    assert actor.counts['observations'] == 12
    assert actor.counts['invalidations'] == actor.counts['recoveries'] == 11
    assert actor.memory[b'old_bt'] == 11
    assert actor.report()['native_verified'] is False
    record_property('SYNTHETIC_ONLY_METRICS', json.dumps(actor.report()))
    record_property('SYNTHETIC_ONLY_TRACE', json.dumps(list(actor.trace)))


def install_decision(actor, body):
    """Deliberate source seam fixture, never called a recovered original branch."""
    actor.mode(b'observation')
    actor.compile(b'function teacher_decide() '+body+b' end', b'@synthetic/decision')()


def test_real_motion_helper_prefix_and_acceptance_are_separate():
    actor = teacher()
    install_decision(actor, b'key_reset();lvr236(ACT_B)')
    previous = None
    emitted = []
    for frame in range(4):
        receipt = None if previous is None else ExecutionReceipt(
            frame-1, previous, 1, frame-1, True, True, accepted_plan_id=actor.serial)
        result = actor.query(packet(frame), schema=SCHEMA, receipt=receipt,
            trusted_boundary=True, legal_actions=MASK)
        assert result['training_label'] is not None, result
        previous = result['training_label']
        emitted.append(decode_action(actor.interface.command(previous)).inputs)
    assert emitted[0][:2] == (0, 1)
    assert emitted[1][:2] == (1, 1)
    assert emitted[2][0] == 1 and emitted[2][3] == 1  # forward+B
    assert actor.counts['accepted'] == 3  # ONLY explicit test receipts


def test_rejection_discards_coroutine_not_observation_history():
    actor = teacher()
    install_decision(actor, b'key_reset();lvr236(ACT_B)')
    first = actor.query(packet(0), schema=SCHEMA, trusted_boundary=True, legal_actions=MASK)
    result = actor.query(packet(1), schema=SCHEMA,
        receipt=ExecutionReceipt(0, (first['training_label']+10) % 90, 1, 0, True),
        trusted_boundary=True, legal_actions=MASK)
    keys = decode_action(actor.interface.command(result['training_label'])).inputs
    assert keys[:2] == (0, 1)  # new 2, never stale 3 of rejected 236
    assert actor.serial == 2 and actor.memory[b'old_bt'] == 1


@pytest.mark.parametrize('kind', ['missing', 'unconsumed', 'turn', 'interrupt', 'hitstop'])
def test_untrusted_receipts_do_not_resume_old_wait(kind):
    actor = teacher()
    install_decision(actor, b'key_on(ACT_B);yield();wait(20)')
    first = actor.query(packet(0), schema=SCHEMA, trusted_boundary=True, legal_actions=MASK)
    receipt = ExecutionReceipt(0, first['training_label'], 1, 0, kind!='unconsumed',
        accepted=kind not in ('unaccepted', 'unconsumed'), interrupted=kind=='interrupt',
        accepted_plan_id=actor.serial)
    result = actor.query(packet(1, hitstop=int(kind=='hitstop'), facing=-1 if kind=='turn' else 1),
        schema=SCHEMA, receipt=None if kind=='missing' else receipt,
        trusted_boundary=False, legal_actions=MASK)
    assert result['training_label'] is None
    assert actor.runner is None
    assert actor.counts['observations'] == 2
    assert actor.state == PlanState.WAITING
    recovery = actor.query(packet(2), schema=SCHEMA,
        receipt=ExecutionReceipt(1, actor.initial_action, 1, 1, True),
        trusted_boundary=True, legal_actions=MASK)
    assert recovery['training_label'] is not None
    assert actor.counts['recoveries'] == 1


def test_plan_writes_and_tables_cannot_pollute_calculation_memory():
    actor = teacher('00_reimu_main.ai')
    install_decision(actor, b'ex_combo=999;key_on(ACT_A);yield()')
    first = actor.query(packet(0, actor.script), schema=SCHEMA,
        trusted_boundary=True, legal_actions=MASK)
    assert first['training_label'] is not None
    assert actor.memory[b'ex_combo'] == 0  # actual observed reset, not proposed 999
    actor.invalidate('TEST_REJECTION')
    install_decision(actor, b'my_preact[0]=999;key_on(ACT_A);yield()')
    result = actor.query(packet(1, actor.script), schema=SCHEMA,
        receipt=ExecutionReceipt(0, actor.initial_action, 1, 0, True),
        trusted_boundary=True, legal_actions=MASK)
    assert result['training_label'] is None
    assert 'PLAN_MUTATES_OBSERVED_TABLE' in result['detail']
    assert actor.memory[b'my_preact'][0] == 0


def test_masks_reset_isolation_and_frame_gap():
    left, right = teacher(), teacher()
    install_decision(left, b'key_on(ACT_B);yield()')
    blocked = left.query(packet(0), schema=SCHEMA, trusted_boundary=True,
        legal_actions=(False,)*90)
    assert blocked['training_label'] is None
    assert right.frame == -1 and right.memory[b'old_bt'] == 0
    with pytest.raises(ValueError, match='consecutive'):
        left.query(packet(2), schema=SCHEMA)
    assert left.frame == 0


def test_wait_continues_only_on_consumed_matching_real_inputs():
    actor = teacher()
    install_decision(actor, b'key_reset();key_on(ACT_LEFT);wait(4)')
    previous = None
    for frame in range(4):
        result = actor.query(packet(frame), schema=SCHEMA,
            receipt=None if previous is None else ExecutionReceipt(frame-1, previous, 1, frame-1, True),
            trusted_boundary=(frame == 0), legal_actions=MASK)
        previous = result['training_label']
        assert previous is not None and actor.serial == 1
    assert actor.counts['submitted'] == 3 and actor.counts['accepted'] == 0
    assert result['reconciliation']['stage'] == 'SUBMITTED'


def test_missing_dependency_only_masks_this_decision_and_recovers():
    actor = teacher()
    install_decision(actor, b'if absent_native_field then key_on(ACT_A) end;yield()')
    blocked = actor.query(packet(0), schema=SCHEMA, trusted_boundary=True, legal_actions=MASK)
    assert blocked['training_label'] is None and 'UNTRUSTED_SOURCE_DEPENDENCY' in blocked['detail']
    install_decision(actor, b'key_reset();lvr236(ACT_B)')
    restored = actor.query(packet(1), schema=SCHEMA,
        receipt=ExecutionReceipt(0, actor.initial_action, 1, 0, True),
        trusted_boundary=True, legal_actions=MASK)
    assert restored['training_label'] is not None
    assert restored['intent'] == 'lvr236'
    assert actor.memory[b'old_bt'] == 1


@pytest.mark.parametrize('failure', ['foreign_acceptance', 'poll_gap'])
def test_native_evidence_never_inferred_from_equal_keys(failure):
    actor = teacher()
    install_decision(actor, b'key_on(ACT_B);yield();yield()')
    first = actor.query(packet(0), schema=SCHEMA, trusted_boundary=True, legal_actions=MASK)
    actor.last_poll = 7
    receipt = ExecutionReceipt(0, first['training_label'], 1, 9 if failure=='poll_gap' else 8,
        True, accepted=failure=='foreign_acceptance',
        accepted_plan_id=999 if failure=='foreign_acceptance' else None)
    next_row = actor.query(packet(1), schema=SCHEMA, receipt=receipt,
        trusted_boundary=False, legal_actions=MASK)
    assert next_row['training_label'] is None
    assert actor.runner is None
    assert actor.counts['accepted'] == 0


def test_matching_after_rejection_resynchronizes_poll_baseline():
    actor = teacher()
    install_decision(actor, b'key_reset();lvr236(ACT_B)')
    first = actor.query(packet(0), schema=SCHEMA, trusted_boundary=True, legal_actions=MASK)
    next_row = actor.query(packet(1), schema=SCHEMA,
        receipt=ExecutionReceipt(0, (first['training_label']+10)%90, 1, 50, True),
        trusted_boundary=True, legal_actions=MASK)
    continuation = actor.query(packet(2), schema=SCHEMA,
        receipt=ExecutionReceipt(1, next_row['training_label'], 1, 51, True),
        trusted_boundary=False, legal_actions=MASK)
    assert continuation['training_label'] is not None
    assert actor.serial == 2
    assert decode_action(actor.interface.command(continuation['training_label'])).inputs[:2] == (1, 1)


def test_no_receipt_no_fake_neutral_prefix_and_no_teacher_episode_restart():
    actor = teacher()
    first = actor.query(packet(0), schema=SCHEMA, trusted_boundary=True, legal_actions=MASK)
    blocked = actor.query(packet(1), schema=SCHEMA, trusted_boundary=True, legal_actions=MASK)
    assert first['training_label'] is not None and blocked['training_label'] is None
    assert blocked['reason'] == 'WAIT_ACTUAL_INPUT_HISTORY'
    assert actor.initialized and actor.counts['observations'] == 2


def test_partial_calculation_failure_requires_explicit_reset_not_fake_recovery():
    actor = teacher()
    actor.query(packet(0), schema=SCHEMA, trusted_boundary=True, legal_actions=MASK)
    actor.mode(b'observation')
    actor.compile(b'function teacher_observe() old_bt=888;error("synthetic partial write") end',
        b'@synthetic/partial-observation')()
    result = actor.query(packet(1), schema=SCHEMA)
    assert result['reason'] == 'OBSERVATION_DEPENDENCY_FAILURE'
    result = actor.query(packet(2), schema=SCHEMA)
    assert result['reason'] == 'RESET_REQUIRED_AFTER_PARTIAL_OBSERVATION_FAILURE'
    assert actor.counts['observations'] == 1


def test_caller_proven_cancel_boundary_replans_without_finishing_wait():
    actor = teacher()
    install_decision(actor, b'key_reset();key_on(ACT_LEFT);wait(20)')
    first = actor.query(packet(0), schema=SCHEMA, trusted_boundary=True, legal_actions=MASK)
    # This is a synthetic legal-cancel opportunity, not an inferred frame window.
    next_row = actor.query(packet(1), schema=SCHEMA,
        receipt=ExecutionReceipt(0, first['training_label'], 1, 0, True),
        trusted_boundary=True, replan=True, legal_actions=MASK)
    assert next_row['training_label'] is not None and actor.serial == 2
    assert actor.reasons['NATIVE_REPLAN_BOUNDARY'] == 1
    assert actor.memory[b'old_bt'] == 1
