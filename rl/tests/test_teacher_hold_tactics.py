"""Consumed input time and native action acceptance are distinct contracts.

All game observations/receipts below are synthetic. Exact source bodies are used
at declared fixture entries; no coverage of their enclosing native branch assumed.
"""
from soku_rl.env.encoding import decode_action
from soku_rl.env.observation.shared_native import SCHEMA, assemble_shared
from soku_rl.policy.god.continuous import ExecutionReceipt
from test_teacher_continuous import teacher, packet, MASK, install_decision
from test_teacher_shared_native import native_example, public_example
import pytest


def combat_packet(frame):
    state = native_example(frame)
    state.players[0].update(char=0, act=400)
    state.players[1].update(char=1, act=50)
    return assemble_shared(public_example(state), state, public_frame=frame)


def install_source_body(actor, first, last):
    lines = actor.package.files[actor.script].splitlines(keepends=True)
    code = b'\n'*(first-2)+b'function teacher_decide()\n'+b''.join(lines[first-1:last])+b'\nend'
    actor.mode(b'observation')
    actor.compile(code, ('@community/'+actor.script).encode())()


def test_hold_release_and_wait_without_fabricated_action_acceptance():
    actor = teacher()
    install_decision(actor, b'key_reset();key_on(ACT_B);wait(3);key_off(ACT_B);wait(2)')
    previous = None
    outputs = []
    for frame in range(6):
        result = actor.query(packet(frame), schema=SCHEMA,
            receipt=None if previous is None else ExecutionReceipt(frame-1, previous, 1, frame-1, True),
            trusted_boundary=frame==0, legal_actions=MASK)
        previous = result['training_label']
        assert previous is not None, result
        outputs.append(decode_action(actor.interface.command(previous)).inputs[3])
    assert outputs == [1, 1, 1, 0, 0, 0]
    assert actor.counts['accepted'] == 0 and actor.serial == 1
    assert actor.counts['input:pressed:B'] == 1
    assert actor.counts['input:held:B'] == 2
    assert actor.counts['input:released:B'] == 1


def test_original_hold_helper_advances_from_consumption_not_acceptance():
    actor = teacher()
    install_decision(actor, b'key_reset();lvr236h(ACT_B);wait(3);key_reset()')
    previous = None
    for frame in range(7):
        result = actor.query(packet(frame), schema=SCHEMA,
            receipt=None if previous is None else ExecutionReceipt(frame-1, previous, 1, frame-1, True),
            trusted_boundary=frame==0, legal_actions=MASK)
        previous = result['training_label']
        assert previous is not None, result
    assert actor.counts['accepted'] == 0
    assert actor.counts['input:held:B'] > 0


def test_unaccepted_press_does_not_block_whole_plan_or_claim_success():
    actor = teacher()
    install_decision(actor, b'key_on(ACT_A);yield();key_reset();yield()')
    first = actor.query(packet(0), schema=SCHEMA, trusted_boundary=True, legal_actions=MASK)
    result = actor.query(packet(1), schema=SCHEMA,
        receipt=ExecutionReceipt(0, first['training_label'], 1, 0, True),
        trusted_boundary=False, legal_actions=MASK)
    assert result['training_label'] is not None
    assert result['reconciliation']['stage'] == 'SUBMITTED'
    assert result['reconciliation']['action_acceptance'] == 'NOT_OBSERVED'


def test_source_tactic_not_committed_before_matching_native_feedback():
    actor = teacher('00_reimu_main.ai')
    install_source_body(actor, 2081, 2084)  # original flag=1;reset;yield;key_event(B)
    first = actor.query(combat_packet(0), schema=SCHEMA, trusted_boundary=True, legal_actions=MASK)
    assert actor.tactics.counts['staged'] == 1
    assert actor.memory[b'ex_combo'] is None  # no fabricated prehistory in fixture
    second = actor.query(combat_packet(1), schema=SCHEMA,
        receipt=ExecutionReceipt(0, first['training_label'], 1, 0, True),
        trusted_boundary=False, legal_actions=MASK)
    assert actor.memory[b'ex_combo'] is None
    actor.query(combat_packet(2), schema=SCHEMA,
        receipt=ExecutionReceipt(1, second['training_label'], 1, 1, True,
            accepted=True, accepted_plan_id=actor.serial, accepted_buttons=('ACT_B',)),
        trusted_boundary=False, legal_actions=MASK)
    assert actor.memory[b'ex_combo'] == 1
    assert actor.tactics.counts['committed'] == 1
    # Reject the NEXT input; the already confirmed tactical memory survives.
    actor.invalidate('STUDENT_REJECTED_LATER')
    assert actor.memory[b'ex_combo'] == 1


def test_rejected_tactical_input_never_commits_flag():
    actor = teacher('00_reimu_main.ai')
    install_source_body(actor, 2081, 2084)
    actor.query(combat_packet(0), schema=SCHEMA, trusted_boundary=True, legal_actions=MASK)
    actor.query(combat_packet(1), schema=SCHEMA,
        receipt=ExecutionReceipt(0, 0, 1, 0, True), trusted_boundary=False, legal_actions=MASK)
    assert actor.memory[b'ex_combo'] is None
    assert not actor.tactics.pending


def test_unaccepted_release_cannot_commit_on_a_later_unrelated_b():
    actor = teacher('00_reimu_main.ai')
    install_source_body(actor, 2081, 2084)
    previous = None
    for frame in range(5):
        result = actor.query(combat_packet(frame), schema=SCHEMA,
            receipt=None if previous is None else ExecutionReceipt(frame-1, previous, 1, frame-1, True),
            trusted_boundary=frame==0, legal_actions=MASK)
        previous = result['training_label']
        assert previous is not None
    assert actor.memory[b'ex_combo'] is None
    assert actor.tactics.counts['discarded_unaccepted_release'] == 1


def test_explicit_release_acceptance_can_commit_without_current_button_down():
    actor = teacher('00_reimu_main.ai')
    install_source_body(actor, 2081, 2084)
    previous = None
    for frame in range(5):
        ack = frame == 4  # receipt describes the B release from source key_event
        result = actor.query(combat_packet(frame), schema=SCHEMA,
            receipt=None if previous is None else ExecutionReceipt(frame-1, previous, 1, frame-1, True,
                accepted=ack, accepted_plan_id=actor.serial if ack else None,
                accepted_buttons=('ACT_B',) if ack else ()),
            trusted_boundary=frame==0, legal_actions=MASK)
        previous = result['training_label']
        assert previous is not None
    assert actor.memory[b'ex_combo'] == 1
    assert actor.tactics.counts['committed'] == 1
    assert result['reconciliation']['input_changes']['released'] == ['B']


def test_unreviewed_write_and_card_flag_do_not_become_history():
    actor = teacher('00_reimu_main.ai')
    install_decision(actor, b'ex_combo=999;flag_kishitu=2;key_on(ACT_B);yield()')
    first = actor.query(combat_packet(0), schema=SCHEMA, trusted_boundary=True, legal_actions=MASK)
    actor.query(combat_packet(1), schema=SCHEMA,
        receipt=ExecutionReceipt(0, first['training_label'], 1, 0, True,
            accepted=True, accepted_plan_id=actor.serial, accepted_buttons=('ACT_B',)),
        trusted_boundary=False, legal_actions=MASK)
    assert actor.memory[b'ex_combo'] is None
    assert actor.memory[b'flag_kishitu'] != 2
    assert actor.tactics.counts['committed'] == 0


def test_original_marisa_twenty_frame_hold_and_release_body():
    actor = teacher()
    install_source_body(actor, 1847, 1853)
    previous = None
    for frame in range(22):
        result = actor.query(packet(frame), schema=SCHEMA,
            receipt=None if previous is None else ExecutionReceipt(frame-1, previous, 1, frame-1, True),
            trusted_boundary=frame==0, legal_actions=MASK)
        if frame < 21:
            previous = result['training_label']
            assert previous is not None, result
        else:
            assert result['training_label'] is None  # finished;no fabricated new choice
    assert actor.counts['input:pressed:B'] == 1
    assert actor.counts['input:held:B'] == 19
    assert actor.counts['input:released:B'] == 1
    assert actor.counts['accepted'] == 0


@pytest.mark.parametrize('confirmed', [False, True])
def test_source_tactic_reset_requires_preceding_accepted_attack(confirmed):
    actor = teacher('00_reimu_main.ai')
    # Explicit synthetic prior confirmed tactic;not a claimed native snapshot.
    actor.memory[b'ex_combo'] = 1
    install_source_body(actor, 1871, 1872)
    previous = None
    for frame in range(5):
        ack = confirmed and frame==1
        result = actor.query(combat_packet(frame), schema=SCHEMA,
            receipt=None if previous is None else ExecutionReceipt(frame-1, previous, 1, frame-1, True,
                accepted=ack, accepted_plan_id=actor.serial if ack else None,
                accepted_buttons=('ACT_B',) if ack else ()),
            trusted_boundary=frame==0, legal_actions=MASK)
        previous = result['training_label']
        assert previous is not None
    assert actor.memory[b'ex_combo'] == (0 if confirmed else 1)


def test_foreign_button_acceptance_cannot_commit_tactics():
    actor = teacher()
    install_decision(actor, b'key_reset();key_on(ACT_A);yield();yield()')
    first = actor.query(packet(0), schema=SCHEMA, trusted_boundary=True, legal_actions=MASK)
    result = actor.query(packet(1), schema=SCHEMA,
        receipt=ExecutionReceipt(0, first['training_label'], 1, 0, True,
            accepted=True, accepted_plan_id=actor.serial, accepted_buttons=('ACT_B',)),
        trusted_boundary=False, legal_actions=MASK)
    assert result['training_label'] is None
    assert result['reconciliation']['reason'] == 'ACCEPTANCE_INPUT_MISMATCH'
    assert actor.counts['accepted'] == 0
