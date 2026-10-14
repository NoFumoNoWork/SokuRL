import sys
from pathlib import Path
from types import SimpleNamespace
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from run_command_input import run, acknowledged
from command_input import encode_action


def test_platform_encoding_all_576_inputs():
    for horizontal in (-1, 0, 1):
        for vertical in (-1, 0, 1):
            for buttons in range(64):
                keys = (horizontal, vertical) + tuple((buttons >> i) & 1 for i in range(6))
                assert encode_action(keys) == ((horizontal + 1) * 3 + vertical + 1) * 64 + buttons


class Client:
    def __init__(self):
        fighter = SimpleNamespace(facing=1)
        self.state = SimpleNamespace(in_gameplay=True, run_state_name='PAUSED',
            latest=SimpleNamespace(segmentId=1, p1=fighter, p2=fighter),
            game_frame=600, ack_seq=0, result_code=2, result_name='COMPLETE')
        self.inputs = []
        self.released = False
    def snapshot(self): return self.state
    def step_with_inputs(self, p1, p2):
        self.inputs.append((p1, p2))
        self.state.game_frame += 1
        self.state.ack_seq += 1
        return self.state.ack_seq
    def wait_for_ack(self, sequence, timeout): return self.state
    def drain_frames(self): return []
    def release(self):
        self.released = True
        self.state.ack_seq += 1
        self.state.result_code = 3
        return self.state.ack_seq


class Events:
    native_input_counters = True
    def drain(self): return []


def test_standalone_runner_steps_exactly_and_releases_selected_instance():
    client = Client()
    report = run(client, Events(), '5B', 1)
    assert report['frames'] == 4 and client.released
    assert client.inputs[0][0] == (0,) * 8
    assert client.inputs[0][1][3] == 1
    assert report['scope'] == 'COMMAND_PROGRAM_COMPLETED_NOT_MOVE_LEGALITY'


def test_standalone_runner_rejects_missing_capability_and_running_instance():
    client = Client()
    events = Events()
    events.native_input_counters = False
    with pytest.raises(RuntimeError, match='advertise'):
        run(client, events, 'B', 0)
    events.native_input_counters = True
    client.state.run_state_name = 'RUNNING'
    with pytest.raises(RuntimeError, match='paused'):
        run(client, events, 'B', 0)


def test_standalone_runner_timeout_is_failure_and_releases_keys():
    client = Client()
    with pytest.raises(TimeoutError, match='budget'):
        run(client, Events(), '5B', 0, max_frames=1)
    assert client.released


def test_missing_ack_is_not_success():
    client = Client()
    with pytest.raises(TimeoutError, match='acknowledge'):
        acknowledged(client, 999, .001)
