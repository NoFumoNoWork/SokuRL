"""Execute a command program on an explicitly selected, paused Platform instance."""
import argparse
import json

from sokurl_platform.bridge_shared import BridgeClient
from sokurl_platform.combat_capture import CombatCaptureClient
from sokurl_platform.command_input import CommandPlayer, compile_commands, NEUTRAL


def acknowledged(client, sequence, timeout):
    snapshot = client.wait_for_ack(sequence, timeout)
    if snapshot.ack_seq != sequence:
        raise TimeoutError('Bridge did not acknowledge the command')
    if snapshot.result_code not in (1, 2, 3):
        raise RuntimeError(f'Bridge command failed: {snapshot.result_name}')
    return snapshot


def run(client, events, text, seat, confirm=False, max_frames=10000, timeout=2.0):
    if not events.native_input_counters:
        raise RuntimeError('Bridge must advertise enabled native input counters (combat ABI8)')
    snapshot = client.snapshot()
    if not snapshot.in_gameplay or snapshot.run_state_name != 'PAUSED':
        raise RuntimeError('select an in-game, paused instance; this tool does not initialize a match')
    events.drain()
    player = CommandPlayer(compile_commands(text, confirm=confirm))
    segment = int(snapshot.latest.segmentId)
    first = snapshot.game_frame
    count = 0
    observed = []
    try:
        while not player.done:
            if count >= max_frames:
                raise TimeoutError('command program exceeded its frame budget')
            if snapshot.latest.segmentId != segment or not snapshot.in_gameplay:
                raise RuntimeError('episode changed or left gameplay')
            fighter = snapshot.latest.p1 if seat == 0 else snapshot.latest.p2
            keys = player.next(snapshot.game_frame, int(fighter.facing), segment=segment)
            pair = (keys, NEUTRAL) if seat == 0 else (NEUTRAL, keys)
            previous = snapshot.game_frame
            snapshot = acknowledged(client, client.step_with_inputs(*pair), timeout)
            if snapshot.game_frame != previous + 1:
                raise RuntimeError('expected exactly one simulation frame')
            native = events.drain()
            player.observe_native(native, seat)
            observed.extend(e for e in native if e['kind'] == 'ACTION_TRANSITION')
            client.drain_frames()
            count += 1
        return dict(status='PASS', scope='COMMAND_PROGRAM_COMPLETED_NOT_MOVE_LEGALITY',
                    first_frame=first, last_frame=snapshot.game_frame, frames=count,
                    segment=segment, seat=seat, action_transitions=observed)
    finally:
        acknowledged(client, client.release(), timeout)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pid', type=int, required=True)
    parser.add_argument('--seat', type=int, choices=(0, 1), default=0)
    parser.add_argument('--commands', required=True)
    parser.add_argument('--confirm', action='store_true')
    parser.add_argument('--max-frames', type=int, default=10000)
    parser.add_argument('--timeout', type=float, default=2.0)
    args = parser.parse_args()
    if args.pid <= 0 or args.max_frames <= 0 or args.timeout <= 0:
        parser.error('PID, frame budget and timeout must be positive')
    with BridgeClient(args.pid) as client:
        events = CombatCaptureClient(args.pid)
        try:
            print(json.dumps(run(client, events, args.commands, args.seat,
                                 args.confirm, args.max_frames, args.timeout), indent=2))
        finally:
            events.close()


if __name__ == '__main__':
    main()
