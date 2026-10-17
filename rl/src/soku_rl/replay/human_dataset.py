"""Causal human replay shards over existing Platform observation interfaces."""
from collections import Counter, deque
from contextlib import ExitStack
from dataclasses import replace
import gzip
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from soku_rl.env.encoding import AGENTS, encode_action
from soku_rl.env.hisouten_env import EpisodeConfig
from soku_rl.env.match import MatchConfig, PlayerSetup
from soku_rl.env.observation.animation import observe_animation
from soku_rl.env.observation_history import ObservationHistory
from soku_rl.env.wrappers.learning import LearningConfig, LearningEpisode, LearningInterface
from soku_rl.policy.contract import read_training_contract
from soku_rl.replay.human import consumed_pair, supervision_action


SCHEMA = 'human_replay_dataset_v1'


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=True), encoding='utf-8')


def policy_interface(config, characters):
    if tuple(sorted(characters)) != (0, 1):
        raise ValueError('current human pilot requires Reimu versus Marisa')
    episode = replace(EpisodeConfig.from_dict(config['episode']), match=MatchConfig(
        *(PlayerSetup(character=c, palette=0, deck=0) for c in characters)))
    return LearningInterface(episode, LearningConfig(**config['wrappers']))


class LabelHistory:
    """Track missing past commands without removing their time positions."""
    def __init__(self, length):
        self.known = deque([False] * length, maxlen=length)
        self.previous = None

    def label(self, pair, seat, interface, blocked, compatible):
        known = all(self.known)
        action, reason = ((None, 'NO_NATIVE_POLL') if pair is None else
                          supervision_action(pair['logical'][seat], interface, blocked))
        if reason is None and not known:
            reason = 'UNKNOWN_COMMAND_HISTORY'
        if reason is None and not compatible:
            reason = 'INCOMPATIBLE_PREFIX'
        current = None if pair is None else tuple(pair['logical'][seat])
        changed = current is not None and self.previous is not None and current != self.previous
        previous_action = (-1 if self.previous is None else
                           supervision_action(self.previous, interface, blocked)[0])
        self.known.append(pair is not None)
        self.previous = current
        return dict(action=-1 if action is None else action, supervised=reason is None,
                    history_known=known, changed=changed,
                    previous_action=-1 if previous_action is None else previous_action,
                    reason=reason)


def validate_arrays(arrays, words, mask, seat, interface, blocked):
    n = len(arrays['actions'])
    if (n == 0 or any(len(v) != n for v in arrays.values())
            or arrays['observations'].shape != (n, *interface.observation_space.shape)
            or not np.isfinite(arrays['observations']).all()):
        raise ValueError('invalid human shard dimensions or observations')
    if not np.array_equal(arrays['consumption_frames'], arrays['observation_frames'] + 1):
        raise ValueError('labels must follow their pre-action observation by one native step')
    if n > 1 and not np.all(np.diff(arrays['consumption_frames']) == 1):
        raise ValueError('human shard compressed or reordered time')
    if not np.all(arrays['seats'] == seat):
        raise ValueError('wrong learner seat')
    for i in range(n):
        # Boundary rows deliberately carry no label; never consume a new stream
        # using the previous round's source cursor.
        if arrays['boundary'][i]:
            if arrays['supervised'][i] or arrays['raw_words_present'][i]:
                raise ValueError('round boundary must be excluded from supervision')
            continue
        pair = consumed_pair(words, mask, int(arrays['remaining_before'][i]),
                             int(arrays['remaining_after'][i]))
        if bool(pair is not None) != bool(arrays['raw_words_present'][i]):
            raise ValueError('source interval presence differs')
        if pair is not None:
            if (list(arrays['raw_words'][i]) != pair['raw_words']
                    or arrays['poll_index'][i] != pair['poll_index']):
                raise ValueError('raw label differs from original replay')
            expected, _ = supervision_action(pair['logical'][seat], interface, blocked)
            if arrays['actions'][i] != (-1 if expected is None else expected):
                raise ValueError('label is not the represented original human command')
        if arrays['supervised'][i] and (pair is None or not arrays['history_known'][i]
                or not arrays['compatible'][i] or arrays['actions'][i] in blocked
                or not 0 <= arrays['actions'][i] < interface.action_space.n):
            raise ValueError('invalid supervised label')


def export_replay(row, config, config_path, blocked, output, frames=7200, chunk_size=1024,
                  prefix_only=False):
    """Preserve the original first record and all observed chronology, not rewards.

    Stop at the first proven round/segment boundary, input exhaustion or horizon.
    Incompatible conditions mask the complete remaining prefix, including recovery.
    """
    from sokurl_platform.replay import launch_replay
    from sokurl_platform.replay_format import Replay
    from sokurl_platform.combat_capture import CombatCaptureClient
    from sokurl_platform.images import ImageClient
    from sokurl_platform.memory import ProcessMemory
    from sokurl_platform.native_status import remaining_replay_inputs
    from sokurl_platform.bridge_shared import WEATHER_CLEAR

    source = Path(row['file'])
    if sha256(source) != row['sha256']:
        raise ValueError('source replay changed since audit')
    replay = Replay.decode(source.read_bytes())
    if replay.encode() != source.read_bytes() or len(replay.matches) != 1:
        raise ValueError('pilot requires one intact original input record; no implicit concatenation')
    match = replay.matches[0]
    chars = tuple(p.character for p in match.players)
    interface = policy_interface(config, chars)
    read_training_contract(config_path, interface)
    seat = chars.index(1)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    result = dict(schema=SCHEMA, status='RUNNING', replay_id=row['id'], source=str(source),
        source_sha256=row['sha256'], seat=seat, characters=chars, source_record=0,
        source_records=len(replay.matches), set_id=row['set_id'], source_seed=match.seed,
        original_poll_pairs=len(match.input_words)//2, rows=0, supervised=0, shards=[],
        exclusions={}, boundaries=[], examples=[], condition_failure=None,
        full_replay_parity='NOT_TESTED', frames_removed=0,
        label_source='RAW_REP_WORDS_WITH_EXISTING_NATIVE_REMAINING_COUNTS',
        scope_source='EXISTING_CHARACTER_HANDLEINPUTS_ENTRY',
        sample_source='EXISTING_FRAME_END_PLAYER_KEYS_NOT_RAW_LABELS',
        observation_shape=list(interface.observation_space.shape), num_actions=int(interface.action_space.n),
        masked_actions=sorted(map(int,blocked)), timing='S[f-1] -> original words consumed during step f',
        training=False, fabricated_returns=False)
    save_json(output/'result.json', result)
    buffer, exclusions, event_counts = [], Counter(), Counter()
    label_history = LabelHistory(interface.config.action_history)

    def flush():
        if not buffer:
            return
        arrays = {key: np.asarray([r[key] for r in buffer]) for key in buffer[0]}
        arrays['observations'] = arrays['observations'].astype(np.float32)
        validate_arrays(arrays, match.input_words, replay.header[8], seat, interface, blocked)
        name = f'shard-{len(result["shards"]):04d}.npz'
        np.savez_compressed(output/name, **arrays)
        result['shards'].append(dict(path=name, sha256=sha256(output/name), rows=len(buffer),
            supervised=int(arrays['supervised'].sum()), first_frame=int(arrays['consumption_frames'][0]),
            last_frame=int(arrays['consumption_frames'][-1])))
        buffer.clear()
        save_json(output/'result.json', result)

    try:
        with ExitStack() as resources:
            game = launch_replay(source, 45., True, 'state')
            resources.callback(game.close)
            events = CombatCaptureClient(game.pid); resources.callback(events.close)
            image = ImageClient(game.pid); resources.callback(image.close)
            memory = ProcessMemory(game.pid); resources.callback(memory.close)
            states = resources.enter_context(gzip.open(output/'states.raw.gz', 'wb'))
            evidence = resources.enter_context(gzip.open(output/'events.jsonl.gz', 'wt', encoding='utf-8'))
            if events.version != 8 or events.replay_input_capture:
                raise ValueError('pilot requires existing ABI8 with new input diagnostic OFF')
            events.drain()
            raw = game.state
            if (int(raw.p1.characterId), int(raw.p2.characterId)) != chars or raw.frameId != 0:
                raise ValueError('wrong native replay seats or initial frame')
            initial_skills = (tuple(raw.p1.skillLevels), tuple(raw.p2.skillLevels))
            result['initial'] = dict(weather=int(raw.activeWeather), skills=initial_skills,
                stage=int(raw.stageId), round=int(raw.roundId), segment=int(raw.segmentId))
            # The known fresh default reference has zero exported skill levels.
            # Unobserved skill-map internals are not inferred from this check.
            compatible = raw.activeWeather == WEATHER_CLEAR and not any(map(any, initial_skills))
            if not compatible:
                result['condition_failure'] = dict(frame=0, reason='INITIAL_WEATHER_OR_SKILL_LEVEL')
            history = ObservationHistory(interface.episode, AGENTS)
            commands = LearningEpisode(interface)
            history.reset(0, observe_animation(raw, image.read_state(0, 3.), interface.episode.visibility,
                interface.episode.animation_object_slots, interface.episode.max_frames))
            for agent in AGENTS:
                commands.reset_agent(agent, history.observations()[agent])
            memory.begin_frame(); remaining = remaining_replay_inputs(memory)
            if remaining != len(match.input_words):
                raise ValueError('replay did not begin at the original input cursor')
            states.write(bytes(raw))
            result['stop_reason'] = 'HORIZON'
            for frame in range(1, frames+1):
                prior = raw
                observation = commands.observation(AGENTS[seat], history.observations()[AGENTS[seat]], frame-1)
                raw = game.step_native(5.)
                native = events.drain()
                memory.begin_frame(); after = remaining_replay_inputs(memory)
                if raw.frameId != frame or game.client.snapshot().dropped_frames:
                    raise ValueError('native frame gap or ring loss')
                boundary = (raw.roundId != prior.roundId or raw.segmentId != prior.segmentId)
                if after > remaining and not boundary:
                    raise ValueError('input stream reset without proven round/segment boundary')
                pair = None if boundary else consumed_pair(match.input_words, replay.header[8], remaining, after)
                samples = [e for e in native if e['kind'] == 'NATIVE_INPUT_SAMPLE']
                if (len(samples) != 2 or {e['attacker_seat'] for e in samples} != {0, 1}
                        or any(e['frame'] != frame for e in native)):
                    raise ValueError('missing, duplicate or wrong-frame native event evidence')
                reason = ('ROUND_BOUNDARY' if boundary else 'WEATHER' if raw.activeWeather != WEATHER_CLEAR
                    else 'SKILL_LEVEL_CHANGED' if (tuple(raw.p1.skillLevels),tuple(raw.p2.skillLevels)) != initial_skills
                    else 'CARD_INPUT' if pair and any(any(p[6:]) for p in pair['logical']) else None)
                if reason and compatible:
                    result['condition_failure'] = dict(frame=frame, reason=reason)
                compatible = compatible and reason is None
                label = label_history.label(pair, seat, interface, blocked, compatible)
                if boundary:
                    label.update(supervised=False, reason='ROUND_BOUNDARY')
                    result['boundaries'].append(dict(frame=frame, before_round=int(prior.roundId),
                        after_round=int(raw.roundId), before_segment=int(prior.segmentId),
                        after_segment=int(raw.segmentId), before_remaining=remaining, after_remaining=after,
                        handling='EXCLUDE_TRANSITION_AND_STOP_NO_CROSS_ROUND_HISTORY'))
                buffer.append(dict(observations=observation, actions=label['action'],
                    supervised=label['supervised'], history_known=label['history_known'],
                    changed=label['changed'], previous_action=label['previous_action'],
                    reason=label['reason'] or '', compatible=compatible, boundary=boundary,
                    raw_words=[0,0] if pair is None else pair['raw_words'], raw_words_present=pair is not None,
                    poll_index=-1 if pair is None else pair['poll_index'], remaining_before=remaining,
                    remaining_after=after, observation_frames=frame-1, consumption_frames=frame,
                    seats=seat, round_before=int(prior.roundId), round_after=int(raw.roundId)))
                record = dict(frame=frame, remaining_before=remaining, remaining_after=after,
                    pair=pair, label=label, compatible=compatible, native_events=native)
                evidence.write(json.dumps(record, ensure_ascii=True)+'\n')
                states.write(bytes(raw))
                event_counts.update(e['kind'] for e in native)
                result['rows'] += 1
                result['supervised'] += int(label['supervised'])
                exclusions.update([label['reason']] if label['reason'] else [])
                if len(result['examples']) < 12 and (frame in (1,62,69,70) or
                        raw.p1.hitstop or raw.p2.hitstop or any(e['kind'].endswith('RESOLUTION') for e in native)):
                    result['examples'].append(record)
                remaining = after
                if len(buffer) >= chunk_size:
                    flush()
                if boundary or after == 0 or (prefix_only and not compatible):
                    result['stop_reason'] = ('ROUND_BOUNDARY' if boundary else
                        'INPUT_EXHAUSTED' if after == 0 else 'CONDITION_PREFIX_END')
                    break
                for s, agent in enumerate(AGENTS):
                    commands.record_command(agent, 256 if pair is None else encode_action(pair['logical'][s]))
                scene = observe_animation(raw, image.read_state(frame, 3.), interface.episode.visibility,
                    interface.episode.animation_object_slots, interface.episode.max_frames)
                history.append(frame, scene)
            flush()
        result.update(status='PASS_EXPORT' if result['supervised'] else 'BLOCKED_NO_QUALIFIED_PREFIX',
            exclusions=dict(exclusions), native_event_counts=dict(event_counts),
            raw_sha256=sha256(output/'states.raw.gz'), events_sha256=sha256(output/'events.jsonl.gz'),
            seconds=time.perf_counter()-started)
        result['supervised_frames_per_second'] = result['supervised']/result['seconds']
    except BaseException as error:
        result.update(status='FAIL', error=repr(error), rows_buffered_uncommitted=len(buffer),
                      seconds=time.perf_counter()-started)
        raise
    finally:
        save_json(output/'result.json', result)
    return result
