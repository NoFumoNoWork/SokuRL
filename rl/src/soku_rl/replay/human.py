"""Fail-closed bookkeeping for existing native replay-consumption evidence."""
from sokurl_platform.replay_inputs import decode_keys, FIELDS


def consumed_pair(words, mask, before, after):
    if mask != 3:
        raise ValueError('human pilot requires both native replay input streams')
    total = len(words)
    if any(type(n) is not int or n < 0 or n > total for n in (before, after)):
        raise ValueError('remaining input count outside source record')
    if after > before:
        raise ValueError('replay stream reset/reinitialized; start a new proven boundary')
    start, stop = total-before, total-after
    if start % 2 or stop % 2:
        raise ValueError('unpaired native poll boundary')
    if start == stop:
        return None
    if stop-start != 2:
        raise ValueError('multiple native polls in one battle step; no unique decision label')
    pair = tuple(tuple(decode_keys(word)[name] for name in FIELDS) for word in words[start:stop])
    return dict(poll_index=start//2, word_range=[start, stop], raw_words=list(words[start:stop]), logical=pair)


def verify_pair(pair, samples, frame):
    """Check key agreement, not whether the game accepted the raw human action.

    A mismatch is diagnostic: postprocessing may alter keys, so it does not by
    itself prove that the source interval is wrong or justify replacing labels.
    """
    if len(samples) != 2 or {e['attacker_seat'] for e in samples} != {0, 1}:
        raise ValueError('requires exactly one sample per seat')
    for event in samples:
        if event['frame'] != frame:
            raise ValueError('native input sample belongs to a different battle frame')
        values = event['native_inputs_before']
        logical = tuple((n > 0)-(n < 0) for n in values[:2]) + tuple(int(n > 0) for n in values[2:])
        if logical != pair['logical'][event['attacker_seat']]:
            raise ValueError('consumed REP keys and native player keys disagree')


def supervision_action(inputs, interface, blocked):
    from soku_rl.env.encoding import encode_action
    command = encode_action(inputs)
    action = interface.inverse.get(command)
    if action is None:
        return None, 'UNREPRESENTABLE'
    if action in blocked:
        return None, 'POLICY_MASKED'
    return action, None
