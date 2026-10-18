"""Commit reviewed Lua intentions only against this plan's real acceptance.

These are source-script variable write sites, not a game-action handler table.
No native event is decoded here; acceptance attribution is supplied by the caller.
"""
from collections import Counter, deque


REIMU_WRITES = {
    2081: (1, 'NEXT', 'ACT_B'),
    2086: (1, 'NEXT', 'ACT_B'),
    2175: (1, 'NEXT', 'ACT_B'),
    1872: (0, 'PREVIOUS', 'ACT_B'),
    1879: (0, 'PREVIOUS', 'ACT_A'),
}


class TacticalState:
    def __init__(self, teacher):
        self.teacher = teacher
        self.pending = []
        self.accepted_buttons = set()
        self.counts = Counter()
        self.history = deque(maxlen=128)
        # Full source hash is checked by the owning teacher; additionally bind
        # the meaning of each reviewed write to its exact line contents.
        lines = teacher.package.files['00_reimu_main.ai'].splitlines()
        for line, (value, _, _) in REIMU_WRITES.items():
            if lines[line-1].strip() != f'ex_combo = {value};'.encode():
                raise ValueError('reviewed tactical source write changed')

    def discard(self):
        self.counts['discarded_unconfirmed'] += len(self.pending)
        self.pending.clear()
        self.accepted_buttons.clear()

    def commit(self, row):
        self.teacher.memory[b'ex_combo'] = row['value']
        self.counts['committed'] += 1
        self.history.append(dict(row, status='ACCEPTANCE_CONFIRMED', frame=self.teacher.frame))

    def observe_writes(self, rows):
        for row in rows:
            name = row['name']
            if name not in ('ex_combo', 'crush_combo', 'flag_kishitu'):
                continue
            spec = REIMU_WRITES.get(row['line'])
            if (self.teacher.script != '00_reimu_main.ai' or name != 'ex_combo' or
                    row['source'] != '@community/00_reimu_main.ai' or spec is None or
                    row['value'] != spec[0]):
                self.counts['plan_only_unreviewed_or_card'] += 1
                continue
            value, direction, key = spec
            record = dict(row, value=value, direction=direction, key=key,
                plan_id=self.teacher.serial)
            if direction == 'PREVIOUS':
                if key in self.accepted_buttons:
                    self.commit(record)
                else:
                    self.counts['unconfirmed_previous_acceptance'] += 1
            else:
                self.pending.append(record)
                self.counts['staged'] += 1

    def input_transition(self, previous, current, receipt):
        buttons = set(receipt.accepted_buttons)
        released = set()
        for key in ('ACT_A', 'ACT_B', 'ACT_C'):
            i = int(self.teacher.env[key.encode()])-2
            if current[i] and not previous[i]:
                self.accepted_buttons.discard(key)
            elif previous[i] and not current[i]:
                released.add(key)
        # An unaccepted release cannot leave an old flag waiting to be committed
        # by a later, unrelated press of the same button.
        kept = []
        for row in self.pending:
            if row['key'] in released and row['key'] not in buttons:
                self.counts['discarded_unaccepted_release'] += 1
            else:
                kept.append(row)
        self.pending = kept
        if receipt.accepted and not buttons:
            self.counts['acceptance_button_unattributed'] += 1
        self.accepted_buttons.update(buttons)
        remaining = []
        for row in self.pending:
            if row['plan_id'] == self.teacher.serial and row['key'] in buttons:
                self.commit(row)
            else:
                remaining.append(row)
        self.pending = remaining

    def report(self):
        return dict(counts=dict(self.counts), pending=len(self.pending),
            commits=list(self.history),
            policies=dict(ex_combo='reviewed next/previous native-accepted input only',
                crush_combo='original observation reset only;no active producer in pinned main',
                flag_kishitu='calculation owns persistent value;main card-use write disabled',
                loop_count='plan-local execution counter',
                math='original episode initialization;not tactical history'))
