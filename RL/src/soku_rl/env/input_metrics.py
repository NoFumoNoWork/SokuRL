"""Executed logical inputs, distinct from successful attacks and card actions."""
from collections import Counter

from soku_rl.env.encoding import decode_action, encode_action


class InputMetrics:
    def __init__(self):
        self.reset()

    def reset(self):
        self.frames = 0
        self.counts = [Counter(), Counter()]
        self.previous = ()
        self.changes = [0, 0]

    def step(self, joint):
        inputs = tuple(decision.inputs for decision in joint)
        if len(inputs) != 2:
            raise ValueError("input metrics require both players")
        for seat, value in enumerate(inputs):
            self.counts[seat][value] += 1
            if self.frames:
                self.changes[seat] += int(value != self.previous[seat])
        self.previous = inputs
        self.frames += 1

    def snapshot(self, seat):
        if type(seat) is not int or seat not in (0, 1):
            raise ValueError("input metrics require an actual player seat")
        return {"available": True, "schema": 1,
            "measurement": "logical_inputs_applied_per_frame", "frames": self.frames,
            **{role: {"changed_commands": self.changes[player],
                "command_counts": {str(encode_action(value)): count
                    for value, count in sorted(self.counts[player].items())}}
                for role, player in (("own", seat), ("opponent", 1 - seat))}}


def summarize_inputs(records):
    measured = [record for record in records if record.get("available") is True]
    result = {"episodes": len(records), "measured_episodes": len(measured)}
    if not measured:
        return result
    for record in measured:
        if (record["schema"] != 1 or record["measurement"] != "logical_inputs_applied_per_frame"
                or type(record["frames"]) is not int or record["frames"] < 1):
            raise ValueError("input summaries require completed nonempty measured episodes")
    frames = sum(record["frames"] for record in measured)
    transitions = frames - len(measured)
    rates = {}
    for role in ("own", "opponent"):
        counts = Counter()
        changes = 0
        for record in measured:
            values = record[role]
            commands = values["command_counts"]
            if (any(type(n) is not int or n < 1 for n in commands.values())
                    or sum(commands.values()) != record["frames"]
                    or type(values["changed_commands"]) is not int
                    or not 0 <= values["changed_commands"] < record["frames"]):
                raise ValueError("input counts must match the completed episode's frames")
            counts.update(commands)
            changes += values["changed_commands"]
        decoded = [(decode_action(int(command)).inputs, count) for command, count in counts.items()]
        rates.update({f"{role}_{name}_rate": sum(count for inputs, count in decoded if select(inputs)) / frames
            for name, select in (
                ("attack_key", lambda inputs: any(inputs[2:5])),
                ("spell_key", lambda inputs: inputs[7]),
                ("change_card_key", lambda inputs: inputs[6]),
                ("left", lambda inputs: inputs[0] == -1),
                ("right", lambda inputs: inputs[0] == 1),
                ("horizontal_neutral", lambda inputs: inputs[0] == 0),
                ("neutral", lambda inputs: not any(inputs)))})
        rates[f"{role}_modal_command_fraction"] = max(counts.values()) / frames
        rates[f"{role}_mean_run_frames"] = frames / (changes + len(measured))
        if transitions:
            rates[f"{role}_command_change_rate"] = changes / transitions
    result.update(frames=frames, transitions=transitions, pooled=rates)
    return result
