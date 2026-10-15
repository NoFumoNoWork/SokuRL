"""Per-frame HP accounting, kept outside policy observations and rewards."""
from soku_rl.env.observation.privileged import PrivilegedObservation


def summarize_combat(records):
    measured = [record for record in records if record.get("available") is True]
    result = {"episodes": len(records), "measured_episodes": len(measured)}
    if measured:
        result["means"] = {key: sum(record[key] for record in measured) / len(measured)
            for key in ("own_hp_loss", "opponent_hp_loss", "own_hp_loss_frames",
                        "opponent_hp_loss_frames", "own_final_hp", "opponent_final_hp")}
    action_records = [record for record in measured if "own_spell_action_entries" in record]
    result["action_measured_episodes"] = len(action_records)
    if action_records:
        result["action_means"] = {key: sum(record[key] for record in action_records) / len(action_records)
            for key in ("own_spell_action_entries", "opponent_spell_action_entries")}
    return result


class CombatMetrics:
    def __init__(self):
        self.previous = ()
        self.loss = [0, 0]
        self.loss_frames = [0, 0]
        self.actions = ()
        self.entries = [{}, {}]

    def reset(self, observation):
        self.previous = self._health(observation)
        self.loss = [0, 0]
        self.loss_frames = [0, 0]
        self.actions = self._actions(observation)
        self.entries = [{}, {}]

    def step(self, observation):
        current = self._health(observation)
        for seat, (before, after) in enumerate(zip(self.previous, current, strict=True)):
            amount = max(0, before - after)
            self.loss[seat] += amount
            self.loss_frames[seat] += int(amount > 0)
        self.previous = current
        actions = self._actions(observation)
        for seat, (before, after) in enumerate(zip(self.actions, actions, strict=True)):
            if before != after:
                key = str(after)
                self.entries[seat][key] = self.entries[seat].get(key, 0) + 1
        self.actions = actions

    def snapshot(self, seat):
        if not self.previous:
            return {"available": False, "reason": "requires_privileged_state"}
        return {"available": True, "schema": 2, "measurement": "per_frame_hp_decrease",
            "own_hp_loss": self.loss[seat], "opponent_hp_loss": self.loss[1 - seat],
            "own_hp_loss_frames": self.loss_frames[seat],
            "opponent_hp_loss_frames": self.loss_frames[1 - seat],
            "own_final_hp": self.previous[seat], "opponent_final_hp": self.previous[1 - seat],
            "own_action_entries": dict(self.entries[seat]),
            "opponent_action_entries": dict(self.entries[1 - seat]),
            # SokuLib Action.hpp: USING_SC_ID_200..219; ALT_EFFECT is 650..669.
            # This measures action entry, not hits, card cost, or confirmed casts.
            "own_spell_action_entries": self._spell_entries(seat),
            "opponent_spell_action_entries": self._spell_entries(1 - seat)}

    def _spell_entries(self, seat):
        return sum(count for action, count in self.entries[seat].items() if 600 <= int(action) <= 619)

    @staticmethod
    def _actions(observation):
        if not isinstance(observation, PrivilegedObservation):
            return ()
        return tuple(int(player["act"]) for player in observation.players)

    @staticmethod
    def _health(observation):
        if not isinstance(observation, PrivilegedObservation):
            return ()
        return tuple(int(player["hp"]) for player in observation.players)
