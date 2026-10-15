"""RL replay persistence over the public Platform transport."""
from pathlib import Path
import hashlib
import json
from uuid import uuid4
from sokurl_platform.transport import PROTOCOL, MAX_MESSAGE, _read_exact, receive, send, WorkerConnection

class WorkerBackend(WorkerConnection):
    """Adapt a worker connection to independently reset offline game slots."""

    def __init__(self, command, cwd, log_path, timeout, launch_timeout, mute_audio, game_directory,
                 fresh_reset=False, record_replays=False, cards_disabled=False, fixed_weather=None,
                 combat_events=False, native_input_counters=False):
        self.replay_directory = Path(log_path).parent / "replays" / uuid4().hex
        self.replay_number = 0
        super().__init__(command, cwd, log_path, timeout, launch_timeout, mute_audio, game_directory,
                         fresh_reset, record_replays, cards_disabled, fixed_weather, combat_events,native_input_counters)

    def request(self, operation, payload):
        result = super().request(operation, payload)
        if operation == "initialize":
            return result
        self._save_replays(result["replays"])
        return result["value"]

    def _response_value(self, response):
        if not response["ok"]:
            self._save_replays(response["replays"])
        return super()._response_value(response)

    def _save_replays(self, replays):
        for replay in replays:
            self.replay_number += 1
            self.replay_directory.mkdir(parents=True, exist_ok=True)
            stem = f"episode-{self.replay_number:08d}-slot-{replay['slot']}"
            data = replay["data"]
            path = self.replay_directory / (stem + ".rep")
            with path.open("xb") as target:
                target.write(data)
            metadata = {key: value for key, value in replay.items() if key != "data"}
            metadata.update(sha256=hashlib.sha256(data).hexdigest(), replay=path.name)
            with path.with_suffix(".json").open("x", encoding="utf-8") as target:
                json.dump(metadata, target, indent=2)

    def reset_slots(self, seeds):
        return self.request("reset", seeds)

    def reset_matchups(self, seeds, matches):
        return self.request("reset_matchups", {"seeds": seeds, "matches": matches})

    def configure_observation(self, mode):
        return self.request("configure_observation", mode)

    def step(self, actions):
        return self.request("step", actions)
