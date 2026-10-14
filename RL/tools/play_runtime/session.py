"""Own network engines, sample complete frames and submit bounded local input."""
from contextlib import ExitStack, closing
from dataclasses import asdict, replace
import os
import time

from game_runtime.privileged import PrivilegedReader
from game_runtime.diagnostic import DiagnosticReader
from network_runtime.game import NetworkGame
from play_runtime.channels import RealtimeHistory, RealtimeInput
from soku_rl.env.observation.visible_state import observe_visible_states
from soku_rl.play.match import MatchFrame
from soku_rl.play.settings import client_plan


class Menu:
    def __init__(self, game, character):
        self.game, self.character = game, character
        self.pending, self.next_pulse = 0, 0.

    def poll(self, state):
        events = []
        if self.pending:
            reply = self.game.clients["input"].read_reply(self.pending)
            if reply != "pending":
                events.append({"kind": "menu_result", "reply": reply, "request": self.pending})
                self.pending = 0
        now = time.monotonic()
        if not self.pending and now >= self.next_pulse:
            if state.scene in (8, 9):
                menu = self.game.clients["menu"]
                if menu.block.commandSeq == menu.block.ackSeq:
                    menu.menu_choose_character(self.character)
                    self.next_pulse = now+.1
                    events.append({"kind": "menu_selection", "character": self.character})
            elif state.in_battle and max(state.scores) >= 2:
                self.pending = self.game.clients["input"].confirm_result(state)
                self.next_pulse = now+1
                events.append({"kind": "menu_confirm", "request": self.pending})
        return events


class RealtimeSession:
    def __init__(self, settings, episode, timeout):
        if episode.observation_mode not in {"state", "privileged_state", "diagnostic_state"}:
            raise ValueError("realtime transport requires state, privileged_state or diagnostic_state observations")
        self.episode = episode
        self.plan = client_plan(settings)
        self.stack = ExitStack()
        self.games, self.menus, self.readers = {}, {}, {}
        self.cursor, self.seen_battle = 0, False
        self.menu_identity, self.input_identity = (), ()
        try:
            for client in self.plan:
                address = "127.0.0.1" if settings["connection"] == "local" else settings["network"]["address"]
                network = settings["network"] | {"address": address, "role": client["role"], "automate_menu": False}
                os.environ["SOKURL_MUTE_AUDIO"] = "1" if client["mute_audio"] else "0"
                game = self.stack.enter_context(closing(NetworkGame(network, asdict(episode.visibility),
                    timeout, client["render"], client["realtime"])))
                self.games[client["name"]] = game
                game.set_caption("SokuRL - AI" if client["realtime"] else "SokuRL - Player")
                if client["automate_menu"]:
                    self.menus[client["name"]] = Menu(game, client["character"])
                if client["role"] == "host":
                    game.wait_host()
                if client["realtime"]:
                    self.seat, self.character = client["seat"], client["character"]
                    self.history = self.stack.enter_context(closing(RealtimeHistory(game.process.pid)))
                    self.input = self.stack.enter_context(closing(RealtimeInput(game.process.pid, self.seat)))
        except BaseException:
            self.close()
            raise

    def identity(self):
        return {"games": {name: game.process.pid for name, game in self.games.items()},
                "seat": self.seat, "character": self.character, "clients": self.plan}

    def poll(self):
        try:
            return self._poll()
        except TimeoutError as error:
            # A busy snapshot/menu channel must not close either game process.
            return {"closed": False, "records": (),
                    "events": ({"kind": "observation_wait", "reason": str(error)},)}

    def _poll(self):
        for game in self.games.values():
            code = game.process.poll()
            if code is not None:
                if code != 0:
                    raise RuntimeError(f"owned game exited with error {code}")
                return {"closed": True, "termination": "game_closed"}
        events = []
        states = {name: self.games[name].clients["state"].read_status(.2) for name in self.menus}
        for name, menu in self.menus.items():
            events.extend(event | {"client": name} for event in menu.poll(states[name]))
        self.cursor, captures, dropped = self.history.read_latest(self.cursor)
        if dropped:
            events.append({"kind": "observation_resync", "skipped": dropped})
        records = []
        for capture in captures:
            characters = (capture.raw.p1.characterId, capture.raw.p2.characterId)
            if characters[self.seat] != self.character:
                raise RuntimeError("AI character differs from the selected opponent")
            if self.episode.observation_mode == "privileged_state":
                match = capture.match.match
                if match not in self.readers:
                    self.readers = {match: PrivilegedReader(capture.memory)}
                reader = self.readers[match]
                reader.memory = capture.memory
                observations = reader.observe_snapshot(capture.raw)
            elif self.episode.observation_mode == "diagnostic_state":
                observations = DiagnosticReader(capture.memory).observe_snapshot(capture.raw)
            else:
                observations = observe_visible_states(capture.raw, capture.render, self.episode.visibility)
            records.append({"frame": MatchFrame(capture.match, observations), "characters": characters,
                "engine_inputs": tuple(tuple(getattr(player.input, key) for key, _ in player.input._fields_)
                                       for player in (capture.raw.p1, capture.raw.p2))})
            self.seen_battle = True
        latest = self.games["ai"].clients["state"].read_status(.2)
        if not latest.in_battle:
            state = latest.match_state
            if state.phase == "disconnected":
                if self.seen_battle:
                    return {"closed": True, "termination": "disconnected"}
                state = replace(state, phase="loading")
            identity = (latest.scene, state.match, state.frame, state.scores)
            if identity != self.menu_identity:
                records.append({"frame": MatchFrame(state, ()), "characters": (), "engine_inputs": ()})
                self.menu_identity = identity
            self.games["ai"]._check_dialogs()
        else:
            self.menu_identity = ()
        current = self.input.status()
        if current is not None:
            identity = (current["acknowledged"], current["result"], current["applied"])
            if identity != self.input_identity:
                events.append({"kind": "input_status", **current})
                self.input_identity = identity
                if current["result"] == "invalid":
                    raise RuntimeError(f"invalid realtime input protocol: {current}")
        return {"closed": False, "records": records, "events": events}

    def submit(self, state, keys):
        return self.input.submit_current(state, keys, self.episode.latency_frames, self.episode.decision_frames)

    def close(self):
        self.stack.close()
