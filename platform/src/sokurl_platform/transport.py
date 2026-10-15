"""Trusted local child-process transport between Linux learners and Wine."""
import os
from pathlib import Path
import pickle
import queue
import struct
import subprocess
import threading
import hashlib
import json
from uuid import uuid4


PROTOCOL = 4
MAX_MESSAGE = 64 * 1024 * 1024


def _read_exact(stream, size):
    chunks = bytearray()
    while len(chunks) < size:
        chunk = stream.read(size - len(chunks))
        if not chunk:
            raise EOFError("rollout worker pipe closed")
        chunks.extend(chunk)
    return bytes(chunks)


def receive(stream):
    size, = struct.unpack("!I", _read_exact(stream, 4))
    if not 0 < size <= MAX_MESSAGE:
        raise ValueError("invalid worker message size")
    # Only decode messages from the child we started. Never expose this to a socket.
    return pickle.loads(_read_exact(stream, size))


def send(stream, value):
    data = pickle.dumps(value, protocol=5)
    if len(data) > MAX_MESSAGE:
        raise ValueError("worker message exceeds size limit")
    packet = memoryview(struct.pack("!I", len(data)) + data)
    while packet:
        written = stream.write(packet)
        if not written:
            raise BrokenPipeError("worker message write made no progress")
        packet = packet[written:]
    stream.flush()


class WorkerConnection:
    """Exchange bounded requests with one owned Python/Wine worker."""
    def __init__(self, command, cwd, log_path, timeout, launch_timeout, mute_audio, game_directory,
                 fresh_reset=False, record_replays=False, cards_disabled=False, fixed_weather=None,
                 combat_events=False, native_input_counters=False):
        if not command or timeout <= launch_timeout or launch_timeout <= 0:
            raise ValueError("worker timeout must exceed the positive launch timeout")
        if type(mute_audio) is not bool:
            raise TypeError("mute_audio must be a boolean")
        if not isinstance(game_directory, str) or not game_directory:
            raise ValueError("a worker game directory is required")
        if any(type(value) is not bool for value in (fresh_reset, record_replays, cards_disabled, combat_events,native_input_counters)):
            raise TypeError("reset/recording/cards switches must be booleans")
        if fixed_weather not in (None, "clear", "typhoon"):
            raise ValueError("unsupported fixed weather")
        if native_input_counters and not combat_events:
            raise ValueError('experimental native input counters require combat sidecar capability verification')
        if fresh_reset and (not cards_disabled or fixed_weather != "clear" or record_replays):
            raise ValueError("fresh reset requires Clear/cards-disabled and recording OFF in this integration")
        self.timeout = timeout
        self.closed = False
        self.broken = False
        destination = Path(log_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        self.log = destination.open("ab", buffering=0)
        try:
            environment = os.environ.copy()
            environment["SOKURL_MUTE_AUDIO"] = "1" if mute_audio else "0"
            environment["SOKURL_FRESH_RESET"] = "1" if fresh_reset else "0"
            environment['SOKURL_COMBAT_EVENTS'] = '1' if combat_events else '0'
            environment['SOKURL_NATIVE_INPUT_COUNTERS'] = '1' if native_input_counters else '0'
            self.process = subprocess.Popen(
                list(command), cwd=cwd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=self.log, bufsize=0, env=environment,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        except BaseException:
            self.log.close()
            raise
        self.replies = queue.Queue()
        self.lock = threading.Lock()
        threading.Thread(target=self._read_replies, daemon=True).start()
        try:
            self.identity = self.request("initialize", {"protocol": PROTOCOL,
                "launch_timeout": launch_timeout, "game_directory": game_directory,
                "fresh_reset": fresh_reset, "record_replays": record_replays,
                "cards_disabled": cards_disabled, "fixed_weather": fixed_weather})
        except BaseException:
            self.close()
            raise

    def _read_replies(self):
        try:
            while True:
                self.replies.put(receive(self.process.stdout))
        except BaseException as error:
            self.replies.put(error)

    def request(self, operation, payload):
        with self.lock:
            if self.closed or self.broken:
                raise RuntimeError("worker is closed or failed")
            try:
                send(self.process.stdin, (operation, payload))
                response = self._read_response(operation)
                if isinstance(response, BaseException):
                    raise response
                return self._response_value(response)
            except BaseException:
                self.broken = True
                raise

    def _read_response(self, operation):
        return self.replies.get(timeout=self.timeout)

    def _response_value(self, response):
        if not response["ok"]:
            raise RuntimeError(response["error"])
        return response["value"]

    def close(self):
        if self.closed:
            return
        try:
            if not self.broken and self.process.poll() is None:
                self.request("close", {})
        finally:
            self.closed = True
            self.process.stdin.close()
            # EOF lets the worker finish its bounded operation and clean up games.
            # Do not silently kill the worker and leave its game processes orphaned.
            try:
                self.process.wait(timeout=self.timeout)
            finally:
                self.log.close()
            self.process.stdout.close()
            # A failed request already reported the worker's original exception.
            # Cleanup must not replace it with an exit-code-only exception.
            if self.process.returncode and not self.broken:
                raise RuntimeError(f"worker exited with code {self.process.returncode}")
