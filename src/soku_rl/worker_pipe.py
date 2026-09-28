"""Trusted local child-process transport between Linux learners and Wine."""
import os
from pathlib import Path
import pickle
import queue
import struct
import subprocess
import threading


PROTOCOL = 1
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


class WorkerBackend:
    """One persistent Python/Wine worker owns independently reset game slots."""
    def __init__(self, command, cwd, log_path, timeout, launch_timeout):
        if not command or timeout <= launch_timeout or launch_timeout <= 0:
            raise ValueError("worker timeout must exceed the positive launch timeout")
        self.timeout = timeout
        self.closed = False
        self.broken = False
        destination = Path(log_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        self.log = destination.open("ab", buffering=0)
        try:
            self.process = subprocess.Popen(
                list(command), cwd=cwd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=self.log, bufsize=0,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        except BaseException:
            self.log.close()
            raise
        self.replies = queue.Queue()
        self.lock = threading.Lock()
        threading.Thread(target=self._read_replies, daemon=True).start()
        try:
            self.identity = self._request("initialize", {"protocol": PROTOCOL, "launch_timeout": launch_timeout})
        except BaseException:
            self.close()
            raise

    def _read_replies(self):
        try:
            while True:
                self.replies.put(receive(self.process.stdout))
        except BaseException as error:
            self.replies.put(error)

    def _request(self, operation, payload):
        with self.lock:
            if self.closed or self.broken:
                raise RuntimeError("worker is closed or failed")
            try:
                send(self.process.stdin, (operation, payload))
                response = self.replies.get(timeout=self.timeout)
                if isinstance(response, BaseException):
                    raise response
                if not response["ok"]:
                    raise RuntimeError(response["error"])
                return response["value"]
            except BaseException:
                self.broken = True
                raise

    def reset_slots(self, seeds):
        return self._request("reset", seeds)

    def step(self, actions):
        return self._request("step", actions)

    def close(self):
        if self.closed:
            return
        try:
            if not self.broken and self.process.poll() is None:
                self._request("close", {})
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
            if self.process.returncode:
                raise RuntimeError(f"rollout worker exited with code {self.process.returncode}")
