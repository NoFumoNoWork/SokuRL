"""Serve trusted parent requests using the existing Windows game controller."""
import sys
import traceback

from soku_rl.worker_pipe import PROTOCOL, receive, send
from game_batch import SokuGameBatch
from runtime_identity import fingerprints


def main():
    requests, replies = sys.stdin.buffer, sys.stdout.buffer
    # Human logs must not corrupt the framed binary replies.
    sys.stdout = sys.stderr
    operation, config = receive(requests)
    if operation != "initialize" or config["protocol"] != PROTOCOL:
        raise ValueError("unsupported rollout worker protocol")
    backend = SokuGameBatch(config["launch_timeout"])
    send(replies, {"ok": True, "value": {"protocol": PROTOCOL, "reset": "native_scene_reload",
                                         "fingerprints": fingerprints()}})
    try:
        while True:
            try:
                operation, payload = receive(requests)
            except EOFError:
                break
            try:
                if operation == "configure_observation":
                    backend.configure_observation(payload)
                    value = {"observation_mode": payload}
                elif operation == "reset":
                    value = backend.reset_slots(payload)
                elif operation == "step":
                    value = backend.step(payload)
                elif operation == "close":
                    backend.close()
                    send(replies, {"ok": True, "value": {}})
                    break
                else:
                    raise ValueError(f"unsupported worker operation: {operation}")
                send(replies, {"ok": True, "value": value})
            except Exception:
                send(replies, {"ok": False, "error": traceback.format_exc()})
                raise
    finally:
        backend.close()


if __name__ == "__main__":
    main()
