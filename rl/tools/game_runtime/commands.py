"""Dispatch diagnostic game commands from the shared Hydra configuration space."""
from pathlib import Path
import json

import hydra


def run(config):
    command = config.launch.command
    if command not in {"practice", "vs", "replay", "anchor_save", "anchor_load", "script", "list", "status", "shutdown"}:
        raise ValueError(f"unknown game command: {command}")
    timeout = float(config.launch.timeout)
    if timeout <= 0:
        raise ValueError("launch.timeout must be positive")
    pid = config.launch.pid
    if pid is not None and (type(pid) is not int or pid <= 0):
        raise ValueError("launch.pid must be a positive integer or null")
    if command in {"anchor_save", "script", "shutdown"} and pid is None:
        raise ValueError(f"{command} requires an explicit launch.pid")
    # Import Windows bindings only when executing, so configuration composition
    # and command validation do not open or inspect game processes.
    import sokurl
    from game_runtime.startup import configure_game
    configure_game(config.runtime.game_directory)
    if command == "practice":
        return sokurl.practice(timeout, pid)
    if command == "vs":
        return sokurl.versus(timeout, config.launch.headless, config.launch.unlimited)
    if command == "replay":
        return sokurl.replay(Path(config.launch.path), config.launch.frame, timeout)
    if command == "list":
        return sokurl.list_instances()
    if command == "status":
        return sokurl.status(pid)
    if command == "shutdown":
        return sokurl.shutdown(timeout, pid)
    if command == "script":
        from scenario_runner import parse_script, run_script
        result = run_script(parse_script(Path(config.launch.path)), pid)
        print(json.dumps(result, indent=2))
        return 0
    from scenario_runner import anchor_path, load_anchor, save_anchor
    name = config.launch.name
    if command == "anchor_save":
        document = save_anchor(name, pid)
        print(json.dumps({"anchor": name, "path": str(anchor_path(name)),
            "target_frame": document["target_frame"], "target_hash": document["target_hash"]}, indent=2))
        return 0
    if pid is not None:
        sokurl.shutdown(timeout, pid)
    instance = load_anchor(name)
    try:
        snapshot = instance.client.snapshot()
        print(json.dumps({"anchor": name, "pid": instance.pid, "frame": snapshot.game_frame,
            "hash": f"{snapshot.latest.stateHash:016X}", "state": snapshot.run_state_name}, indent=2))
    finally:
        instance.client.close()
    print("ANCHOR_READY")
    return 0


@hydra.main(version_base="1.3", config_path=str(Path(__file__).resolve().parents[2] / "config"), config_name="game_control")
def main(config):
    result = run(config)
    if result:
        raise SystemExit(result)
