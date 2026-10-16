"""Offline checks for recording orchestration, provenance and failure boundaries."""
import hashlib
import json

import pytest
from omegaconf import OmegaConf

from soku_rl.replay import Replay, ReplayMatch, ReplayPlayer
from soku_rl.replay.review import build_catalog, recording_command, reviewed_replay


def recorded_run(tmp_path):
    players = tuple(ReplayPlayer(c, 0, tuple(range(20)), s, 0, 1)
                    for s, c in enumerate((1, 0)))
    replay = Replay(0xD2, bytes((0, 0, 9, 30, 0, 0, 0, 1, 3, 0)),
                    (ReplayMatch(players, 3, 0, 0, 1732, (0, 2, 0x300)),))
    path = tmp_path / "replays" / "worker" / "episode.rep"
    path.parent.mkdir(parents=True)
    data = replay.encode()
    path.write_bytes(data)
    path.with_suffix(".json").write_text(json.dumps({"seed": 1732, "slot": 0,
        "frames": 600, "reason": "p1_win", "replay": path.name,
        "sha256": hashlib.sha256(data).hexdigest()}))
    game = {"world_seed": 1732, "frames": 600, "outcome": "p1_win",
            "trial_id": "test-trial", "match": {f"player_{s}": {"character": c}
                                                  for s, c in enumerate((1, 0))}}
    config = {"runtime": {"fresh_reset": False, "record_replays": True,
                         "cards_disabled": True, "fixed_weather": "clear"},
              "checkpoint_sha256": "model-hash", "training_config_sha256": "config-hash"}
    OmegaConf.save(OmegaConf.create(config), tmp_path / "config.yaml")
    (tmp_path / "result.json").write_text(json.dumps({"success": True,
                                                    "result": {"games": [game]}}))
    return path, config, game


def test_recording_command_overrides_only_review_runtime(tmp_path):
    (tmp_path / "config.yaml").write_text("{}")
    (tmp_path / "final.zip").write_bytes(b"model")
    cmd = recording_command("python", "run.py", tmp_path, tmp_path / "out",
                            "final.zip", [1732])
    assert "runtime.fresh_reset=false" in cmd
    assert "runtime.record_replays=true" in cmd
    assert "require_complete=false" in cmd
    assert 'opponent_names=["pressure"]' in cmd
    assert '+benchmark.learner_seats=[0]' in cmd
    assert not any("reward" in a or "max_frames" in a for a in cmd)
    with pytest.raises(ValueError, match="unique"):
        recording_command("python", "run.py", tmp_path, tmp_path / "out",
                          "final.zip", [1732, 1732])
    with pytest.raises(FileExistsError):
        recording_command("python", "run.py", tmp_path, tmp_path,
                          "final.zip", [1732])


def test_catalog_preserves_bytes_and_binds_trial_without_claiming_parity(tmp_path):
    path, _, game = recorded_run(tmp_path)
    before = path.read_bytes()
    catalog = build_catalog(tmp_path)
    assert catalog["status"] == "SAVED_NOT_PLAYBACK_VERIFIED"
    assert catalog["games"][0]["trial"] == game
    assert catalog["games"][0]["playback_parity"] == "NOT TESTED"
    assert path.read_bytes() == before
    manifest = tmp_path / "catalog.json"
    manifest.write_text(json.dumps(catalog))
    assert reviewed_replay(manifest, 0)[0] == path
    with pytest.raises(ValueError, match="index"):
        reviewed_replay(manifest, -1)
    catalog["games"][0]["playback_parity"] = "FAIL"
    manifest.write_text(json.dumps(catalog))
    with pytest.raises(ValueError, match="known replay mismatch"):
        reviewed_replay(manifest, 0)
    catalog["games"][0]["playback_parity"] = "NOT TESTED"
    manifest.write_text(json.dumps(catalog))
    path.write_bytes(before + b"changed")
    with pytest.raises(ValueError, match="hash"):
        reviewed_replay(manifest, 0)


@pytest.mark.parametrize("fault", ["hash", "seed", "missing", "ambiguous", "fresh", "failed"])
def test_bad_recordings_fail_closed(tmp_path, fault):
    path, config, game = recorded_run(tmp_path)
    result = {"success": True, "result": {"games": [game]}}
    if fault in ("hash", "seed"):
        metadata = json.loads(path.with_suffix(".json").read_text())
        metadata["sha256" if fault == "hash" else "seed"] = "bad" if fault == "hash" else 99
        path.with_suffix(".json").write_text(json.dumps(metadata))
    elif fault == "missing":
        path.unlink()
    elif fault == "ambiguous":
        result["result"]["games"].append(game.copy())
    elif fault == "fresh":
        config["runtime"]["fresh_reset"] = True
        OmegaConf.save(OmegaConf.create(config), tmp_path / "config.yaml")
    else:
        result["success"] = False
    (tmp_path / "result.json").write_text(json.dumps(result))
    with pytest.raises(ValueError):
        build_catalog(tmp_path)


@pytest.mark.parametrize("weather_ok", [True, False])
def test_viewer_uses_native_run_and_checks_weather_without_input_injection(tmp_path, monkeypatch, weather_ok):
    from pathlib import Path
    import runpy
    import sys
    from types import SimpleNamespace
    import sokurl_platform.replay

    recorded_run(tmp_path)
    catalog = tmp_path / "catalog.json"
    catalog.write_text(json.dumps(build_catalog(tmp_path)))
    calls = []

    class Client:
        def set_fixed_weather(self, weather):
            calls.append(("weather", weather))
            return 1

        def wait_for_ack(self, seq, timeout):
            return SimpleNamespace(ack_seq=seq, result_name="CHECKPOINT_INVALIDATED"
                                   if seq == 1 else "ACCEPTED")

        def run(self):
            calls.append(("run",))
            return 2

        def snapshot(self):
            return SimpleNamespace(game_frame=120, latest=SimpleNamespace(
                activeWeather=21 if weather_ok else 0, displayedWeather=21,
                p1=SimpleNamespace(hp=10000), p2=SimpleNamespace(hp=9000)))

        def drain_frames(self):
            pass

    game = SimpleNamespace(client=Client(), process=SimpleNamespace(is_running=lambda: True),
                           close=lambda: calls.append(("close",)))
    monkeypatch.setattr(sokurl_platform.replay, "launch_replay", lambda *a, **kw: game)
    monkeypatch.setattr(sys, "argv", ["watch_battle", "--catalog", str(catalog),
                                     "--preview-frames", "120"])
    main = runpy.run_path(str(Path(__file__).parents[1] / "tools/watch_battle.py"))["main"]
    if weather_ok:
        main()
    else:
        with pytest.raises(RuntimeError, match="requested weather"):
            main()
    assert calls == [("weather", "clear"), ("run",), ("close",)]
