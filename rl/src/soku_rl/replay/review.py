"""Review metadata over unchanged Platform replay encoding and evaluation."""
import hashlib
import json
from pathlib import Path

from sokurl_platform.replay_format import Replay


def recording_command(python, launcher, training, output, checkpoint, seeds,
                      opponents="pressure", num_envs=1, device="cpu", seat="left"):
    training = Path(training).resolve(strict=True)
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    if not (training / "config.yaml").is_file():
        raise ValueError("training config.yaml is required")
    model = (training / checkpoint).resolve(strict=True)
    if not model.is_relative_to(training):
        raise ValueError("checkpoint must belong to training directory")
    if not seeds or any(type(s) is not int or not 0 <= s < 0xFFFFFFFF for s in seeds):
        raise ValueError("nonempty uint32 seeds below the native reserved value required")
    if len(set(seeds)) != len(seeds) or not 1 <= num_envs <= 4:
        raise ValueError("unique seeds and one to four review workers required")
    if not opponents or any(c in opponents for c in "\n\r=+{}[]"):
        raise ValueError("use a single opponent name or all")
    if seat not in ("left", "right", "both"):
        raise ValueError("seat must be left, right or both")
    names = "all" if opponents == "all" else json.dumps([opponents])
    seats = {"left": [0], "right": [1], "both": [0, 1]}[seat]
    return [str(python), "-B", str(launcher), "benchmark_br",
            "--config-name", "benchmark_combat_v2",
            f"training_directory={training.as_posix()}", f"checkpoint={checkpoint}",
            f"output={output.as_posix()}", f"device={device}", f"num_envs={num_envs}",
            "require_complete=false",
            f"opponent_names={names}", f"benchmark.world_seeds={json.dumps(seeds)}",
            f"+benchmark.learner_seats={json.dumps(seats)}",
            "runtime.fresh_reset=false", "runtime.record_replays=true"]


def build_catalog(directory):
    """Bind original REP bytes to completed trials; never infer playback parity."""
    directory = Path(directory).resolve(strict=True)
    from omegaconf import OmegaConf
    config = OmegaConf.to_container(OmegaConf.load(directory / "config.yaml"), resolve=True)
    result = json.loads((directory / "result.json").read_text(encoding="utf-8"))
    runtime = config["runtime"]
    if runtime["fresh_reset"] or not runtime["record_replays"]:
        raise ValueError("catalog requires explicit native recording without fresh reset")
    games = result.get("result", {}).get("games", [])
    records, used = [], set()
    for path in sorted((directory / "replays").rglob("*.rep")):
        data = path.read_bytes()
        metadata = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        digest = hashlib.sha256(data).hexdigest()
        if digest != metadata["sha256"] or metadata["replay"] != path.name:
            raise ValueError(f"replay identity mismatch: {path}")
        replay = Replay.decode(data)
        if len(replay.matches) != 1 or not replay.matches[0].input_words:
            raise ValueError(f"expected one nonempty native episode: {path}")
        match = replay.matches[0]
        if match.seed != metadata["seed"]:
            raise ValueError(f"replay seed mismatch: {path}")
        characters = [p.character for p in match.players]
        candidates = [(i, g) for i, g in enumerate(games)
                      if g["world_seed"] == match.seed and g["frames"] == metadata["frames"]
                      and [g["match"][f"player_{s}"]["character"] for s in range(2)] == characters
                      and g["outcome"] == metadata["reason"]]
        if len(candidates) != 1 or candidates[0][0] in used:
            raise ValueError(f"cannot uniquely bind REP to completed trial: {path}")
        index, game = candidates[0]
        used.add(index)
        records.append({"replay": path.relative_to(directory).as_posix(), "sha256": digest,
                        "frames": metadata["frames"], "trial": game,
                        "input_boundary": metadata.get("input_boundary", "UNPROTECTED_LEGACY"),
                        "input_word_count": len(match.input_words),
                        "playback_parity": "NOT TESTED"})
    if len(used) != len(games) or not records or result.get("success") is not True:
        raise ValueError("missing recordings or incomplete evaluation; preserve run for inspection")
    return {"status": "SAVED_NOT_PLAYBACK_VERIFIED", "format": "TH123 native REP",
            "conditions": {"cards_disabled": runtime["cards_disabled"],
                           "fixed_weather": runtime["fixed_weather"]},
            "checkpoint_sha256": config["checkpoint_sha256"],
            "training_config_sha256": config["training_config_sha256"],
            "runtime_identity": result.get("runtime"), "games": records,
            "limitations": ["Dedicated evaluation, not a historical fast-reset training episode.",
                            "Protected recording uses neutral startup and native key latching; legacy training injection is unchanged.",
                            "Action NPZ stores requested commands; pre-ready commands are neutralized.",
                            "Logical action NPZ is not the native input poll stream.",
                            "Reward totals are trial metadata; no per-frame attention captured.",
                            "Use the recorded weather/card conditions, not an unmodified vanilla launch."]}


def reviewed_replay(catalog_path, index):
    catalog_path = Path(catalog_path).resolve(strict=True)
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    if type(index) is not int or not 0 <= index < len(catalog["games"]):
        raise ValueError("game index out of range")
    entry = catalog["games"][index]
    if entry.get("playback_parity") == "FAIL":
        raise ValueError("known replay mismatch: do not use this recording to judge the checkpoint")
    if type(entry["frames"]) is not int or entry["frames"] < 1:
        raise ValueError("positive recording frame count required")
    path = (catalog_path.parent / entry["replay"]).resolve(strict=True)
    if not path.is_relative_to(catalog_path.parent):
        raise ValueError("replay escapes catalog directory")
    if hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
        raise ValueError("replay hash changed")
    Replay.decode(path.read_bytes())
    return path, entry, catalog["conditions"]
