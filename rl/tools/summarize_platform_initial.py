"""Compare two completed paired evaluations without treating timeouts as wins."""
import argparse
import json
from pathlib import Path
import statistics


def summarize(path):
    report = json.loads((path / "result.json").read_text(encoding="utf-8"))
    if report["success"] is not True:
        raise ValueError(f"evaluation failed: {path}")
    games = report["result"]["games"]
    counts = {"win": 0, "loss": 0, "time_limit": 0, "double_ko": 0}
    dealt, received = [], []
    for game in games:
        seat, outcome = game["learner_seat"], game["outcome"]
        if outcome in ("time_limit", "double_ko"):
            counts[outcome] += 1
        elif outcome == ("p1_win" if seat == 0 else "p2_win"):
            counts["win"] += 1
        else:
            counts["loss"] += 1
        hp = game["public_hp_loss_fraction_by_seat"]
        received.append(hp[seat])
        dealt.append(hp[1 - seat])
    return {"games": len(games), "counts": counts,
            "mean_public_hp_loss_dealt": statistics.mean(dealt),
            "mean_public_hp_loss_received": statistics.mean(received),
            "mean_public_hp_advantage": statistics.mean(dealt) - statistics.mean(received),
            "seconds": report["seconds"]}, games


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline", type=Path)
    parser.add_argument("final", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    before, a = summarize(args.baseline)
    after, b = summarize(args.final)
    key = lambda g: (g["world_seed"], g["learner_seat"], tuple(g["policy_seeds"]),
                     json.dumps(g["match"], sort_keys=True))
    if sorted(key(g) for g in a) != sorted(key(g) for g in b):
        raise ValueError("evaluation world/seat/actor seeds or matches differ")
    result = {"paired_conditions": "PASS", "initial": before, "final": after,
              "scope": "four validation seeds, two seats, same rush opponent; not a strength gate",
              "hp_metric": "quantized public HP-bar decreases, not native damage/hit counts"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
