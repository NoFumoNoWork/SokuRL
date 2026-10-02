from __future__ import annotations

import argparse
import json
from pathlib import Path

from ken43 import (
    AlwaysBlockBot,
    AlwaysDIBot,
    AlwaysJinraiBot,
    AlwaysMashBot,
    AlwaysODDPBot,
    AlwaysThrowBot,
    BackdashHeavyBot,
    HeuristicDefenderBot,
    HeuristicOffenseBot,
    Ken43GymEnv,
    MixedBot,
    RandomBot,
)


BOT_TYPES = {
    "block": AlwaysBlockBot,
    "mash": AlwaysMashBot,
    "throw": AlwaysThrowBot,
    "od_dp": AlwaysODDPBot,
    "di": AlwaysDIBot,
    "jinrai": AlwaysJinraiBot,
    "backdash_heavy": BackdashHeavyBot,
    "random": RandomBot,
    "mixed": MixedBot,
    "heuristic": HeuristicDefenderBot,
    "heuristic_offense": HeuristicOffenseBot,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train MaskablePPO on Ken43")
    parser.add_argument("--bot", choices=sorted(BOT_TYPES), default="block")
    parser.add_argument("--role", choices=("attacker", "defender"), default="attacker")
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--episode-offset", type=int, default=0)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--drive-reward-weight", type=float, default=1.0)
    parser.add_argument("--timeout-penalty", type=float, default=0.5)
    parser.add_argument("--time-pressure-start", type=int, default=360)
    parser.add_argument("--time-pressure-total", type=float, default=0.5)
    parser.add_argument("--time-pressure-power", type=float, default=2.0)
    parser.add_argument("--bootstrap-timeouts", action="store_true")
    parser.add_argument("--gamma", type=float, default=0.999)
    parser.add_argument("--gae-lambda", type=float, default=0.97)
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--checkpoint-every-episodes", type=int, default=500)
    parser.add_argument("--output", type=Path, default=Path("Ken43/checkpoints/fixed_bot"))
    args = parser.parse_args()
    if args.episodes is not None and args.episodes <= 0:
        parser.error("--episodes must be positive")
    if args.episode_offset < 0:
        parser.error("--episode-offset must be non-negative")
    if args.episodes is not None and args.episode_offset >= args.episodes:
        parser.error("--episode-offset must be smaller than --episodes")
    if args.checkpoint_every_episodes < 0:
        parser.error("--checkpoint-every-episodes must be non-negative")
    if args.num_envs <= 0:
        parser.error("--num-envs must be positive")

    try:
        from sb3_contrib import MaskablePPO
        from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
        from stable_baselines3.common.callbacks import BaseCallback
        from stable_baselines3.common.monitor import Monitor
        from stable_baselines3.common.vec_env import SubprocVecEnv
    except ImportError as exc:
        raise SystemExit(
            "PPO dependencies are not installed. Run: "
            r".\.venv\python.exe -m pip install -e .[ppo]"
        ) from exc

    def make_env():
        return Ken43GymEnv(
            opponent=BOT_TYPES[args.bot](),
            learning_agent="player_0" if args.role == "attacker" else "player_1",
            max_frames=600,
            drive_reward_weight=args.drive_reward_weight,
            timeout_penalty=args.timeout_penalty,
            time_pressure_start=args.time_pressure_start,
            time_pressure_total=args.time_pressure_total,
            time_pressure_power=args.time_pressure_power,
            bootstrap_timeouts=args.bootstrap_timeouts,
        )

    if args.num_envs == 1:
        env = make_env()
    else:
        env = SubprocVecEnv(
            [lambda: Monitor(make_env()) for _ in range(args.num_envs)],
            start_method="spawn",
        )

    class EpisodeLimitCallback(BaseCallback):
        def __init__(
            self,
            episode_limit: int | None,
            episode_offset: int,
            checkpoint_every: int,
            output: Path,
        ):
            super().__init__()
            self.episode_limit = episode_limit
            self.checkpoint_every = checkpoint_every
            self.output = output
            self.episodes = episode_offset
            self.next_checkpoint = (
                ((episode_offset // checkpoint_every) + 1) * checkpoint_every
                if checkpoint_every > 0 else 0
            )

        def _on_step(self) -> bool:
            self.episodes += int(sum(bool(done) for done in self.locals.get("dones", [])))
            while (
                self.next_checkpoint > 0
                and self.episodes > 0
                and self.episodes >= self.next_checkpoint
            ):
                self.output.mkdir(parents=True, exist_ok=True)
                self.model.save(
                    self.output / f"ppo_{args.role}_{args.bot}_episode_{self.next_checkpoint}"
                )
                self.next_checkpoint += self.checkpoint_every
            return self.episode_limit is None or self.episodes < self.episode_limit

    callback = EpisodeLimitCallback(
        args.episodes,
        args.episode_offset,
        args.checkpoint_every_episodes,
        args.output,
    )
    if args.resume is not None:
        model = MaskablePPO.load(args.resume, env=env, device=args.device)
    else:
        model = MaskablePPO(
            MaskableActorCriticPolicy,
            env,
            seed=args.seed,
            verbose=1,
            device=args.device,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
        )
    model.learn(
        total_timesteps=args.timesteps,
        callback=callback,
        reset_num_timesteps=args.resume is None,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    run_label = f"episode_{args.episodes}" if args.episodes is not None else str(args.timesteps)
    checkpoint = args.output / f"ppo_{args.role}_{args.bot}_{run_label}"
    model.save(checkpoint)
    config = {
        "environment": "ken43_corner_drive_v2",
        "action_interface": "flat_discrete_complete_commands",
        "bot": args.bot,
        "learning_role": args.role,
        "timesteps_requested": args.timesteps,
        "episodes_requested": args.episodes,
        "episode_offset": args.episode_offset,
        "episodes_completed": callback.episodes,
        "resumed_from": str(args.resume) if args.resume is not None else None,
        "seed": args.seed,
        "max_frames": 600,
        "timeout_penalty": args.timeout_penalty,
        "time_pressure": {
            "start_frame": args.time_pressure_start,
            "total_at_timeout": args.time_pressure_total,
            "power": args.time_pressure_power,
        },
        "bootstrap_timeouts": args.bootstrap_timeouts,
        "gamma": args.gamma,
        "gae_lambda": args.gae_lambda,
        "num_envs": args.num_envs,
        "device": args.device,
        "drive_reward": {
            "mode": "signed_net_drive_delta",
            "weight": args.drive_reward_weight,
        },
        "drive_regen": {
            "per_frame": 0.0025,
            "delay_frames": 90,
            "forward_walk_after_frames": 11,
            "forward_walk_multiplier": 1.5,
        },
        "action_space_n": int(env.action_space.n),
    }
    checkpoint.with_suffix(".json").write_text(
        json.dumps(config, indent=2) + "\n",
        encoding="ascii",
    )
    env.close()


if __name__ == "__main__":
    main()
