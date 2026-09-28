"""Use OpenSpiel PSROSolver with live vector rollouts instead of tree traversal."""
import json
from pathlib import Path

from open_spiel.python.algorithms.psro_v2.psro_v2 import PSROSolver

from .population import PopulationEvaluator, UniformPolicy
from .ppo_response import PPOResponseOracle


class PlayerRoles:
    """Metadata required by the sampled meta-solver, not a pyspiel game tree."""
    def num_players(self):
        return 2


class SampledPSROSolver(PSROSolver):
    def __init__(self, evaluator, oracle, simulations, prd_iterations):
        self.evaluator = evaluator
        super().__init__(
            PlayerRoles(), oracle, simulations,
            initial_policies=[UniformPolicy(f"uniform-p{p}", evaluator.env.single_action_space.n) for p in (0, 1)],
            rectifier="", training_strategy_selector="probabilistic",
            meta_strategy_method="prd", sample_from_marginals=True,
            number_policies_selected=1, symmetric_game=False,
            prd_iterations=prd_iterations,
        )

    def sample_episodes(self, policies, num_episodes):
        # Upstream's recursive state traversal needs a game tree and can exceed
        # Python recursion depth at 7200 frames. Live rollouts need neither.
        return self.evaluator.evaluate(policies, num_episodes)


def train_psro(env, config, device, seed, directory):
    if min(config["iterations"], config["simulations_per_entry"],
           config["prd_iterations"], config["response"]["timesteps_per_response"]) < 1:
        raise ValueError("PSRO iteration and sampling budgets must be positive")
    evaluator = PopulationEvaluator(env, seed, config["timeout_payoff"])
    oracle = PPOResponseOracle(env, config["response"], device, seed + 1, directory)
    solver = SampledPSROSolver(evaluator, oracle, config["simulations_per_entry"], config["prd_iterations"])
    report = {}
    for iteration in range(config["iterations"] + 1):
        if iteration:
            solver.iteration()
        report = {"iteration": iteration, "timeout_payoff": config["timeout_payoff"],
                  "meta_game": [v.tolist() for v in solver.get_meta_game()],
                  "meta_strategies": [v.tolist() for v in solver.get_meta_strategies()],
                  "populations": [[{"name": p.name, "fingerprint": p.fingerprint}
                                   for p in role] for role in solver.get_policies()],
                  "evaluation_games": evaluator.records}
        destination = Path(directory) / "population.json"
        temporary = destination.with_suffix(".pending.json")
        temporary.write_text(json.dumps(report, indent=2), encoding="utf-8")
        temporary.replace(destination)
    return report
