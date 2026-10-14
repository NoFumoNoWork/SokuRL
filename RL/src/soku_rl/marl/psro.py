"""Use OpenSpiel PSROSolver with live vector rollouts instead of tree traversal."""
import json
import hashlib
from pathlib import Path
import numpy as np

from open_spiel.python.algorithms.psro_v2.psro_v2 import PSROSolver

from soku_rl.evaluation.population import PopulationEvaluator
from soku_rl.policy.population import UniformPolicy, PPOPolicy
from soku_rl.marl.response import PPOResponseOracle
from soku_rl.policy.recurrent import RecurrentPPOPolicy
from soku_rl.policy.dqn import DQNPolicy
from soku_rl.env.encoding import AGENTS
from soku_rl.policy.contract import read_training_contract
from soku_rl.policy.loader import load_policy


class PlayerRoles:
    """Metadata required by the sampled meta-solver, not a pyspiel game tree."""
    def num_players(self):
        return 2


def policy_artifact(policy, directory):
    metadata = {"name": policy.name, "fingerprint": policy.fingerprint}
    if isinstance(policy, UniformPolicy):
        return metadata | {"kind": "uniform", "num_actions": int(policy.num_actions)}
    if isinstance(policy, PPOPolicy):
        kind = ("sb3_dqn" if isinstance(policy, DQNPolicy) else
                "sb3_recurrent" if isinstance(policy, RecurrentPPOPolicy) else "sb3")
        return metadata | {"kind": kind, "path": str(policy.path.relative_to(directory))}
    raise TypeError("unsupported PSRO population member")


def initial_policies(config, interface, device, directory):
    """Validate both seat contracts, then retain independent checkpoint copies."""
    if set(config) != set(AGENTS):
        raise ValueError("PSRO initial population requires both player roles")
    policies = []
    for index, seat in enumerate(AGENTS):
        spec = config[seat]
        if spec == {"kind": "uniform"}:
            policy = UniformPolicy(f"uniform-p{index}", interface.action_space.n)
        elif spec["kind"] in {"sb3", "sb3_recurrent", "sb3_dqn"}:
            policy = load_policy(f"initial-p{index}", spec, interface, device)
        else:
            raise ValueError("initial policies must be uniform, sb3, or sb3_recurrent")
        policies.append(policy)
    return [retain_policy(policy, directory) for policy in policies]


def retain_policy(policy, directory):
    if not isinstance(policy, PPOPolicy):
        return policy
    data = policy.path.read_bytes()
    if hashlib.sha256(data).hexdigest() != policy.fingerprint:
        raise ValueError("checkpoint changed while loading")
    path = Path(directory) / (policy.name + ".zip")
    with path.open("xb") as stream:
        stream.write(data)
    return type(policy)(policy.name, policy.model, path)


def restore_population(config, interface, device, directory):
    resume = config["resume"]
    if set(resume) != {"kind", "path", "training_config"} or resume["kind"] != "checkpoint":
        raise ValueError("PSRO continuation requires a complete population checkpoint")
    previous = read_training_contract(resume["training_config"], interface)["algorithm"]
    for key in ("response", "simulations_per_entry", "prd_iterations", "timeout_payoff"):
        if previous[key] != config[key]:
            raise ValueError(f"PSRO continuation changed {key}")
    saved = json.loads(Path(resume["path"]).read_text(encoding="utf-8"))
    if "training_state" not in saved:
        raise ValueError("this old population supports inference but lacks continuation state")
    policies = []
    for seat in AGENTS:
        mixture = load_policy(seat, {"kind": "psro_mixture", "path": resume["path"],
            "training_config": resume["training_config"], "player": seat}, interface, device)
        policies.append([retain_policy(policy, directory) for policy in mixture.members])
    return policies, saved


class SampledPSROSolver(PSROSolver):
    def __init__(self, evaluator, oracle, simulations, prd_iterations, policies, saved):
        self.evaluator, self.saved = evaluator, saved
        self.restoring = bool(saved)
        super().__init__(
            PlayerRoles(), oracle, simulations,
            initial_policies=policies,
            rectifier="", training_strategy_selector="probabilistic",
            meta_strategy_method="prd", sample_from_marginals=True,
            number_policies_selected=1, symmetric_game=False,
            prd_iterations=prd_iterations,
        )

    def _initialize_policy(self, initial_policies):
        if self.restoring:
            self._policies, self._new_policies = initial_policies, [[], []]
        else:
            super()._initialize_policy(initial_policies)

    def _initialize_game_state(self):
        if self.restoring:
            shape = tuple(len(role) for role in self._policies)
            self._meta_games = [np.asarray(value, dtype=np.float64) for value in self.saved["meta_game"]]
            if len(self._meta_games) != 2 or any(v.shape != shape or not np.isfinite(v).all() for v in self._meta_games):
                raise ValueError("saved PSRO payoff tables differ from the populations")
            self._iterations = self.saved["iteration"]
        else:
            super()._initialize_game_state()

    def update_meta_strategies(self):
        if self.restoring:
            self._meta_strategy_probabilities = [np.asarray(v) for v in self.saved["meta_strategies"]]
            self._non_marginalized_probabilities = np.asarray(self.saved["training_state"]["joint_probabilities"])
            self.restoring = False
        else:
            super().update_meta_strategies()

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
    if config["resume"] == {"kind": "fresh"}:
        policies = initial_policies(config["initial_population"], env.interface, device, directory)
        saved = {}
    else:
        policies, saved = restore_population(config, env.interface, device, directory)
    solver = SampledPSROSolver(evaluator, oracle, config["simulations_per_entry"], config["prd_iterations"], policies, saved)
    first = saved["iteration"] if saved else 0
    if saved:
        state = saved["training_state"]
        evaluator.rng.bit_generator.state = state["evaluation_rng"]
        evaluator.records = saved["evaluation_games"]
        oracle.rng.bit_generator.state = state["oracle_rng"]
        oracle.responses = state["responses"]
        rng = state["selection_rng"]
        np.random.set_state((rng[0], np.asarray(rng[1], dtype=np.uint32), *rng[2:]))
    report = {}
    for iteration in range(first, first + config["iterations"] + 1):
        if iteration > first:
            solver.iteration()
        rng = np.random.get_state()
        report = {"format": "sokurl-psro-population-v1",
                  "iteration": iteration, "timeout_payoff": config["timeout_payoff"],
                  "meta_game": [v.tolist() for v in solver.get_meta_game()],
                  "meta_strategies": [v.tolist() for v in solver.get_meta_strategies()],
                  "populations": [[policy_artifact(p, directory)
                                   for p in role] for role in solver.get_policies()],
                  "evaluation_games": evaluator.records,
                  "training_state": {"evaluation_rng": evaluator.rng.bit_generator.state,
                    "oracle_rng": oracle.rng.bit_generator.state, "responses": oracle.responses,
                    "selection_rng": [rng[0], rng[1].tolist(), *rng[2:]],
                    "joint_probabilities": np.asarray(solver._non_marginalized_probabilities).tolist()}}
        destination = Path(directory) / "population.json"
        temporary = destination.with_suffix(".pending.json")
        temporary.write_text(json.dumps(report, indent=2), encoding="utf-8")
        temporary.replace(destination)
    return report
