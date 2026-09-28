"""Keep model selection and final evaluation seed blocks disjoint."""
from pathlib import Path

from hydra import compose, initialize_config_dir


def test_benchmark_composes_distinct_validation_and_test_seeds():
    root = Path(__file__).resolve().parents[1] / "config"
    with initialize_config_dir(version_base="1.3", config_dir=str(root)):
        validation = compose(config_name="benchmark")
        final = compose(config_name="benchmark", overrides=["evaluation=test"])
    assert validation.evaluation.name == "validation"
    assert final.evaluation.name == "test"
    left, right = set(validation.benchmark.world_seeds), set(final.benchmark.world_seeds)
    assert len(left) == len(right) == 32
    assert not left & right
    assert validation.benchmark.policy_seed != final.benchmark.policy_seed
    assert validation.benchmark.opponents == validation.rules.roster
