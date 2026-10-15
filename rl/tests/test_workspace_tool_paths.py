"""Profiling tools belong to RL; they must not launch a Platform-side trainer."""
from pathlib import Path


def test_legacy_profilers_resolve_learning_root():
    tools = Path(__file__).resolve().parents[1] / 'tools'
    for name in ('benchmark_ppo_ab', 'benchmark_observation_ab',
                 'profile_environment_interaction', 'profile_training_sampling'):
        text = (tools / f'{name}.py').read_text(encoding='utf-8')
        assert 'root = Path(os.environ["SOKURL_RL_ROOT"])' in text
        assert 'root / "rl/runs' not in text
        assert 'root / "th123_jp"' not in text
