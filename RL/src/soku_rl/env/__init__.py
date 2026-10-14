"""Public simultaneous two-player environment interfaces."""
def __getattr__(name):
    # The Wine worker imports match configuration without NumPy or Gymnasium.
    if name == "TwoPlayerVectorEnv":
        from soku_rl.env.vector_env import TwoPlayerVectorEnv
        return TwoPlayerVectorEnv
    if name in {"EpisodeConfig", "HisoutenParallelEnv"}:
        from . import hisouten_env
        return getattr(hisouten_env, name)
    raise AttributeError(name)

__all__ = ["EpisodeConfig", "HisoutenParallelEnv", "TwoPlayerVectorEnv"]
