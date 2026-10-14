from dataclasses import replace

import numpy as np
import pytest

from test_env_timing import RecordingBackend, VISIBILITY
from soku_rl.env import EpisodeConfig, HisoutenParallelEnv, TwoPlayerVectorEnv
from soku_rl.env.encoding import AGENTS
from soku_rl.env.match import LEGACY_MATCH
from soku_rl.pomg import Outcome


@pytest.mark.parametrize("terminal_frame", [None, 1, 2, 3, 5])
@pytest.mark.parametrize("latency", [0, 5])
def test_deferred_results_match_eager_with_early_terminal(terminal_frame, latency):
    class Backend(RecordingBackend):
        def _state(self, slot):
            state = super()._state(slot)
            if terminal_frame is not None and state.frame == terminal_frame:
                return replace(state, outcome=Outcome.P1_WIN, rewards=(1., -1.))
            return state
    config = EpisodeConfig(7, 4, 3, latency, "diagnostic_state", VISIBILITY, LEGACY_MATCH)
    single_backend, vector_backend = Backend(), Backend()
    single = HisoutenParallelEnv(single_backend, config)
    vector = TwoPlayerVectorEnv(vector_backend, 2, config)
    single.reset(seed=1)
    vector.reset({0: 1, 1: 2})
    while single.agents:
        actions = dict.fromkeys(AGENTS, 0)
        eager = single.step(actions)
        deferred = vector.step({0: actions})
        for agent in AGENTS:
            np.testing.assert_array_equal(eager[0][agent], deferred[0][0][agent])
        assert eager[1:] == tuple(v[0] for v in deferred[1:])
    assert single_backend.inputs[0] == vector_backend.inputs[0]
    assert vector_backend.frames[1] == 0
    # Partial reset clears history and delayed controls, without touching slot1.
    vector.reset({0: 3})
    assert vector.episodes[0].frame == 0 and vector_backend.frames[1] == 0
