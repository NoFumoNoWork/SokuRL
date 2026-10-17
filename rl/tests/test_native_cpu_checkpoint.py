from dataclasses import replace

import numpy as np

from native_cpu_benchmark import CheckpointControls
from sokurl_platform.bridge_shared import RawFrameState
from sokurl_platform.render_state import RenderEntity, RenderSnapshot
from soku_rl.env.encoding import AGENTS, decode_action
from soku_rl.env.hisouten_env import EpisodeConfig
from soku_rl.env.match import LEGACY_MATCH
from soku_rl.env.observation.animation import observe_animation
from soku_rl.env.observation.visibility import VisibilityConfig
from soku_rl.env.observation_history import ObservationHistory
from soku_rl.env.wrappers.learning import LearningConfig, LearningEpisode, LearningInterface


def test_checkpoint_cpu_adapter_matches_existing_learning_history_and_reset():
    visibility=VisibilityConfig(8,.5,.02,.1,48,96,16,.25)
    episode=EpisodeConfig(7200,8,1,0,'animation',visibility,LEGACY_MATCH)
    adapter=CheckpointControls.__new__(CheckpointControls)
    adapter.episode=episode
    adapter.interface=LearningInterface(episode,LearningConfig('combat',False,8,0.))
    raw=RawFrameState()
    for p in (raw.p1,raw.p2):
        p.hp=10000;p.spirit=p.maxSpirit=1000
    raw.p1.characterId=1
    render=RenderSnapshot(0,400,1,0,
        (RenderEntity(100,0,1,1,1),RenderEntity(500,0,1,-1,1)),((),()),False)
    for _ in range(2):
        adapter.reset()
        reference=ObservationHistory(episode,AGENTS)
        learning=LearningEpisode(adapter.interface)
        for frame in range(15):
            raw.frameId=frame
            raw.p1.animationFrame=frame
            observations=observe_animation(raw,render,visibility)
            infos={a:{'frame':frame} for a in AGENTS}
            if frame==0:
                reference.reset(frame,observations)
                expected,_=learning.reset(reference.observations(),infos)
            else:
                reference.append(frame,observations)
                flags={a:False for a in AGENTS}
                values={a:0. for a in AGENTS}
                expected=learning.step(commands,(reference.observations(),values,flags,flags,infos))[0]
            np.testing.assert_array_equal(adapter.observation(raw,render),expected['player_0'])
            action=frame%adapter.interface.action_space.n
            commands={a:adapter.interface.command(action) for a in AGENTS}
            assert adapter.command(action)==decode_action(commands['player_0']).inputs
