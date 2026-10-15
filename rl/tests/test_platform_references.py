"""Old model/transport names resolve to Platform types without duplicate engines."""
import pickle


def test_legacy_pickle_type_paths_still_resolve():
    from sokurl_platform.pixels import RGBFrame
    from sokurl_platform.replay_format import Replay
    from sokurl_platform.match import MatchConfig
    for old, expected in (('soku_rl.env.observation.pixels\nRGBFrame', RGBFrame),
                          ('soku_rl.replay.format\nReplay', Replay),
                          ('soku_rl.env.match\nMatchConfig', MatchConfig)):
        assert pickle.loads(('c' + old + '\n.').encode('ascii')) is expected


def test_readers_and_frame_execution_have_unique_owners():
    from game_runtime.frames import drain_frames_into
    from game_runtime.privileged import ProcessMemory
    from game_runtime.images import ImageClient
    from sokurl_platform.frames import drain_frames_into as frames
    from sokurl_platform.memory import ProcessMemory as memory
    from sokurl_platform.images import ImageClient as image
    assert drain_frames_into is frames
    assert ProcessMemory is memory
    assert ImageClient is image
