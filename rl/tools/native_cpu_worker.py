"""Scoped launch adapter; all process/reset/step/read implementations stay shared."""
from unittest.mock import patch
import hashlib
import json
import os
from pathlib import Path

from sokurl_platform import sokurl
import rollout_worker
from game_runtime.batch import SokuGameBatch


def cpu_launcher(original):
    def launch(*args, **kwargs):
        if (kwargs.get('characters') != (1, 0) or not kwargs.get('cards_disabled')
                or kwargs.get('original_replay_recording')):
            raise ValueError('native CPU worker supports Marisa-left/Reimu-right and no cards/recording')
        return original(*args, **kwargs, native_cpu=True)
    return launch


class NativeCPUTrainingBatch(SokuGameBatch):
    def _launch_slots(self, seeds):
        if self.fresh_reset or self.recording_enabled or self.fixed_weather != 'clear':
            raise ValueError('CPU fast reset/recording not verified; use Clear scene reset')
        audit = json.loads(Path(os.environ['SOKURL_NATIVE_CPU_CONFIG_AUDIT']).read_text('utf-8'))
        digest = hashlib.sha256((sokurl.GAME_DIR / 'config123.dat').read_bytes()).hexdigest()
        if (audit['status'] != 'STATIC_VERIFIED_CONFIG_SERIALIZATION'
                or audit['difficulty'] != 3 or audit['during_sha256'] != digest):
            raise ValueError('original Lunatic configuration audit/hash mismatch')
        # Scope the existing launcher option only to this dedicated worker call.
        # No native field extraction or alternate frame scheduler is introduced.
        with patch.object(sokurl, '_launch_vs_group_from_title',
                          cpu_launcher(sokurl._launch_vs_group_from_title)):
            return super()._launch_slots(seeds)


if __name__ == '__main__':
    with patch.object(rollout_worker, 'SokuGameBatch', NativeCPUTrainingBatch):
        rollout_worker.main()
