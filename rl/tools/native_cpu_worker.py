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
    def reset_slots(self, seeds):
        if os.environ.get('SOKURL_CPU_RESET_PROCESS_RESTART')=='1':
            self._close_slots(set(seeds)&set(self.processes))
        return super().reset_slots(seeds)

    def _create_reader(self, pid):
        reader = super()._create_reader(pid)
        reader.expected_battle_mode = 2
        return reader

    def _launch_slots(self, seeds):
        if (self.recording_enabled or self.fixed_weather != 'clear' or
                (self.fresh_reset and os.environ.get('SOKURL_NATIVE_CPU_FRESH_RESET')!='1')):
            raise ValueError('CPU reset requires Clear/no recording and explicit verified fresh-reset opt-in')
        if self.fresh_reset and os.environ.get('SOKURL_CPU_RESET_PROCESS_RESTART')=='1':
            raise ValueError('fresh reset and process restart are mutually exclusive')
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
