"""Explicit ownership contract for the existing original CPU launch path."""
from pathlib import Path

from soku_rl.policy.base import RulePolicy
from soku_rl.env.encoding import encode_action


def validate_configuration(config):
    opponents = config['algorithm'].get('opponents', [])
    native = any(entry['policy'].get('kind') == 'native_cpu' for entry in opponents)
    worker = Path(config['runtime']['command'][-1]).name
    if not native and worker != 'native_cpu_worker.py':
        return
    if (not native or len(opponents) != 1 or opponents[0]['probability'] != 1.0
            or opponents[0]['policy'] != {'kind': 'native_cpu', 'difficulty': 3}
            or worker != 'native_cpu_worker.py'
            or config['algorithm']['name'] != 'br' or config['algorithm']['player'] != 0
            or config['runtime']['fresh_reset'] or config['runtime']['record_replays']
            or not config['runtime']['cards_disabled'] or config['runtime']['fixed_weather'] != 'clear'
            or config['algorithm']['matchups']['mode'] != 'sampled'
            or config['algorithm']['matchups']['learner']['character'] != 1
            or opponents[0]['setup']['character'] != 0):
        raise ValueError('native CPU requires explicit Lunatic worker, Marisa-left/Reimu-right, scene reset and Clear/no cards')


class NativeCPUPlaceholder(RulePolicy):
    """Neutral transport placeholder; the Bridge ignores CPU-side external keys.

    This does not implement the CPU policy. Only the dedicated worker may be used.
    """
    def __init__(self, name, interface):
        self.name = name
        self.neutral = interface.action(encode_action((0,) * 8))
        self.fingerprint = 'th123-1.10a-original-vscom-right-config3-v1'

    def spawn(self, seed):
        return self

    def act(self, observation):
        return self.neutral
