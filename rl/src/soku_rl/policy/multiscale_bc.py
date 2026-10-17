"""Per-episode BC context. Re-observe and execute only head0 every native frame."""
from collections import deque
from pathlib import Path

import numpy as np
import torch

from soku_rl.policy.base import RLPolicy
from soku_rl.policy.contract import read_training_contract
from soku_rl.replay.human_dataset import sha256
from soku_rl.replay.human_windows import window_inputs
from soku_rl.rl.multiscale_bc import load_actor, tensor_batch


class MultiscaleBCPolicy(RLPolicy):
    def __init__(self,name,path,interface,device):
        path = Path(path).resolve(strict=True)
        read_training_contract(path.parent/'config.yaml',interface)
        if interface.episode.decision_frames != 1 or interface.episode.latency_frames != 0:
            raise ValueError('BC history requires 1F decisions and zero latency')
        self.name, self.device = name, device
        self.model,self.actor,self.metadata = load_actor(path,device)
        if self.model.observation_space != interface.observation_space:
            raise ValueError('BC actor observation contract mismatch')
        self.fingerprint = sha256(path)
        self.horizon = interface.episode.max_frames

    def spawn(self, seed):
        return MultiscaleEpisode(self,np.random.default_rng(seed))


class MultiscaleEpisode:
    def __init__(self,policy,rng):
        self.policy,self.rng = policy,rng
        self.observations = deque(maxlen=61)
        self.inputs = deque(maxlen=60)
        self.last_frame = None

    @torch.inference_mode()
    def act(self,observation):
        value = np.asarray(observation,np.float32)
        if value.shape != (1544,) or not np.isfinite(value).all():
            raise ValueError('BC actor expects original1544 observation')
        # Frame is public normalized time, not an extra native read.
        frame = int(round(float(value[1479])*self.policy.horizon))
        if self.last_frame is not None and frame != self.last_frame+1:
            raise ValueError('BC episode history crossed a frame gap or reset;spawn a new actor')
        if self.last_frame is not None:
            # The existing wrapper exposes only OUR last submitted logical command.
            self.inputs.append(value[-8:].copy())
        self.observations.append(value.copy())
        self.last_frame = frame
        known = len(self.observations)==61 and len(self.inputs)==60
        states = np.stack(([self.observations[0]]*(61-len(self.observations)))+list(self.observations))
        commands = np.stack(([np.zeros(8,np.float32)]*(60-len(self.inputs)))+list(self.inputs))
        data = window_inputs(states,commands,np.array([60]))
        batch = tensor_batch(data,self.policy.device)
        batch['context_known'] = torch.tensor([known],device=self.policy.device)
        probabilities = self.policy.actor.logits(batch)[:,0].softmax(-1)[0].cpu().numpy().astype(np.float64)
        probabilities /= probabilities.sum()
        return int(self.rng.choice(90,p=probabilities))
