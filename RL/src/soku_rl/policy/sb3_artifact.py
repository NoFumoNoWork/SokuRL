"""Load trusted SB3 artifacts by their saved policy type, independent of the trainer."""
import io
import json
from zipfile import ZipFile

from sb3_contrib import RecurrentPPO
from sb3_contrib.common.recurrent.policies import RecurrentActorCriticPolicy
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.save_util import json_to_data
from stable_baselines3.dqn.policies import DQNPolicy


def load_sb3_artifact(checkpoint_bytes, device):
    # NFSP's average and response can have different types under one training
    # contract. Inspect existing SB3 metadata, including historical artifacts.
    # Decode only the policy class; leave tensors to the selected loader.
    with ZipFile(io.BytesIO(checkpoint_bytes)) as archive:
        metadata = json.loads(archive.read("data"))
    policy = json_to_data(json.dumps({"policy_class": metadata["policy_class"]}))["policy_class"]
    if not isinstance(policy, type):
        raise ValueError("SB3 artifact must declare a policy class")
    if issubclass(policy, DQNPolicy):
        from soku_rl.rl.dqn import DoubleDQN
        algorithm, kind = DoubleDQN, "sb3_dqn"
    elif issubclass(policy, RecurrentActorCriticPolicy):
        algorithm, kind = RecurrentPPO, "sb3_recurrent"
    elif issubclass(policy, ActorCriticPolicy):
        algorithm, kind = PPO, "sb3"
    else:
        raise ValueError(f"unsupported SB3 policy class: {policy}")
    return algorithm.load(io.BytesIO(checkpoint_bytes), device=device), kind
