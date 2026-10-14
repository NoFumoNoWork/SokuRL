"""Check saved public-token PPO inference, resume and optional actor costs."""
import argparse
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import torch
from stable_baselines3 import PPO
from soku_rl.rl.ppo import parameter_hash


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    checks = {}
    for name, path in [('tokens',args.candidate),('mlp',args.baseline)]:
        model = PPO.load(path, device='cuda')
        obs = np.zeros((4,*model.observation_space.shape),dtype=np.float32)
        saved_hash = parameter_hash(model.policy)
        restored = PPO.load(path, device='cpu')
        assert parameter_hash(restored.policy) == saved_hash
        cpu_action, _ = restored.predict(obs, deterministic=True)
        gpu_action, _ = model.predict(obs, deterministic=True)
        assert np.array_equal(cpu_action,gpu_action)
        batch = torch.as_tensor(obs, device='cuda')
        for _ in range(5):
            with torch.no_grad():
                model.policy(batch,deterministic=True)
        torch.cuda.synchronize()
        started = time.perf_counter()
        for _ in range(25):
            with torch.no_grad():
                model.policy(batch,deterministic=True)
        torch.cuda.synchronize()
        duration = time.perf_counter()-started
        before = saved_hash
        actions = torch.as_tensor(gpu_action,device='cuda')
        values,logp,entropy = model.policy.evaluate_actions(batch,actions)
        loss = values.square().mean() - logp.mean() - .01*entropy.mean()
        model.policy.optimizer.zero_grad()
        loss.backward()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.policy.parameters())
        model.policy.optimizer.step()
        assert parameter_hash(model.policy) != before
        checks[name] = {'status':'PASS','checkpoint':str(path.resolve()),
            'parameter_count':sum(p.numel() for p in model.policy.parameters()),
            'hash':saved_hash,'forward_ms_batch4':duration/25*1000,
            'loaded_optimizer_update':'PASS','cpu_cuda_zero_observation_actions':'PASS'}
        del restored,model
    root = Path(__file__).resolve().parents[2]
    git = lambda *parts:subprocess.check_output(['git','-C',str(root),*parts],text=True).strip()
    report = {'status':'PASS','scope':'synthetic zero-observation actor cost,not environment throughput',
        'git':{'head':git('rev-parse','HEAD'),'branch':git('branch','--show-current'),
               'status':git('status','--porcelain')},'torch':torch.__version__,
        'device':torch.cuda.get_device_name(),'checks':checks}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(checks,indent=2))

if __name__ == '__main__':
    main()
