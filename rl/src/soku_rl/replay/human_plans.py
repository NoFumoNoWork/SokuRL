"""Hash-bound export selection only;no native reads or label semantics."""
import json
from pathlib import Path

from soku_rl.replay.human_dataset import sha256


def export_plan(audit, partition, explicit=None):
    if partition not in ('train','validation'):
        raise ValueError('sealed test export is forbidden')
    audit = Path(audit)
    if json.loads((audit/'audit.json').read_text())['status']!='PASS':
        raise ValueError('source audit must pass')
    if explicit:
        path = Path(explicit).resolve(strict=True)
        plan = json.loads(path.read_text())
        splits = audit/'splits.json'
        if (plan.get('schema')!='human_replay_export_plan_v1' or plan['partition']!=partition
                or plan['split_sha256']!=sha256(splits)):
            raise ValueError('export plan is not bound to the audited partition')
        pool = {r['id']:r for r in json.loads(splits.read_text())[partition]}
        rows = plan['rows']
        if (not 1<=len(rows)<=200 or len({r['id'] for r in rows})!=len(rows)
                or any(r!=pool.get(r['id']) for r in rows)):
            raise ValueError('plan contains duplicate,changed or non-partition rows')
        return path,plan
    if partition=='train':
        path = audit/'pilot.json'
        plan = json.loads(path.read_text())
    else:
        from audit_human_replays import choose_pilot
        path = audit/'splits.json'
        plan = dict(rows=choose_pilot(json.loads(path.read_text())['validation']))
    if len(plan['rows'])!=20: raise ValueError('default pilot must contain20rows')
    return path,plan
