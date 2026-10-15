"""Rebuild derived input attribution without modifying native capture evidence."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from soku_rl.env.input_lineage import InputLineage
import soku_rl.env.input_lineage as input_lineage


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--capture',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    mode=parser.add_mutually_exclusive_group()
    mode.add_argument('--diagnostic-action-mapping',action='store_true')
    mode.add_argument('--accepted-action-roots',action='store_true')
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError('preserve existing evidence')
    raw=args.capture.read_bytes()
    data=json.loads(raw)
    if data.get('status')!='PASS':raise ValueError('native capture did not pass')
    states=data.get('states',[])
    characters=tuple(states[0]['characters']) if states and 'characters' in states[0] else None
    if args.diagnostic_action_mapping and characters is None:
        raise ValueError('diagnostic action mapping requires character metadata')
    if characters is not None and any(tuple(state['characters'])!=characters for state in states):
        raise ValueError('capture changes characters; split into epochs first')
    derived=InputLineage(diagnostic_action_mapping=args.diagnostic_action_mapping,
                         accepted_action_roots=args.accepted_action_roots).observe(data['events'],characters)
    allowed={'independent_input_id','input_candidates','independent_input_segment',
             'independent_attack_index','input_lineage_evidence','new_independent_attack',
             'native_combo_id','independent_action_instance','attack_start_evidence',
             'attack_root_evidence','combo_evidence','input_acceptance_evidence'}
    for original,event in zip(data['events'],derived,strict=True):
        if {k:v for k,v in original.items() if k not in allowed}!={k:v for k,v in event.items() if k not in allowed}:
            raise RuntimeError('attribution modified native evidence')
    hits=[e for e in derived if e['kind']=='HIT_RESOLUTION' and e['hp_after']<e['hp_before']]
    data.update(events=derived,input_lineage_provenance=dict(
        source_sha256=hashlib.sha256(raw).hexdigest(),
        implementation_sha256=hashlib.sha256(Path(input_lineage.__file__).read_bytes()).hexdigest(),
        evidence='DERIVED_NATIVE_INPUT_SCOPE_CONTACT_JOIN',
        scope='OBSERVED_WINDOW_ONLY',
        diagnostic_action_mapping=args.diagnostic_action_mapping,
        accepted_action_roots=args.accepted_action_roots,
        coverage=dict(Counter(str(e['independent_input_segment']) for e in hits))))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(data,indent=2),encoding='utf-8')
    print(json.dumps(data['input_lineage_provenance'],indent=2))


if __name__=='__main__':main()
