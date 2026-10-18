"""Reconcile an existing byte-bound Lua AST audit with continuous-plan ownership.

No new parser, native read, original-source modification, or gameplay measurement.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path

from soku_rl.policy.god.source_adapter import SOURCES, digest


def build(audit, package):
    if audit['sources'] != SOURCES or any(
            digest((package/name).read_bytes()) != sha for name, sha in SOURCES.items()):
        raise ValueError('AST/source identity mismatch')
    groups = defaultdict(list)
    for row in audit['assignments']:
        for name in row['writes']:
            groups[name].append(dict(source=row['source'], line=row['line'],
                function=row['enclosing_function'], reads=row['reads'], calls=row['calls']))
    rows = []
    for name, sites in sorted(groups.items()):
        funcs = {x['function'] for x in sites}
        role = ('SHARED_TACTICAL_AND_OBSERVATION' if {'main', 'calculation'} <= funcs else
                'OBSERVED_HISTORY_OR_SOURCE_INITIALIZATION' if 'calculation' in funcs else
                'MAIN_PLAN_OR_INITIALIZATION' if 'main' in funcs else 'HELPER_OR_INITIALIZATION')
        policy = ('calculation writes persist; main writes plan-local; no inferred card execution'
                  if name == 'flag_kishitu' else
                  'main writes plan-local; only Reimu lines53-63 observation rules persist'
                  if name in ('ex_combo', 'crush_combo') else
                  'source initialization/per-frame observation persist; decision writes disposable')
        rows.append(dict(name=name, role=role, policy=policy, sites=sites))
    return dict(schema='continuous_teacher_state_audit_v1', sources=SOURCES,
        evidence='VERIFIED_PINNED_AST_WRITE_SITES; NOT_NATIVE_OR_BRANCH_COVERAGE',
        variables=rows,
        main_writes=sorted(n for n, s in groups.items() if any(x['function']=='main' for x in s)),
        shared_main_calculation=sorted(n for n, s in groups.items()
            if {'main', 'calculation'} <= {x['function'] for x in s}),
        coroutine=dict(calculation='original body once per real supplied observation',
            main='fresh reviewed loop entry on safe recovery; no rejected continuation',
            wait='original local counter; advances only after trustworthy receipt',
            input='ScriptAPI local proposals, not game writes',
            rng='persistent actor-local RNG, draws are teacher randomness not environment facts',
            delay='v1 requires zero key/data delay; no delayed rejected input queue retained'),
        adaptations=['observation-before-decision differs from original scheduler',
            'main-only tactical writes deliberately not retained across replans',
            'observed-table mutation in decision masks current plan',
            'unsupported/private API path masks query; partial calculation error requires Reset'],
        native_status='BLOCKED_EXISTING_PAIRED_CAPTURE_TIMING_AND_RECEIPT_ADAPTER_NOT_APPROVED')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit', required=True)
    parser.add_argument('--package', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    root = Path(__file__).resolve().parents[2]
    if not output.is_relative_to(root/'rl/runs') or output.exists():
        raise ValueError('new repository-local run output required')
    result = build(json.loads(Path(args.audit).read_text('utf-8')), Path(args.package))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(dict(output=str(output), variables=len(result['variables']),
        main_writes=result['main_writes'], shared=result['shared_main_calculation'])))


if __name__ == '__main__':
    main()
