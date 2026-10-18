"""Source-bound pure Boolean guard ordering, never missing-field substitution."""
import json
from pathlib import Path

from .source_adapter import SOURCES, digest


# Exact delayed public scalars. No tables, hidden timers, or dynamic properties.
NUMERIC = frozenset({'my_act', 'enemy_act', 'my_act_block', 'enemy_act_block',
    'my_frame', 'enemy_frame', 'my_dir', 'enemy_dir', 'my_char', 'enemy_char',
    'x', 'y', 'ex', 'ey', 'dis', 'dis_x', 'dis_y', 'dis2',
    'my_allframe', 'enemy_allframe', 'weather', 'weather2', 'weather_time'})
BOOLEAN = frozenset({'is_dir_front'})
# Proven scalar types in the pinned source, NOT provided values. The gravity
# estimates are original def_clc.lua:915-923 scalar velocity differences.
UNAVAILABLE_NUMERIC = frozenset({'my_xspeed', 'my_yspeed', 'enemy_xspeed',
                                'enemy_yspeed', 'e_gs', 'm_gs'})
POLICY = 'PURE_BOOLEAN_REQUIRED_CONJUNCTS_ONLY_NO_EFFECT_BARRIER_CROSSING'


def replacements(data, source, registry):
    """Return validated byte edits for the exact original source, not AST output."""
    if (registry.get('schema') != 'teacher_pure_guard_order_v1' or
            registry.get('sources') != SOURCES or
            registry.get('numeric') != sorted(NUMERIC) or
            registry.get('boolean') != sorted(BOOLEAN) or
            registry.get('unavailable_numeric') != sorted(UNAVAILABLE_NUMERIC) or
            registry.get('policy') != POLICY or digest(data) != SOURCES[source]):
        raise ValueError('Unreviewed pure-guard source or type contract')
    edits = []
    previous = len(data)
    rows = sorted((r for r in registry['guards'] if r['source'] == source),
                  key=lambda r: r['start'], reverse=True)
    for row in rows:
        start, end = row['start'], row['end']
        clauses, order = row['clauses'], row['order']
        if (type(start) is not int or type(end) is not int or
                not 0 <= start < end <= previous or row['policy'] != POLICY or
                digest(data[start:end]) != row['sha256'] or
                sorted(order) != list(range(len(clauses))) or
                any(type(i) is not int for i in order)):
            raise ValueError('Invalid pure-guard condition binding')
        parts = []
        cursor = start
        for clause in clauses:
            first, last = clause['start'], clause['end']
            if (not cursor <= first < last <= end or
                    digest(data[first:last]) != clause['sha256']):
                raise ValueError('Invalid pure-guard clause binding')
            parts.append(data[first:last])
            cursor = last
        expected = list(range(len(clauses)))
        first = 0
        while first < len(clauses):
            if clauses[first]['kind'] == 'BARRIER':
                first += 1
                continue
            last = first
            while last < len(clauses) and clauses[last]['kind'] != 'BARRIER':
                if clauses[last]['kind'] not in ('PUBLIC_SCALAR', 'DEFERRED_NUMERIC'):
                    raise ValueError('Invalid pure-guard clause type')
                last += 1
            expected[first:last] = sorted(range(first, last),
                key=lambda i: clauses[i]['kind'] == 'DEFERRED_NUMERIC')
            first = last
        if order != expected:
            raise ValueError('Invalid pure-guard effect barrier/order')
        code = b' and '.join(b'(' + parts[i] + b')' for i in order)
        # Preserve physical source-line attribution for all later code.
        if code.count(b'\n') > data[start:end].count(b'\n'):
            raise ValueError('Pure-guard line preservation failed')
        code += b'\n' * (data[start:end].count(b'\n') - code.count(b'\n'))
        edits.append((start, end, code))
        previous = start
    return edits


def load_registry():
    data = Path(__file__).with_name('pure_guard_order.json').read_bytes()
    return json.loads(data), digest(data)
