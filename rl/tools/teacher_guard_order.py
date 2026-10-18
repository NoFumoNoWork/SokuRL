"""Isolated AST proof/generation of a narrow pure Boolean guard permutation.

Only source conditions containing get_skill_lv are considered. Its original
native getter is read-only and numeric; it remains unavailable to the teacher.
Known scalar conjuncts can disprove a branch before this unavailable read.
RNG, unknown calls, indexing, arithmetic traps and local bindings are barriers.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import types


def generate(package, api, module):
    from luaparser import ast, astnodes as n
    sha = lambda data: hashlib.sha256(data).hexdigest()
    sources = {name: (package / name).read_bytes() for name in module.SOURCES}
    if {k: sha(v) for k, v in sources.items()} != module.SOURCES:
        raise ValueError('Unreviewed community source')
    guards = []

    def numeric(node):
        if isinstance(node, n.Number): return True, False
        if isinstance(node, n.Name):
            return node.id in module.NUMERIC | module.UNAVAILABLE_NUMERIC, node.id in module.UNAVAILABLE_NUMERIC
        if isinstance(node, n.UMinusOp): return numeric(node.operand)
        if isinstance(node, (n.AddOp, n.SubOp, n.MultOp)):
            left, right = numeric(node.left), numeric(node.right)
            return left[0] and right[0], left[1] or right[1]
        if isinstance(node, n.FloatDivOp) and isinstance(node.right, n.Number) and node.right.n != 0:
            return numeric(node.left)
        if (isinstance(node, n.Call) and isinstance(node.func, n.Name) and
                node.func.id == 'get_skill_lv' and len(node.args) == 2 and
                all(isinstance(a, n.Number) for a in node.args) and
                node.args[0].n in (0, 1) and 0 <= node.args[1].n <= 14):
            return True, True
        return False, False

    def boolean(node):
        if isinstance(node, (n.LessThanOp, n.LessOrEqThanOp, n.GreaterThanOp,
                             n.GreaterOrEqThanOp, n.EqToOp, n.NotEqToOp)):
            left, right = numeric(node.left), numeric(node.right)
            return left[0] and right[0], left[1] or right[1]
        if isinstance(node, n.Name): return node.id in module.BOOLEAN, False
        if isinstance(node, (n.AndLoOp, n.OrLoOp)):
            left, right = boolean(node.left), boolean(node.right)
            return left[0] and right[0], left[1] or right[1]
        return False, False

    def flatten(node):
        if isinstance(node, n.AndLoOp): return flatten(node.left) + flatten(node.right)
        return [node]

    for name in ('01_marisa_main.ai', '00_reimu_main.ai'):
        raw = sources[name]
        text = raw.decode('cp932', 'surrogateescape')
        offsets = [0]
        for char in text:
            offsets.append(offsets[-1] + len(char.encode('cp932', 'surrogateescape')))
        tree = ast.parse(text)
        main = next(x for x in tree.body.body if isinstance(x, n.Function) and
                    isinstance(x.name, n.Name) and x.name.id == 'main')
        # No rewriting under a local or nested-function scope: declarations can
        # shadow the reviewed scalar names or getter. Fail closed for that main.
        bound = set()
        nested = False
        for item in ast.walk(main):
            if isinstance(item, (n.LocalAssign, n.Forin)):
                bound.update(x.id for x in item.targets if isinstance(x, n.Name))
            if isinstance(item, n.Fornum): bound.add(item.target.id)
            if isinstance(item, n.LocalFunction) and isinstance(item.name, n.Name):
                bound.add(item.name.id)
            if isinstance(item, n.Function) and item is not main: nested = True
        if nested or bound & (module.NUMERIC | module.UNAVAILABLE_NUMERIC | module.BOOLEAN | {'get_skill_lv'}):
            continue
        for node in ast.walk(main):
            if not isinstance(node, (n.If, n.ElseIf)): continue
            clauses = flatten(node.test)
            traits = [boolean(x) for x in clauses]
            if not any(isinstance(x,n.Call) and isinstance(x.func,n.Name) and
                       x.func.id=='get_skill_lv' for x in ast.walk(node.test)): continue
            if not any(pure and deferred for pure, deferred in traits): continue
            order = list(range(len(clauses)))
            first = 0
            while first < len(clauses):
                if not traits[first][0]:
                    first += 1
                    continue
                last = first
                while last < len(clauses) and traits[last][0]: last += 1
                # Stable partition inside each proven-pure run. Never move any
                # expression across RNG/unknown calls or unproved predicates.
                order[first:last] = sorted(range(first, last), key=lambda i: traits[i][1])
                first = last
            if order == list(range(len(clauses))): continue
            span = lambda x: (offsets[x.first_token.start], offsets[x.last_token.stop + 1])
            start, end = span(node.test)
            fragments = []
            for clause, trait in zip(clauses, traits):
                a, b = span(clause)
                fragments.append(dict(start=a, end=b, sha256=sha(raw[a:b]),
                    kind='DEFERRED_NUMERIC' if trait[1] and trait[0] else
                         'PUBLIC_SCALAR' if trait[0] else 'BARRIER'))
            guards.append(dict(source=name, line=node.first_token.line,
                start=start, end=end, sha256=sha(raw[start:end]),
                clauses=fragments, order=order, policy=module.POLICY))
    return dict(schema='teacher_pure_guard_order_v1', sources=module.SOURCES,
        api_sha256=sha(api.read_bytes()), numeric=sorted(module.NUMERIC),
        boolean=sorted(module.BOOLEAN), policy=module.POLICY, guards=guards,
        unavailable_numeric=sorted(module.UNAVAILABLE_NUMERIC),
        generator_sha256=sha(Path(__file__).read_bytes().replace(b'\r\n', b'\n')),
        evidence=['th123_ai main.cpp:925-932 read-only numeric GetSkillLv',
            'scriptLua.cpp:212-222 numeric argument conversion/return',
            'def_clc.lua:915-923 gravity estimates are scalar velocity differences; values remain unavailable',
            'document.txt:139-143 default0/unavailable-1/learned1..4; NOT current task values'],
        limits='Boolean condition equivalence under declared numeric types; NOT hidden-data access or complete policy support')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--site', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    site, output = Path(args.site).resolve(strict=True), Path(args.output).resolve()
    if (not site.is_relative_to(root / 'rl/.tools') or
            not output.is_relative_to(root / 'rl/runs') or output.exists()):
        raise ValueError('Use isolated repository tools and a NEW run output')
    sys.path.insert(0, str(site))
    # Load constants without importing Lupa, PPO or the training environment.
    module_path = root / 'rl/src/soku_rl/policy/god/guard_order.py'
    package = types.ModuleType('teacher_guard_contract')
    package.__path__ = [str(module_path.parent)]
    sys.modules[package.__name__] = package
    spec = importlib.util.spec_from_file_location(package.__name__ + '.guard_order', module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    result = generate(root / 'third_party/th123_ai/package/th123ai/script',
        root / 'third_party/th123_ai/source/th123_ai/api.ai', module)
    output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(dict(guards=len(result['guards']), output=str(output))))


if __name__ == '__main__': main()
