"""Isolated static audit. Run python -I -S with explicit --site, never in PPO.

AST dependency availability is not an executable-branch or native-legality proof.
"""
import argparse
from collections import Counter
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import sys


def audit(package, api, available):
    from luaparser import ast, astnodes as n
    names = ('01_marisa_main.ai','00_reimu_main.ai','def_clc.lua','function_command.txt')
    sources = {name:(package/name).read_bytes() for name in names}
    texts = {name:data.decode('cp932','surrogateescape') for name,data in sources.items()}
    offsets = {}
    for name,text in texts.items():
        positions=[0]
        for char in text:positions.append(positions[-1]+len(char.encode('cp932','surrogateescape')))
        if positions[-1] != len(sources[name]):raise ValueError('source byte mapping failed')
        offsets[name]=positions
    # Static text parsing only. CP932 trail bytes can be ASCII backslashes; a
    # Latin-1 decode misidentifies them as escapes. Never re-execute AST output.
    trees = {name:ast.parse(text) for name,text in texts.items()}
    functions = {}
    assigned = set()

    def children(node):
        for key,value in vars(node).items():
            if key.startswith('_') or key in ('comments','wrapped'):
                continue
            if isinstance(value,n.Node):
                yield value
            elif isinstance(value,list):
                yield from (x for x in value if isinstance(x,n.Node))

    def scan(node, bound=frozenset()):
        """Conservative free reads/calls/writes; comments and labels are not reads."""
        reads,calls,writes = set(),set(),set()
        if isinstance(node,n.Name):
            return ({node.id} if node.id not in bound else set()),calls,writes
        if isinstance(node,(n.Function,n.LocalFunction)):
            extra = {x.id for x in node.args if isinstance(x,n.Name)}
            return scan(node.body,bound|extra)
        if isinstance(node,n.Assign):
            for target in node.targets:
                base = target
                while isinstance(base,n.Index):
                    base = base.value
                if isinstance(base,n.Name) and base.id not in bound:
                    writes.add(base.id)
                if isinstance(target,n.Index):
                    r,c,w = scan(target,bound); reads|=r;calls|=c;writes|=w
            items = node.values
        elif isinstance(node,n.Index):
            # x.foo reads x, not a second global named foo.
            items = [node.value]
            if getattr(node,'notation',None) != n.IndexNotation.DOT:
                items.append(node.idx)
        else:
            items = list(children(node))
        if isinstance(node,n.Call) and isinstance(node.func,n.Name):
            calls.add(node.func.id)
        elif isinstance(node,n.Call) and isinstance(node.func,n.Index) and isinstance(node.func.value,n.Name) and isinstance(node.func.idx,n.Name):
            calls.add(node.func.value.id+'.'+node.func.idx.id)
        if isinstance(node,n.Fornum):
            bound |= {node.target.id}
        if isinstance(node,n.Forin):
            bound |= {x.id for x in node.targets}
        if isinstance(node,n.LocalAssign):
            items = node.values
        for item in items:
            r,c,w = scan(item,bound);reads|=r;calls|=c;writes|=w
        return reads,calls,writes

    for name,tree in trees.items():
        for node in ast.walk(tree):
            if isinstance(node,n.Function) and isinstance(node.name,n.Name):
                functions[node.name.id] = (name,node)
            if isinstance(node,n.Assign):
                for target in node.targets:
                    base = target
                    while isinstance(base,n.Index):base=base.value
                    if isinstance(base,n.Name):assigned.add(base.id)
    constants = set()
    for node in ast.walk(ast.parse(api.read_bytes().decode('cp932','surrogateescape'))):
        if isinstance(node,n.Assign):
            for target,value in zip(node.targets,node.values):
                if isinstance(target,n.Name) and isinstance(value,(n.Number,n.String)):
                    constants.add(target.id)
    intrinsic = {'math','string','table','print','tonumber','tostring','pairs','ipairs','type','unpack'} | constants

    def expand(node, bound=frozenset()):
        reads,calls,writes = scan(node,bound)
        pending = list(calls)
        visited = set()
        while pending:
            name = pending.pop()
            if name in visited or name not in functions:continue
            visited.add(name)
            r,c,w = scan(functions[name][1])
            reads|=r;calls|=c;writes|=w;pending.extend(c-visited)
        return reads-intrinsic-set(functions),calls,writes

    def reasons(reads):
        missing = []
        for field in sorted(reads-set(available)):
            kind = 'SCRIPT_MEMORY_REQUIRES_PHASE2' if field in assigned else 'UNAVAILABLE_FIELD_OR_API'
            if field in ('enemy_obj_n','my_obj_n','get_obj_data','get_obj_attackarea','get_obj_hitarea','obj_dis'):
                kind = 'NATIVE_OBJECT_OR_BOX_SEMANTICS_NOT_PUBLIC_VIEW'
            if field in ('get_key_stat2','get_key_map2','enemy_keys'):
                kind = 'HIDDEN_INPUT_NOT_ALLOWED'
            if field in ('command','propose_intent','key_on','key_off','key_reset','key_event','key_event2','wait','yield'):
                kind = 'PHASE3_INPUT_OPERATION_NOT_AN_ATOMIC_LABEL'
            missing.append(dict(name=field,reason=kind))
        return missing

    branches = []
    assignments = []
    def visit(node, source, path=(), scope=None, bound=frozenset(), loops=()):
        if isinstance(node,(n.Function,n.LocalFunction)):
            args={x.id for x in node.args if isinstance(x,n.Name)}
            name=node.name.id if isinstance(node.name,n.Name) else 'UNKNOWN'
            visit(node.body,source,(),name,bound|args)
            return
        if isinstance(node,n.Assign):
            reads,calls,writes=scan(node,bound)
            if writes:
                assignments.append(dict(source=source,line=node.first_token.line,
                    enclosing_function=scope,writes=sorted(writes),reads=sorted(reads),
                    calls=sorted(calls),path=list(path),
                    note='direct statement dependencies; enclosing guards/loops remain necessary'))
        if isinstance(node,(n.While,n.Fornum,n.Forin,n.Repeat)):
            if isinstance(node,(n.While,n.Repeat)):
                expressions=[node.test]
            elif isinstance(node,n.Fornum):
                expressions=[x for x in (node.start,node.stop,node.step) if isinstance(x,n.Node)]
            else:
                expressions=node.iter
            loop_reads=set()
            for expression in expressions:loop_reads.update(expand(expression,bound)[0])
            context=dict(line=node.first_token.line,kind=type(node).__name__,
                reads=sorted(loop_reads),missing=reasons(loop_reads),
                note='entry dependencies only; iteration execution/order not verified')
            if isinstance(node,n.Fornum):bound|={node.target.id}
            if isinstance(node,n.Forin):bound|={x.id for x in node.targets}
            visit(node.body,source,path,scope,bound,loops+(context,))
            return
        if isinstance(node,n.Fornum):bound|={node.target.id}
        if isinstance(node,n.Forin):bound|={x.id for x in node.targets}
        if isinstance(node,(n.If,n.ElseIf)):
            line = node.first_token.line
            reads,calls,writes = expand(node.test,bound)
            br,bc,bw = expand(node.body,bound)
            missing = reasons(reads)
            branches.append(dict(id=f'{source}:{line}',source=source,line=line,enclosing_function=scope,
                kind=type(node).__name__,
                enclosing_bindings=sorted(bound),loop_context=list(loops),
                condition_start=offsets[source][node.test.first_token.start],
                condition_end=offsets[source][node.test.last_token.stop+1],
                condition=texts[source][node.test.first_token.start:node.test.last_token.stop+1],
                path=list(path),guard_reads=sorted(reads),guard_calls=sorted(calls),
                missing_guard=missing,body_missing=reasons(br),body_calls=sorted(bc),body_writes=sorted(bw),
                guard_status='DEPENDENCY_READY_ONLY' if not missing else 'UNSUPPORTED_GUARD',
                execution_status='NOT_ENABLED_REQUIRES_PHASE2_3',
                priority='earlier false guards in path must be known; no skipping them as if false'))
            visit(node.body,source,path+((line,True),),scope,bound,loops)
            if node.orelse is not None:
                otherwise=path+((line,False),)
                if isinstance(node.orelse,n.Block):
                    er,ec,ew=expand(node.orelse,bound)
                    branches.append(dict(id=f'{source}:{node.orelse.first_token.line}:else',source=source,
                        line=node.orelse.first_token.line,enclosing_function=scope,kind='Else',condition='else',
                        enclosing_bindings=sorted(bound),loop_context=list(loops),
                        path=list(otherwise),guard_reads=[],guard_calls=[],missing_guard=[],
                        body_missing=reasons(er),body_calls=sorted(ec),body_writes=sorted(ew),
                        guard_status='DEPENDENCY_READY_ONLY',execution_status='NOT_ENABLED_REQUIRES_PHASE2_3',
                        priority='else requires ALL preceding guards false; local readiness is not enough'))
                visit(node.orelse,source,otherwise,scope,bound,loops)
        else:
            for child in children(node):visit(child,source,path,scope,bound,loops)
    for name,tree in trees.items():visit(tree,name)
    guards={(x['source'],x['line']):x for x in branches if x['kind']!='Else'}
    for branch in branches:
        path_reads=set()
        for line,_ in branch['path']:
            ancestor=guards.get((branch['source'],line))
            if ancestor is None:raise ValueError('unresolved enclosing/priority guard')
            path_reads.update(ancestor['guard_reads'])
        branch['missing_path']=reasons(path_reads)
        branch['missing_loops']=[m for loop in branch['loop_context'] for m in loop['missing']]
        branch['path_status']='DEPENDENCY_READY_ONLY' if not branch['missing_guard'] and not branch['missing_path'] and not branch['missing_loops'] else 'UNSUPPORTED_PATH'
    return dict(schema='teacher_branch_dependencies_v1',
        evidence='STATIC_TRANSITIVE_OVERAPPROX_NOT_DYNAMIC_COVERAGE_OR_POLICY_EQUIVALENCE',
        encoding='CP932 text AST with original byte-offset map; no AST code is executed',
        parser_versions={name:version(name) for name in ('luaparser','antlr4-python3-runtime','multimethod')},
        sources={name:hashlib.sha256(data).hexdigest() for name,data in sources.items()},
        available=sorted(available),counts=dict(Counter(x['guard_status'] for x in branches)),
        path_counts=dict(Counter(x['path_status'] for x in branches)),
        branches=branches,
        assignments=assignments,
        functions={name:dict(source=source,line=node.first_token.line,
            reads=sorted(expand(node)[0]),writes=sorted(expand(node)[2]))
            for name,(source,node) in functions.items()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--site',required=True)
    parser.add_argument('--package',required=True)
    parser.add_argument('--api',required=True)
    parser.add_argument('--available',required=True)
    parser.add_argument('--output',required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    site = Path(args.site).resolve(strict=True)
    output = Path(args.output).resolve()
    if not site.is_relative_to(root/'rl/.tools') or not output.is_relative_to(root/'rl/runs'):
        raise ValueError('isolated tools/output must remain in approved repository directories')
    if output.exists():raise FileExistsError(output)
    sys.path.insert(0,str(site))
    result = audit(Path(args.package),Path(args.api),json.loads(Path(args.available).read_text('utf-8')))
    output.write_text(json.dumps(result,indent=2,ensure_ascii=True),encoding='utf-8')
    print(json.dumps(dict(counts=result['counts'],branches=len(result['branches']),output=str(output))))


if __name__ == '__main__':main()
