# SPDX-License-Identifier: Apache-2.0
"""Flag `x op= ...` where x has no plain assignment earlier in the function.

Run against the *installed* vllm inside the framework container by
kimi-k3-merged-v3-disagg-dcp.sh, over the files the patch chain rewrites.

Python raises UnboundLocalError for this at runtime; Triton's AST walker raises
NameError at *compile* time, which for a @triton.jit kernel is the first time
that specialization is compiled -- i.e. warmup, twenty-five minutes into a run.
A substring check on the file cannot see it.
"""
import ast, sys

def check(path):
    tree = ast.parse(open(path).read())
    bad = []
    for fn in ast.walk(tree):
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        params = {a.arg for a in fn.args.args + fn.args.kwonlyargs}
        assigned = {}          # name -> earliest plain-assign lineno
        for n in ast.walk(fn):
            if isinstance(n, ast.Assign):
                for t in n.targets:
                    for nm in ast.walk(t):
                        if isinstance(nm, ast.Name):
                            assigned.setdefault(nm.id, n.lineno)
                            assigned[nm.id] = min(assigned[nm.id], n.lineno)
            elif isinstance(n, ast.AnnAssign) and n.value is not None:
                # `x: T = ...` binds just as `x = ...` does.
                if isinstance(n.target, ast.Name):
                    assigned[n.target.id] = min(
                        assigned.get(n.target.id, n.lineno), n.lineno
                    )
            elif isinstance(n, (ast.For, ast.comprehension)):
                tgt = n.target
                for nm in ast.walk(tgt):
                    if isinstance(nm, ast.Name):
                        assigned.setdefault(nm.id, getattr(n, "lineno", 0))
        for n in ast.walk(fn):
            if isinstance(n, ast.AugAssign) and isinstance(n.target, ast.Name):
                name = n.target.id
                if name in params:
                    continue
                first = assigned.get(name)
                if first is None or first > n.lineno:
                    bad.append((fn.name, n.lineno, name))
    return bad

fail = 0
for p in sys.argv[1:]:
    for fnname, lineno, name in check(p):
        fail = 1
        print(f"  FAIL {p}:{lineno} in {fnname}(): '{name} op= ...' with no earlier assignment")
if not fail:
    print("  ok    no augmented assignment reads an unbound name")
sys.exit(fail)
