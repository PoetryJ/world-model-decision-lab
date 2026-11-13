"""A10 G2 runtime leak check on actual policy stacks (A8 belief stack, A8+A6 stack) on VAL states. [toy-run]
python tests/evaluation/runtime_leak_g2.py -> runs/a10/g2_runtime.json"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from evaluation import audit as A
from src.env.push_env import PushEnv
from src.probing.loop import run_episode
from src.probing import experiments as X

class TrackedEnv(PushEnv):
    def reset(self, init_state, hidden, seed):
        o = super().reset(init_state, hidden, seed); A.track_env_hidden(self); return o

val = A.load_frozen("val", who="A10 G2 runtime")[:3]
conds = {"mu_lo": {"mu": 0.1}, "mu_hi": {"mu": 0.6}}
out = {"label": "[toy-run]", "results": {}}
for stack in ("a8", "a6"):
    for m in X.METHODS:
        A.TrackedHidden.violations = []
        viol = []
        try:
            for st in val:
                for cid, h in conds.items():
                    comps = X.build_a6(m, st["seed"], 0.01, 0.0) if stack == "a6" else X.build(m, st["seed"], 0.01)
                    G = [A.GuardedPolicy(c, f"{stack}.{m}.{type(c).__name__}") for c in comps]
                    run_episode(TrackedEnv(), st["init"], h, st["seed"], *G, step_budget=60, probe_budget=5,
                                method=m, max_task_decisions=1)
                    for g in G: viol += g.violations
            out["results"][f"{stack}/{m}"] = {"arg_violations": viol[:10], "n_arg_viol": len(viol),
                                             "hidden_access_from_policy": A.TrackedHidden.violations[:10],
                                             "n_hidden_access": len(A.TrackedHidden.violations)}
        except Exception as e:
            out["results"][f"{stack}/{m}"] = {"error": f"{type(e).__name__}: {e}"[:300]}
json.dump(out, open(os.path.join(A.ROOT, "runs", "a10", "g2_runtime.json"), "w"), indent=1, default=str)
for k, v in out["results"].items(): print(k, {kk: vv for kk, vv in v.items() if kk.startswith("n_") or kk == "error"})
