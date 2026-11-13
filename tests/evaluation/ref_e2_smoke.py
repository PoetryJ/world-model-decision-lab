"""A10 analyzer smoke: two trivial reference policies, E2 A-3 structure, on VAL states (never test). [toy-run]
python tests/evaluation/ref_e2_smoke.py -> runs/a10/ref_e2_val_decisions.jsonl"""
import json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np
from shared.interface import DecisionRecord, to_jsonable
from src.env.push_env import PushEnv
from src.env.scenarios import strike_grid, probe_set
from evaluation.audit import load_frozen, ROOT

def run(method, st, cid, h, f):
    env = PushEnv(); obs = env.reset(st["init"], h, st["seed"]); i = 0; eid = f"{method}|{st['init_state_id']}|{cid}"
    base = dict(run_id="a10_ref", episode_id=eid, method=method, init_state_id=st["init_state_id"], hidden_condition_id=cid, seed=st["seed"])
    def rec(kind, ex, before, after, term, **kw):
        nonlocal i
        r = DecisionRecord(**base, step_index=i, kind=kind, candidates=kw.get("C"), predictions=None, scores=kw.get("S"),
                           chosen_index=kw.get("k"), executed=ex, obs_before=before, obs_after=after,
                           wall_time_s=kw.get("w", {}), termination=term, candidate_counts=kw.get("cc"))
        f.write(json.dumps(to_jsonable(r.__dict__)) + "\n"); i += 1
    term = None
    if method == "tap_then_fixed":
        seg = probe_set(obs)["tap_fast"]; res = env.step_segment(seg); o2 = res[-1].obs
        term = res[-1].info.get("termination"); rec("probe", seg[:len(res)], obs, o2, term); obs = o2
    if term is None:
        t0 = time.perf_counter(); C, lab = strike_grid(obs); C = C[[23, 29, 35, 41]]   # 4 fixed strikes
        S = -np.abs(np.arange(4) - 1.0); k = int(np.argmax(S)); w = {"propose": time.perf_counter() - t0}
        res = env.step_segment(C[k]); o2 = res[-1].obs; term = res[-1].info.get("termination")
        rec("task", C[k][:len(res)], obs, o2, term, C=C, S=S, k=k, w=w, cc={"requested": 4, "returned": 4, "failed": 0, "seeds": [0]}); obs = o2
    while term is None:
        if np.linalg.norm(env.ov) < 1e-3 and not env.done:   # evaluator-side settle check
            term = "budget"; break
        r = env.step(np.zeros(2)); term = r.info.get("termination")
        rec("task", [[0.0, 0.0]], obs, r.obs, term); obs = r.obs
    if term == "budget":   # mark last record
        pass
    return term

if __name__ == "__main__":
    val = load_frozen("val", who="A10 ref_e2_smoke")[:10]
    conds = {"mu_lo": {"mu": 0.1}, "mu_hi": {"mu": 0.6}}
    p = os.path.join(ROOT, "runs", "a10", "ref_e2_val_decisions.jsonl")
    with open(p, "w") as f:
        for m in ("fixed", "tap_then_fixed"):
            for st in val:
                for cid, h in conds.items():
                    run(m, st, cid, h, f)
    # patch: settle-terminated episodes need termination on their last record
    rows = [json.loads(l) for l in open(p)]
    last = {}
    for j, r in enumerate(rows): last[(r["method"], r["episode_id"])] = j
    for j in last.values():
        if rows[j]["termination"] is None: rows[j]["termination"] = "budget"
    with open(p, "w") as f:
        for r in rows: f.write(json.dumps(r) + "\n")
    print(p, len(rows))
