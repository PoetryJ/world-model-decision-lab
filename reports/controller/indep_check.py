"""Controller 20:15 independent re-check: (1) det repro == sealed per (method,state,cond); (2) stoch pairs recomputed from episodes.csv without A10 code."""
import csv, numpy as np
from collections import defaultdict
def load(d):
    return {(r['method'],r['init_state_id'],r['hidden_condition_id'],r['rep']):(r['success'],r['termination'],round(float(r['final_goal_err']),12),r['steps_used'],r['probe_steps']) for r in csv.DictReader(open(f"runs/controller/{d}/episodes.csv"))}
a,b=load("sealed"),load("repro_sealed_det")
print("det repro: n", len(a), len(b), "same keys", set(a)==set(b), "mismatches", sum(a[k]!=b[k] for k in a))
S=defaultdict(dict)
for r in csv.DictReader(open("runs/controller/sealed_stoch/episodes.csv")):
    S[r['method']][(r['init_state_id'],r['hidden_condition_id'],r['rep'])]=r['success']=='True'
def pair(x,y,sub=None):
    keys=[k for k in S[x] if sub is None or sub in k[0]]; d=defaultdict(list)
    for k in keys: d[k[0]].append(S[x][k]-S[y][k])
    m=np.array([np.mean(v) for _,v in sorted(d.items())]); rng=np.random.default_rng(0)
    bs=[m[rng.integers(0,len(m),len(m))].mean() for _ in range(2000)]
    return round(float(m.mean()),3), [round(float(v),3) for v in np.percentile(bs,[2.5,97.5])]
for x,y in [("wm_tap_update","wm_tap_noupdate"),("wm_goalrule_update","wm_noprobe"),("wm_taskvoi_update","wm_noprobe"),("wm_goalrule_update","direct_physics_obs"),("wm_tap_update","direct_history_tap")]:
    print("stoch", x, "-", y, "all", pair(x,y), "pos", pair(x,y,"pos"), "neg", pair(x,y,"neg"))
