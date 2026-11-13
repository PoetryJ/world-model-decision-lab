"""A10 minimal repro for the A5-vs-A10 E1 discrepancy. [toy-run] Runs on frozen e1 only.
python tests/evaluation/e1_reconcile_repro.py -> runs/a10/e1_reconcile.json"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np
from evaluation.audit import load_frozen, ROOT
from evaluation.offline_rank import rank_candidates
from src.env.push_env import PushEnv
from src.candidates.fixed import FixedLibraryProposer
from src.candidates.release import TimedReleaseProposer
from src.env.scenarios import strike_grid

E1 = load_frozen("e1", who="A10 e1_reconcile")
GRIDS = {"A4grid_0.1/0.25/0.5": {"mu_lo": 0.1, "mu_mid": 0.25, "mu_hi": 0.5},
         "frozen_0.1/0.3/0.6": {"mu_lo": 0.1, "mu_mid": 0.3, "mu_hi": 0.6}}
def strike4(o):
    C, lab = strike_grid(o); return C[[lab.index(x) for x in [(0.2, 6), (0.3, 4), (0.4, 3), (0.5, 2)]]]
LIBS = {"A5_FixedLibrary_K4_H10 (what A10 E1 used)": lambda o: FixedLibraryProposer(K=4).propose(o, [], 0),
        "A5_TimedRelease_K4_H6 (A5 primary, file created 18:09)": lambda o: TimedReleaseProposer(K=4).propose(o, [], 0),
        "A10_STRIKE4": strike4,
        "A4_STRIKE54 (E2 primary)": lambda o: strike_grid(o)[0]}
out = {}
for gname, g in GRIDS.items():
    for lname, f in LIBS.items():
        for settle in (20, 30):
            per = {}
            for cid, mu in g.items():
                succ = []
                for st in E1:
                    env = PushEnv(); obs = env.reset(st["init"], {"mu": mu}, st["seed"])
                    r = rank_candidates(env, env.snapshot(), f(obs), select_rep_seeds=(11,), eval_rep_seeds=(21,), settle_steps=settle)
                    succ.append(max(r["true_success_rate"]) == 1.0)
                per[cid] = float(np.mean(succ))
            per["overall"] = float(np.mean(list(per.values())))
            out[f"{gname} | {lname} | settle{settle}"] = per
            print(f"{gname:22s} {lname:55s} settle{settle}: oracle succ", {k: round(v, 3) for k, v in per.items()})
json.dump(out, open(os.path.join(ROOT, "runs", "a10", "e1_reconcile.json"), "w"), indent=1)
