"""A8 experiment driver (TOY). Paired runs: every method sees the same (init_state, hidden, env seed).
Usage: python -m src.probing.experiments --set a4_e0_pos --budget 60 --probe_budget 9 --tag X
Sets: a4_e0_pos / a4_e0_neg (A4 scenarios.e0_set, 12 states x 3 mu), a4_e1_pos / a4_e1_neg (20 states),
      a10_test (A10 frozen test, 40 states x 3 mu; env seed = test seed per A10 rule), a10_e1.
Writes runs/a8/<tag>/{episodes.jsonl, decisions.jsonl, config.json}.
Hidden params are passed ONLY to env.reset (inside loop.run_episode) and to summary labels by id.
"""
from __future__ import annotations
import argparse, json, os, time
import numpy as np
from src.env.push_env import PushEnv
from src.env import scenarios as SC
from .physics_model import BatchPushModel
from .belief import MuGridBelief
from .stubs import BeliefPredictor, BeliefAdapter, NoAdaptAdapter, StubScorer
from .probers import make_prober
from .loop import run_episode

class StrikeGridProposer:
    """Wraps A4 scenarios.strike_grid (9 speeds x n=1..H steps along visible obj->goal dir). Deterministic."""
    def __init__(self, H=SC.TASK_H): self.H = H
    def propose(self, observation, history, seed):
        return SC.strike_grid(observation, self.H)[0]

A6_METHODS = ("no_probe", "fixed_probe", "fixed_probe_noadapt", "random_probe", "infogain_probe", "task_voi_probe",
              "goaldist_probe", "disagree_probe")
METHODS = ("no_probe", "fixed_probe", "fixed_probe_noadapt", "random_probe", "infogain_probe", "task_voi_probe",
           "goaldist_probe")

def load_set(name):
    if name.startswith("a4_"):
        _, e, pn = name.split("_")
        fn = {"e0": SC.e0_set, "e1": SC.e1_set, "e2": SC.e2_test_set}[e]
        states = fn(negative_control=(pn == "neg"))
        conds = sorted(A10_CONDS.items())   # binding conditions (A10 frozen): mu 0.1 / 0.3 / 0.6
        return [(sid, {k: v for k, v in st.items() if k != "meta"}, cid, h, 1000 + i)
                for i, (sid, st) in enumerate(states) for cid, h in conds]
    from evaluation.audit import load_frozen
    split = name[len("a10_"):]           # e.g. e0, e0_neg, val, val_neg (test* BLOCKED for A8, decision A-8)
    if split.startswith("test") and os.environ.get("A8_ALLOW_TEST") != "controller":
        raise SystemExit("A-8: A8 must not load test/test_neg/test_sealed*; controller sets A8_ALLOW_TEST=controller")
    rows = load_frozen(split, who="A8")
    conds = A10_CONDS   # = PROTOCOL §1 binding conditions (mu 0.1/0.3/0.6)
    return [(r["init_state_id"], r["init"], cid, conds[cid], r["seed"]) for r in rows for cid in sorted(conds)]

A10_CONDS = {"mu_lo": {"mu": 0.1}, "mu_mid": {"mu": 0.3}, "mu_hi": {"mu": 0.6}}

_A6 = {}
def build_a6(method, seed, lam, tie_w):
    """Task decision by A6 ensemble+RLS (padded to settled outcome); probers as in build(); A8 belief only inside
    infogain/task_voi probers."""
    from src.dynamics.model import load_or_train
    from src.adaptation.rls import RLSAdapter, NoAdapter
    from .a6_wiring import PaddedPredictor, DisagreementProbe
    if "pred" not in _A6: _A6["pred"] = load_or_train(seed=0)
    base = _A6["pred"]; pred = PaddedPredictor(base)
    adapter = NoAdapter() if method.endswith("noadapt") else RLSAdapter(base)
    prop = StrikeGridProposer(); pname = method.replace("_noadapt", "")
    if pname == "disagree_probe":
        prober = DisagreementProbe(pred, RLSAdapter(base))
    elif pname == "task_voi_probe":
        prober = make_prober(pname, proposer=prop, seed=seed, lam=lam, belief=MuGridBelief(), tie_w=tie_w)
    else:
        prober = make_prober(pname, seed=seed)
    return prop, pred, adapter, StubScorer(tie_w=tie_w), prober

def build(method, seed, lam, voi_value="success", agent_ro=0.05, agent_tau=0.02, sigma_scale=1.0, tie_w=0.0):
    """agent_ro / agent_tau: AGENT-side model assumptions (mismatch studies); env is untouched."""
    model = BatchPushModel(r_o=agent_ro, tau_act=agent_tau)
    mk = lambda: MuGridBelief(model=model, sigma_pos=0.003 * sigma_scale, sigma_vel=0.02 * sigma_scale)
    prop = StrikeGridProposer()
    adapter = NoAdaptAdapter(mk()) if method.endswith("noadapt") else BeliefAdapter(mk())
    pname = method.replace("_noadapt", "")
    if pname == "task_voi_probe":
        prober = make_prober(pname, proposer=prop, seed=seed, lam=lam, value=voi_value, belief=mk(), tie_w=tie_w)
    elif pname == "infogain_probe":
        prober = make_prober(pname, seed=seed, belief=mk())
    else:
        prober = make_prober(pname, seed=seed)
    return prop, BeliefPredictor(model=model), adapter, StubScorer(tie_w=tie_w), prober

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True); ap.add_argument("--budget", type=int, default=60)
    ap.add_argument("--probe_budget", type=int, default=5); ap.add_argument("--tag", required=True)
    ap.add_argument("--methods", default=",".join(METHODS)); ap.add_argument("--lam", type=float, default=0.01)
    ap.add_argument("--action_noise", type=float, default=0.0); ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--voi_value", default="success")
    ap.add_argument("--agent_ro", type=float, default=0.05); ap.add_argument("--agent_tau", type=float, default=0.02)
    ap.add_argument("--sigma_scale", type=float, default=1.0)
    ap.add_argument("--rep", type=int, default=-1, help="stochastic variant rep r>=0: env seed = seed*10+r (A-3)")
    ap.add_argument("--stack", default="a8", help="a8 = A8 belief stubs; a6 = A6 ensemble+RLS for task decisions")
    ap.add_argument("--tie_w", type=float, default=0.0, help="v2 scorer/VOI margin tie-break (1e-3)")
    ap.add_argument("--one_shot", action="store_true", help="max_task_decisions=1 (commit to one strike)")
    a = ap.parse_args()
    out = os.path.join("runs/a8", a.tag); os.makedirs(out, exist_ok=True)
    json.dump(vars(a), open(os.path.join(out, "config.json"), "w"), indent=1)
    items = load_set(a.set)
    if a.stack == "a6" and a.methods == ",".join(METHODS): a.methods = ",".join(A6_METHODS)
    if a.limit: items = items[: a.limit]
    if a.stack == "a6":
        from src.dynamics.model import DEFAULT_CFG
        cfg = vars(a).copy(); cfg["a6_model"] = f"runs/a6/models/ens_{DEFAULT_CFG['env']}_{DEFAULT_CFG['version']}_s0.pkl"
        json.dump(cfg, open(os.path.join(out, "config.json"), "w"), indent=1)
    fe = open(os.path.join(out, "episodes.jsonl"), "w"); fd = open(os.path.join(out, "decisions.jsonl"), "w")
    t0 = time.time()
    for m in a.methods.split(","):
        for sid, st, cid, h, seed in items:
            if a.rep >= 0: seed = seed * 10 + a.rep
            if a.stack == "a6":
                prop, pred, adp, sc, pr = build_a6(m, seed, a.lam, a.tie_w)
            else:
                prop, pred, adp, sc, pr = build(m, seed, a.lam, a.voi_value, a.agent_ro, a.agent_tau, a.sigma_scale, a.tie_w)
            env = PushEnv(action_noise=a.action_noise)
            s, recs = run_episode(env, st, h, seed, prop, pred, adp, sc, pr, step_budget=a.budget,
                                  probe_budget=a.probe_budget, method=m, run_id=a.tag,
                                  episode_id=f"{a.tag}/{m}/{sid}/{cid}" + (f"/r{a.rep}" if a.rep >= 0 else ""), init_state_id=sid,
                                  hidden_condition_id=cid,
                                  max_task_decisions=(1 if a.one_shot else None))
            fe.write(json.dumps(s) + "\n")
            for r in recs: fd.write(json.dumps(r) + "\n")
        fe.flush(); fd.flush()
        print(m, "done", round(time.time() - t0, 1), "s", flush=True)

if __name__ == "__main__":
    main()
