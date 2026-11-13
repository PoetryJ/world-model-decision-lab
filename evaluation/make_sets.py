"""A10: deterministic generator for FROZEN evaluation sets. Owner: A10 (evaluation/).
Generates init states (visible only) and hidden conditions with disjoint seed ranges per split.
Run once: python -m evaluation.make_sets  -> evaluation/frozen_sets.json (sha256 recorded in PROTOCOL.md).
Policies must never import this file's hidden values; runners join hidden via condition id.
"""
from __future__ import annotations
import hashlib, json, math, os
import numpy as np

VERSION = "a10-sets-v2"  # v2: geometry from A4 src/env/scenarios.init_states (pos far-goal + neg near-goal)
# Disjoint seed ranges (documented in PROTOCOL.md). Each init state i of split s uses seed BASE[s]+i.
SEED_BASE = {"train": 100_000, "val": 200_000, "test": 300_000, "e0": 400_000, "e1": 500_000}
GOAL_RADIUS = 0.05
# Hidden friction conditions (env-side only). Main = clearly different; neg = near-identical high friction.
CONDITIONS = {
    "mu_lo": {"mu": 0.1},
    "mu_mid": {"mu": 0.3},
    "mu_hi": {"mu": 0.6},
}
# Negative control (v2) = A4 'neg' near-goal geometry (0.05-0.08 m) under the SAME 3 test conditions:
# conditions differ, best action expected the same (slow continuous-contact push). Qualifies only via G5.
NEG_CONTROL_CONDITIONS = CONDITIONS
TRAIN_VAL_CONDITIONS = {      # mu=0.3 (test 'mu_mid') deliberately HELD OUT of train/val (interpolation test);
    "mu_lo": {"mu": 0.1}, "tr_mu_0.15": {"mu": 0.15}, "tr_mu_0.2": {"mu": 0.2},   # mu_lo/mu_hi seen in train
    "tr_mu_0.4": {"mu": 0.4}, "tr_mu_0.5": {"mu": 0.5}, "mu_hi": {"mu": 0.6},     # (episode-disjoint only)
}

def gen_init(seed: int) -> dict:
    """Object in [-0.25,0.25]^2, goal 0.15-0.30 m away inside [-0.35,0.35]^2, pusher 0.10 m behind
    the object (opposite goal direction) with +-20 deg jitter. Pusher not in contact (contact dist 0.08)."""
    rng = np.random.default_rng(seed)
    for _ in range(1000):
        obj = rng.uniform(-0.25, 0.25, 2)
        ang = rng.uniform(-math.pi, math.pi)
        d = rng.uniform(0.15, 0.30)
        goal = obj + d * np.array([math.cos(ang), math.sin(ang)])
        if np.any(np.abs(goal) > 0.35):
            continue
        back = ang + math.pi + rng.uniform(-math.radians(20), math.radians(20))
        pusher = obj + 0.10 * np.array([math.cos(back), math.sin(back)])
        if np.any(np.abs(pusher) > 0.45):
            continue
        r = lambda v: [round(float(x), 4) for x in v]
        return {"pusher": r(pusher), "obj": r(obj), "goal": r(goal), "goal_radius": GOAL_RADIUS}
    raise RuntimeError("no valid init")

def make_split(name: str, n: int, neg: bool = False) -> list[dict]:
    """v2: A4 scenarios.init_states(n, seed=SEED_BASE[name] (+1 for neg)). env seed = base+100+i (pos) / base+600+i (neg).
    A4 'meta' is dropped from init (visible bookkeeping only) and kept as a separate field."""
    import sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.env.scenarios import init_states
    base = SEED_BASE[name]
    tag = "neg" if neg else "pos"
    out = []
    for i, (_, st) in enumerate(init_states(n, base + (1 if neg else 0), negative_control=neg)):
        init = {k: ([round(float(x), 6) for x in v] if isinstance(v, list) else v) for k, v in st.items() if k != "meta"}
        out.append({"init_state_id": f"{name}_{tag}_{i:03d}", "seed": base + (600 if neg else 100) + i,
                    "init": init, "meta": st["meta"]})
    return out

def build() -> dict:
    sets = {"version": VERSION, "seed_base": SEED_BASE, "goal_radius": GOAL_RADIUS,
            "conditions": {"test": CONDITIONS, "neg_control": NEG_CONTROL_CONDITIONS, "train_val": TRAIN_VAL_CONDITIONS}}
    sets["train"] = make_split("train", 200); sets["train_neg"] = make_split("train", 100, neg=True)
    sets["val"] = make_split("val", 40); sets["val_neg"] = make_split("val", 20, neg=True)
    sets["e0"] = make_split("e0", 12); sets["e0_neg"] = make_split("e0", 12, neg=True)
    sets["e1"] = make_split("e1", 20)
    sets["test"] = make_split("test", 40)   # E2/E3 frozen test init states
    sets["test_neg"] = make_split("test", 40, neg=True)   # E3 negative control
    # E0 probes: 3 legal probe segments (h=3, 0.3 s), same for all methods. Units m/s, world frame.
    # Expressed in the object->goal frame: 'push_fwd' = toward-goal speed 0.3 m/s; 'push_fast' 0.5; 'tap' 0.2 for 2 steps.
    sets["e0_probes"] = {"source": "src.env.scenarios.probe_set(obs) (A4): tap_fast, push_slow, retreat; h=5 each",
                         "names": ["tap_fast", "push_slow", "retreat"]}
    # E2 budget (binding, same for all methods)
    sets["e2_budget"] = {"max_steps_total": 60, "max_probe_steps": 10, "dt": 0.1, "max_physical_time_s": 6.0,
                         "termination": ["success", "out_of_bounds", "timeout", "budget"],
                         "env_seed_rule": "env seed = test seed (same for all methods; CRN)",
                         "stochastic_variant": {"action_noise": 0.02, "n_reps": 3,
                                                 "rep_seed_rule": "seed*10 + rep"}}
    # Offline ranking reps (CRN). oracle-selection reps and evaluation reps are disjoint.
    sets["offline"] = {"select_rep_seeds": [11, 12, 13], "eval_rep_seeds": [21, 22, 23],
                       "settle_steps": 20, "tie_tol_m": 1e-3}
    return sets

def sha256_of(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()

if __name__ == "__main__":
    out = os.path.join(os.path.dirname(__file__), "frozen_sets.json")
    if os.path.exists(out):
        raise SystemExit(f"{out} exists; frozen sets are immutable. Delete only with controller approval.")
    s = build()
    with open(out, "w") as f:
        json.dump(s, f, indent=1, sort_keys=True)
    print(out, sha256_of(out))

# ---------------- A-8 sealed test split (controller authority) ----------------
SEALED_BASE = 700_000   # reserved range 700000-799999; disjoint from all ranges above and from A4/A6/A8 seeds
SEALED_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sealed", "test_sealed.json")

def build_sealed() -> dict:
    SEED_BASE["test_sealed"] = SEALED_BASE
    pos = make_split("test_sealed", 40); neg = make_split("test_sealed", 40, neg=True)
    for r in neg: r["init_state_id"] = r["init_state_id"].replace("test_sealed_neg", "test_sealed_neg")
    return {"version": "a10-sealed-v1", "seed_base": SEALED_BASE, "conditions": {"test": CONDITIONS},
            "goal_radius_from": "A4 scenarios (0.03)", "e2_budget": "PROTOCOL A-3",
            "test_sealed": pos, "test_sealed_neg": neg}

def write_sealed():
    if os.path.exists(SEALED_PATH):
        raise SystemExit("sealed split exists; immutable")
    os.makedirs(os.path.dirname(SEALED_PATH), exist_ok=True)
    s = build_sealed()
    with open(SEALED_PATH, "w") as f: json.dump(s, f, indent=1, sort_keys=True)
    os.chmod(SEALED_PATH, 0o444)
    return sha256_of(SEALED_PATH)
