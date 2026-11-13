"""A4 scenario generators for PushEnv (TOY).  All generators are deterministic given `seed`.

Returns lists of (init_state_id, init_state) and (hidden_condition_id, hidden).
init_state = {"pusher": [x,y], "obj": [x,y], "goal": [x,y], "goal_radius": float, "meta": {...}}
  meta is visible-only bookkeeping (set name, negative_control flag, goal distance); it never
  contains hidden params, so it is safe for policy-side code, but policies should not need it.
Hidden conditions apply identically to every init state => (init_state, hidden) pairs share the
same visible initial observation (criterion E0-1).

Sets:
  "pos"  : goal FAR (0.22-0.30 m from object centre) -> must release/slide; friction should change
           the best segment.  negative_control=False.
  "neg"  : goal NEAR (0.05-0.08 m) -> slow continuous-contact push suffices; conditions differ but the
           best action should be the same.  negative_control=True.
Probes: probe_set(obs) returns 3 legal probe segments (h=5, shape (5,2)) built from VISIBLE obs only.
"""
from __future__ import annotations
import numpy as np
from shared.interface import DT, V_MAX, clip_action
from .push_env import R_P, R_O

HIDDEN_CONDITIONS = [("mu_lo", {"mu": 0.10}), ("mu_mid", {"mu": 0.25}), ("mu_hi", {"mu": 0.50})]
GOAL_RADIUS = 0.03
INIT_GAP = 0.02          # pusher surface gap behind object [m]
TASK_H = 6               # default task segment length for strike grid (0.6 s)


def hidden_conditions(n: int = 3):
    """[(hidden_condition_id, hidden)]; n=2 -> lo/hi, n=3 -> lo/mid/hi."""
    if n == 2:
        return [HIDDEN_CONDITIONS[0], HIDDEN_CONDITIONS[2]]
    return list(HIDDEN_CONDITIONS[:n])


def _make_state(rng, goal_dist_range, negative_control, set_name):
    for _ in range(1000):
        obj = rng.uniform(-0.15, 0.15, 2)
        ang = rng.uniform(-np.pi, np.pi)
        u = np.array([np.cos(ang), np.sin(ang)])
        d = rng.uniform(*goal_dist_range)
        goal = obj + d * u
        pusher = obj - (R_O + R_P + INIT_GAP) * u
        # keep goal + generous overshoot margin and pusher on table
        if np.all(np.abs(goal + 0.10 * u) < 0.45) and np.all(np.abs(pusher) < 0.45):
            return {"pusher": pusher.tolist(), "obj": obj.tolist(), "goal": goal.tolist(),
                    "goal_radius": GOAL_RADIUS,
                    "meta": {"set": set_name, "negative_control": negative_control,
                             "goal_dist": float(d)}}
    raise RuntimeError("could not sample state")


def init_states(n: int, seed: int, negative_control: bool = False, prefix: str | None = None):
    """[(init_state_id, init_state)].  negative_control=True -> near-goal 'neg' set."""
    rng = np.random.default_rng(seed)
    name = "neg" if negative_control else "pos"
    rng_range = (0.05, 0.08) if negative_control else (0.22, 0.30)
    prefix = prefix or f"{name}_s{seed}"
    return [(f"{prefix}_{i:03d}", _make_state(rng, rng_range, negative_control, name)) for i in range(n)]


# ---- frozen sets (seeds recorded) ----
E0_SEED_POS, E0_SEED_NEG = 1000, 2000


def e0_set(negative_control: bool = False):
    """E0 gate: 12 paired init states (x all hidden_conditions(3))."""
    return init_states(12, E0_SEED_NEG if negative_control else E0_SEED_POS, negative_control,
                       prefix=("e0neg" if negative_control else "e0pos"))


def e1_set(negative_control: bool = False):
    """Suggested E1 set: 20 init states (A10 may freeze its own; seeds 1100/2100)."""
    return init_states(20, 2100 if negative_control else 1100, negative_control,
                       prefix=("e1neg" if negative_control else "e1pos"))


def e2_test_set(negative_control: bool = False):
    """Suggested E2 test set: 40 init states (seeds 1200/2200). A10 owns the final freeze."""
    return init_states(40, 2200 if negative_control else 1200, negative_control,
                       prefix=("e2neg" if negative_control else "e2pos"))


def train_set(n: int = 200, seed: int = 9000, negative_control: bool | None = None):
    """Training/validation pool for model-fitting agents (seed range 9000+, disjoint from eval seeds).
    negative_control=None -> half pos, half neg."""
    if negative_control is None:
        a = init_states(n // 2, seed, False, prefix=f"trpos_s{seed}")
        b = init_states(n - n // 2, seed + 1, True, prefix=f"trneg_s{seed + 1}")
        return a + b
    return init_states(n, seed, negative_control)


# ---- probes (visible-only construction) ----
PROBE_H = 5
PROBE_NAMES = ("tap_fast", "push_slow", "retreat")


def _dir_to_obj(obs):
    d = np.asarray(obs["obj"], float) - np.asarray(obs["pusher"], float)
    n = np.linalg.norm(d)
    return d / n if n > 1e-9 else np.array([1.0, 0.0])


def probe_set(obs):
    """dict name -> (5,2) legal probe segment, from visible obs only.
    tap_fast : 0.4 m/s toward object for 2 steps, then 3 zero steps (object released, slides; watch decel)
    push_slow: 0.1 m/s toward object for 5 steps (continuous contact; expected NON-informative about mu)
    retreat  : 0.1 m/s away from object for 5 steps (no contact; null probe, zero info)
    Each costs 5 control steps (0.5 s) of the episode budget and changes the physical state (tap/push move the
    object toward/along the pusher->object line)."""
    u = _dir_to_obj(obs)
    tap = np.zeros((PROBE_H, 2)); tap[:2] = 0.4 * u
    push = np.tile(0.1 * u, (PROBE_H, 1))
    ret = np.tile(-0.1 * u, (PROBE_H, 1))
    return {k: clip_action(v) for k, v in zip(PROBE_NAMES, (tap, push, ret))}


# ---- fixed candidate grid used by A4's E0 criterion-2 (others may reuse; A5 owns proposers) ----
STRIKE_SPEEDS = tuple(np.round(np.arange(0.10, 0.501, 0.05), 3))   # 9 speeds
def strike_grid(obs, H: int = TASK_H):
    """Fixed grid (K=9*H, H, 2): push along visible pusher->goal-through-object direction at speed s for n
    steps (n=1..H), then zero.  Built from visible obs only.  Returns (cands, labels[(s,n)])."""
    d = np.asarray(obs["goal"], float) - np.asarray(obs["obj"], float)
    u = d / max(np.linalg.norm(d), 1e-9)
    cands, labels = [], []
    for s in STRIKE_SPEEDS:
        for n in range(1, H + 1):
            a = np.zeros((H, 2)); a[:n] = s * u
            cands.append(clip_action(a)); labels.append((float(s), n))
    return np.stack(cands), labels
