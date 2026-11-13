"""Controller method builders for E2/E3 (runner-side wiring only; imports module code read-only, no copies).
Every builder returns (proposer, predictor, adapter, scorer, prober) for src.probing.loop.run_episode.
All methods share the SAME proposer (A4 scenarios.strike_grid K=54 H=6) and the SAME probe library (A4 probe_set).
"""
from __future__ import annotations
import time
import numpy as np
from shared.interface import Prediction
from src.probing.experiments import StrikeGridProposer, build as a8_build
from src.probing.probers import make_prober
from src.probing.belief import MuGridBelief

LAM = 0.01        # A8 frozen task_voi_lam
TIE_W = 1e-3      # A8 frozen scorer_tie_w (used inside A8 VOI / belief scorer only)

WM_METHODS = ("wm_noprobe", "wm_tap_noupdate", "wm_tap_update", "wm_random_update", "wm_goalrule_update", "wm_taskvoi_update")
BELIEF_METHODS = ("belief_noprobe", "belief_tap_noupdate", "belief_tap_update", "belief_goalrule_update", "belief_taskvoi_update")
# A9 direct route: selector x probe. All use the same strike_grid array and the same A4 probe feedback as the WM route.
DIRECT_SPEC = {
  "direct_obs_only":                 ("obs_only", "no_probe"),
  "direct_obs_only_tap":             ("obs_only", "fixed_probe"),
  "direct_history_noprobe":          ("history_direct", "no_probe"),
  "direct_history_tap":              ("history_direct", "fixed_probe"),
  "direct_history_goalrule":         ("history_direct", "goaldist_probe"),
  "direct_physics_obs":              ("physics_obs", "no_probe"),
  "direct_physics_obs_tap":          ("physics_obs", "fixed_probe"),
  "direct_physics_history_noprobe":  ("physics_history", "no_probe"),
  "direct_physics_history_tap":      ("physics_history", "fixed_probe"),
  "direct_physics_history_goalrule": ("physics_history", "goaldist_probe"),
}
DIRECT_METHODS = tuple(DIRECT_SPEC)
ALL_METHODS = WM_METHODS + BELIEF_METHODS + DIRECT_METHODS

PROBE_OF = {"noprobe": "no_probe", "tap": "fixed_probe", "random": "random_probe", "goalrule": "goaldist_probe",
            "taskvoi": "task_voi_probe", "obs": "no_probe", "history": None}

_CACHE = {}

# ---------------------------------------------------------------- WM route (A6 predictor + RLS + A7 scorer)
class WMSettledPredictor:
    """A6 EnsemblePredictor on candidates padded with A7 pad_for_settle (settle_pad from A7 frozen config)."""
    def __init__(self, base, settle):
        self.base, self.settle = base, settle
    def predict(self, observation, actions, context):
        from src.scoring import pad_for_settle
        P = self.base.predict(observation, pad_for_settle(actions, self.settle), context)
        return P

def _a6():
    if "a6" not in _CACHE:
        from src.dynamics.model import load_or_train
        _CACHE["a6"] = load_or_train(seed=0)          # manifest guarantees the frozen pkl exists (never retrains)
    return _CACHE["a6"]

def build_wm(method, seed):
    from src.adaptation.rls import RLSAdapter, NoAdapter
    from src.scoring import make_scorer, load_scorer_config
    cfg = load_scorer_config()
    base = _a6(); prop = StrikeGridProposer()
    pred = WMSettledPredictor(base, int(cfg["settle_pad"]))
    adapter = NoAdapter() if method.endswith("_noupdate") else RLSAdapter(base)   # wm_noprobe: RLS over empty history = prior
    key = method.split("_")[1]
    pname = PROBE_OF[key]
    if pname == "task_voi_probe":   # NOTE: A8 VOI decides whether to probe with its own mu-grid belief (labelled)
        prober = make_prober(pname, proposer=prop, seed=seed, lam=LAM, belief=MuGridBelief(), tie_w=TIE_W)
    else:
        prober = make_prober(pname, seed=seed)
    return prop, pred, adapter, make_scorer(config=cfg), prober

# ---------------------------------------------------------------- privileged structured-physics belief route (A8)
A8_NAME = {"belief_noprobe": "no_probe", "belief_tap_noupdate": "fixed_probe_noadapt", "belief_tap_update": "fixed_probe",
           "belief_goalrule_update": "goaldist_probe", "belief_taskvoi_update": "task_voi_probe"}

def build_belief(method, seed):
    return a8_build(A8_NAME[method], seed, LAM, "success", 0.05, 0.02, 1.0, TIE_W)

# ---------------------------------------------------------------- A9 direct route (no future prediction)
class DirectNoFuturePredictor:
    """Pass-through: NO future prediction. endpoints are NaN (so no endpoint error can be computed for direct methods);
    observation/history/candidates are handed to the scorer via Prediction.extra (not logged)."""
    def __init__(self, a9_proposer):
        self.a9_proposer, self.checked = a9_proposer, False
    def predict(self, observation, actions, context):
        A = np.asarray(actions, float)
        if not self.checked:   # same-candidates guarantee: A9 trained on its own strikegrid source
            B = np.asarray(self.a9_proposer.propose(observation, context["history"], 0), float)
            if B.shape != A.shape or not np.allclose(A, B, atol=1e-12):
                raise RuntimeError("A9 DirectProposer(strikegrid) != A4 StrikeGridProposer candidates")
            self.checked = True
        return Prediction(endpoints=np.full((len(A), 2), np.nan), samples=None, traj=None,
                          extra={"_obs": observation, "_history": context["history"], "_cands": A})

class DirectScorer:
    def __init__(self, selector): self.sel = selector
    def score(self, predictions, goal):
        ex = predictions.extra
        return np.asarray(self.sel.score_candidates(ex["_obs"], ex["_history"], ex["_cands"]), float)

class _NullAdapter:
    def update(self, history): return None

def a9_ready():
    try:
        from src.direct_baselines.api import load_frozen_a9
        b = load_frozen_a9()
        miss = [k for k in ("obs_only", "history_direct") if k not in b]
        return (not miss), (f"missing {miss}" if miss else "ok")
    except Exception as e:   # hook: report clearly, runner skips direct_* methods
        return False, f"A9 API import failed: {e!r}"

def build_direct(method, seed):
    ok, why = a9_ready()
    if not ok:
        raise RuntimeError(f"A9 direct baseline not ready ({why}); direct_* methods skipped")
    if "a9" not in _CACHE:
        from src.direct_baselines.api import load_frozen_a9
        _CACHE["a9"] = load_frozen_a9()
    b = _CACHE["a9"]
    skey, pname = DIRECT_SPEC[method]
    sel = b[skey]
    return StrikeGridProposer(), DirectNoFuturePredictor(b["proposer"]), _NullAdapter(), DirectScorer(sel), make_prober(pname, seed=seed)

def build(method, seed):
    if method.startswith("wm_"): return build_wm(method, seed)
    if method.startswith("belief_"): return build_belief(method, seed)
    if method.startswith("direct_"): return build_direct(method, seed)
    raise KeyError(method)

ROUTE_LABEL = {"wm": "A6 learned ensemble (S=5) + RLS residual + A7 robust_success_then_distance [toy-run]",
               "belief": "PRIVILEGED model class: A8 analytic push model + mu-grid Bayes belief (knows the env's functional form) [toy-run]",
               "direct": "A9 direct selectors [toy-run]; see SELECTOR_LABEL"}
SELECTOR_LABEL = {
  "obs_only": "learned ridge scorer on (current obs, candidate) features; ignores history; no dynamics model, no rollout",
  "history_direct": "learned ridge scorer + probe-feedback features (obj/pusher displacement, mobility, residual speed) and history x candidate interactions; no dynamics model, no rollout",
  "physics_obs": "hand-coded mechanics heuristic, NO learning: object follows pusher during contact, then slide = v^2/(2a) with fixed a = 0.35*g (code MU_NOMINAL=0.35; class docstring says 0.25 - code binds); a closed-form analytic endpoint model (not a learned/rolled-out world model); ignores history",
  "physics_history": "same hand-coded closed-form endpoint heuristic, but the single scalar a is estimated from tap_fast release in the executed probe (else nominal prior); 1 parameter, no learning, no env branching",
}
