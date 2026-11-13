"""LOCAL STUBS (A8-owned) used until A5 proposer / A6 predictor+adapter / A7 scorer are wired.
They implement the frozen Protocols. Belief-based: adapt_state = posterior over mu grid (visible history).
"""
from __future__ import annotations
import numpy as np
from shared.interface import DT, V_MAX, Prediction, clip_action
from .physics_model import BatchPushModel
from .belief import MuGridBelief, last_command
from .probe_library import RR, _unit

SPEEDS = (0.15, 0.25, 0.35, 0.5)
PUSH_STEPS = (2, 4, 6, 8)

class StubProposer:
    """K=16 fixed library: push along pusher->object line at speed s for n steps, then hold (H=10)."""
    def __init__(self, H: int = 10, speeds=SPEEDS, push_steps=PUSH_STEPS):
        self.H, self.speeds, self.push_steps = H, speeds, push_steps
    def propose(self, observation, history, seed):
        p, o, g = (np.asarray(observation[k]) for k in ("pusher", "obj", "goal"))
        u = _unit(o - p)
        out = []
        for s in self.speeds:
            for n in self.push_steps:
                seg = np.zeros((self.H, 2)); seg[:min(n, self.H)] = u * s
                out.append(seg)
        return clip_action(np.array(out))

class BeliefAdapter:
    def __init__(self, belief: MuGridBelief | None = None):
        self.belief = belief or MuGridBelief()
    def update(self, history):
        return {"grid": self.belief.grid.copy(), "w": self.belief.posterior(history),
                "pv0": last_command(history)}

class NoAdaptAdapter(BeliefAdapter):
    """Ablation: same probes executed, belief never updated (prior only)."""
    def update(self, history):
        return {"grid": self.belief.grid.copy(), "w": self.belief.prior(), "pv0": last_command(history)}

class BeliefPredictor:
    """Mixture prediction under posterior. endpoints = posterior-mean object pos at END of segment (K,2).
    samples = per-grid-mu endpoints (K,M,2); extra: weights (M,), settled rest positions (K,M,2), oob (K,M)."""
    def __init__(self, model: BatchPushModel | None = None, top_m: int = 12):
        self.model, self.top_m = model or BatchPushModel(), top_m
    def predict(self, observation, actions, context):
        st = context["adapt_state"]
        grid, w = st["grid"], st["w"]
        idx = np.argsort(-w)[: self.top_m]; idx = np.sort(idx)
        mus, ww = grid[idx], w[idx] / w[idx].sum()
        A = np.asarray(actions, float); K, H, _ = A.shape; M = len(mus)
        B = K * M
        r = self.model.rollout(np.tile(observation["pusher"], (B, 1)), np.tile(observation["obj"], (B, 1)),
                               np.tile(observation["obj_vel"], (B, 1)), np.tile(mus, K),
                               np.repeat(A, M, axis=0), pv=st.get("pv0"))
        rest, _ = self.model.settle(r["o"], r["ov"], np.tile(mus, K))
        end = r["o"].reshape(K, M, 2)
        return Prediction(endpoints=np.einsum("kmd,m->kd", end, ww), samples=end,
                          traj=None, extra={"weights": ww, "mus": mus, "settled": rest.reshape(K, M, 2),
                                            "oob": r["oob"].reshape(K, M)})

def expected_cost(pred: Prediction, goal: dict, tie_w: float = 0.0):
    """Per-candidate expected cost under the mixture: max(dist(rest,goal)-r,0) + 1*oob + tie_w*dist. (K,)
    tie_w>0 (v2) breaks the exact-zero hinge ties toward candidates landing nearer the goal centre (margin)."""
    rest = pred.extra.get("settled", pred.samples if pred.samples is not None else pred.endpoints[:, None])
    w = pred.extra.get("weights", np.full(rest.shape[1], 1.0 / rest.shape[1]))
    d = np.linalg.norm(rest - np.asarray(goal["goal"])[None, None], axis=2)
    c = np.maximum(d - goal["goal_radius"], 0.0) + tie_w * d
    if "oob" in pred.extra:
        c = c + pred.extra["oob"].astype(float)
    return c @ w

class StubScorer:
    """score = -expected cost (higher = better). Remaining ties broken by first index (argmax).
    tie_w=0.0 -> v1 (pure hinge); tie_w=1e-3 -> v2 (margin tie-break)."""
    def __init__(self, tie_w: float = 0.0): self.tie_w = tie_w
    def score(self, predictions, goal):
        return -expected_cost(predictions, goal, self.tie_w)
