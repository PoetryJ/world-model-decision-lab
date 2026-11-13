"""A6 low-cost adaptation: Bayesian linear / recursive-least-squares residual on segment endpoints (TOY).

Residual model:  obj_end_true - obj_end_base  ~=  F(segment) @ w,   w in R^J shared over x,y.
F (2,J) columns are 2-vectors computed from the BASE model rollout of the executed segment
(only observable quantities + known geometry):
  f1 predicted object displacement over segment      (o_H - o_0)
  f2 predicted final object velocity * T_seg          (v_H * H*DT)
  f3 predicted free-sliding displacement (no-contact steps)
  f4 observed initial object velocity * T_seg         (v_0 * H*DT)
FROZEN default (18:22, selected on VAL only, rule in reports/a6/FROZEN_CONFIG.json): feat_idx=(2,) [f3 only],
prior_std=5.0, noise_std=0.005.
Intuition: friction mismatch mostly rescales sliding distance (f3) and post-push velocity (f2).
Prior w ~ N(0, prior_std^2 I), noise N(0, noise_std^2). Posterior computed by RLS recursion over the
history in order (equivalent to batch ridge). update() is a pure function of history (no hidden params).
Each test episode must call update() on ITS OWN history only (fresh state = update([])).

Usage:
    from src.adaptation.rls import RLSAdapter
    ad = RLSAdapter(pred)                 # pred from src.dynamics.model.load_or_train
    st = ad.update(history)               # history: list[Transition]; [] -> prior (w=0) == no adaptation
    P  = pred.predict(obs, cands, {"history": history, "adapt_state": st})
"""
from __future__ import annotations
import numpy as np
from shared.interface import DT

def residual_features(o0, v0, T, V, Cm, actions):
    """T,V (K,H,2) one-member rollout; Cm (K,H) contact mask -> (K,2,J=4)"""
    K, H, _ = T.shape
    Tseg = H * DT
    prev = np.concatenate([np.tile(o0, (K, 1))[:, None], T[:, :-1]], 1)
    dstep = T - prev
    f1 = T[:, -1] - o0
    f2 = V[:, -1] * Tseg
    f3 = (dstep * (~Cm)[..., None]).sum(1)
    f4 = np.tile(v0 * Tseg, (K, 1))
    return np.stack([f1, f2, f3, f4], -1)

class RLSAdapter:
    name = "rls_residual"
    def __init__(self, predictor, prior_std=5.0, noise_std=0.005, use_kinds=("probe", "task"), feat_idx=(2,)):
        self.pred = predictor; self.prior_std = prior_std; self.noise_std = noise_std
        self.use_kinds = tuple(use_kinds)
        self.feat_idx = list(feat_idx)
        self.J = len(self.feat_idx)

    def transition_features(self, tr):
        obs = tr.obs; A = np.asarray(tr.actions, float)[None]
        T, V, Cm = self.pred.rollout(obs, A)
        Tm, Vm, Cmm = T.mean(0), V.mean(0), Cm.mean(0) > 0.5
        F = residual_features(np.asarray(obs["obj"], float), np.asarray(obs["obj_vel"], float), Tm, Vm, Cmm, A)[0][:, self.feat_idx]
        r = np.asarray(tr.next_obs["obj"], float) - Tm[0, -1]
        return F, r

    def update(self, history):
        w = np.zeros(self.J); P = np.eye(self.J) * self.prior_std ** 2
        s2 = self.noise_std ** 2; n = 0; resid_before = []
        for tr in history:
            if tr.kind not in self.use_kinds or len(tr.actions) == 0:
                continue
            F, r = self.transition_features(tr)
            resid_before.append(float(np.linalg.norm(r - F @ w)))
            for d in range(2):                       # scalar RLS updates, one per coordinate
                f = F[d]
                Pf = P @ f; g = Pf / (s2 + f @ Pf)
                w = w + g * (r[d] - f @ w)
                P = P - np.outer(g, Pf)
            n += 1
        return {"kind": "rls_residual", "feat_idx": self.feat_idx, "w": w, "P": P, "sigma2": s2, "n": n,
                "innov_norms": resid_before}

class NoAdapter:
    """Baseline: ignores history (returns None -> predictor uses base model)."""
    name = "none"
    def update(self, history):
        return None
