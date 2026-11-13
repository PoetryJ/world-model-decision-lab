"""Grid posterior over friction mu from VISIBLE history only (Transition obs/actions/next_obs).
Likelihood: Gaussian on next object position & velocity predicted by BatchPushModel under each mu.
Classic: Bayesian system identification on a discrete hypothesis grid.
"""
from __future__ import annotations
import numpy as np
from .physics_model import BatchPushModel

DEFAULT_GRID = np.round(np.geomspace(0.05, 1.0, 25), 4)

class MuGridBelief:
    def __init__(self, grid=DEFAULT_GRID, model: BatchPushModel | None = None,
                 sigma_pos: float = 0.003, sigma_vel: float = 0.02, temper: float = 1.0):
        self.grid = np.asarray(grid, float)
        self.model = model or BatchPushModel()
        self.sp, self.sv, self.temper = sigma_pos, sigma_vel, temper

    def prior(self):
        return np.full(len(self.grid), 1.0 / len(self.grid))

    def loglik_transition(self, obs, actions, next_obs, pv0=None):
        M = len(self.grid)
        a = np.asarray(actions, float)
        if a.ndim != 2 or len(a) == 0:
            return np.zeros(M)
        r = self.model.rollout(np.tile(obs["pusher"], (M, 1)), np.tile(obs["obj"], (M, 1)),
                               np.tile(obs["obj_vel"], (M, 1)), self.grid, np.tile(a, (M, 1, 1)),
                               pv=pv0)
        ep = np.sum((r["o"] - np.asarray(next_obs["obj"])) ** 2, axis=1) / self.sp ** 2
        ev = np.sum((r["ov"] - np.asarray(next_obs["obj_vel"])) ** 2, axis=1) / self.sv ** 2
        return -0.5 * self.temper * (ep + ev)

    def loglik_obs(self, pred_o, pred_ov, obs_o, obs_ov):
        """pred_*: (M,2) under each grid mu; obs_*: (N,2) hypothetical outcomes -> (N,M) loglik."""
        ep = np.sum((obs_o[:, None, :] - pred_o[None]) ** 2, axis=2) / self.sp ** 2
        ev = np.sum((obs_ov[:, None, :] - pred_ov[None]) ** 2, axis=2) / self.sv ** 2
        return -0.5 * self.temper * (ep + ev)

    def posterior(self, history):
        lp = np.log(self.prior())
        pv0 = None
        for tr in history:
            lp = lp + self.loglik_transition(tr.obs, tr.actions, tr.next_obs, pv0)
            a = np.asarray(tr.actions, float)
            pv0 = np.clip(a[-1], -0.5, 0.5) if len(a) else pv0  # actuator state ~ last command (visible)
        lp -= lp.max()
        w = np.exp(lp)
        return w / w.sum()

def last_command(history):
    """Agent-side estimate of actuator state: last commanded velocity (visible), else zero."""
    for tr in reversed(history):
        a = np.asarray(tr.actions, float)
        if len(a):
            return np.clip(a[-1], -0.5, 0.5)
    return np.zeros(2)

def entropy(w):
    w = np.asarray(w, float)
    nz = w > 0
    return float(-np.sum(w[nz] * np.log(w[nz])))
