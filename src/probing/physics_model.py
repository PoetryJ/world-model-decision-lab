"""Policy-side HYPOTHESIS model of planar pushing, parametrised by friction mu.
This is NOT the env: it is the agent's assumed model class (idealised: same functional form as the
fixture/A4 contact model; radii/g are agent-side constants, mu is unknown -> belief).
Never reads env.hidden. Vectorised over a batch B of (state, mu, action-sequence) triples.
Units per shared/interface.py: metres, m/s, DT=0.1 s control step, SUBSTEP=0.01 s.
"""
from __future__ import annotations
import numpy as np
from shared.interface import DT, SUBSTEP, TABLE, V_MAX

G = 9.81

class BatchPushModel:
    def __init__(self, r_p: float = 0.03, r_o: float = 0.05, g: float = G, tau_act: float = 0.02):
        # tau_act: assumed first-order actuator lag (A4 PushEnv documents tau=0.02 s; fixture has none).
        self.r_p, self.r_o, self.g, self.tau = r_p, r_o, g, tau_act
        self.nsub = int(round(DT / SUBSTEP))

    def rollout(self, p, o, ov, mu, actions, pv=None):
        """p,o,ov: (B,2); mu: (B,); actions: (B,H,2). Returns dict with final p,o,ov (B,2),
        obj traj (B,H,2) at end of each control step, oob (B,) bool."""
        p = np.array(p, float); o = np.array(o, float); ov = np.array(ov, float)
        mu = np.asarray(mu, float); acts = np.clip(np.asarray(actions, float), -V_MAX, V_MAX)
        B, H, _ = acts.shape
        traj = np.empty((B, H, 2)); oob = np.zeros(B, bool)
        rr = self.r_p + self.r_o
        pv = np.zeros_like(p) if pv is None else np.array(np.broadcast_to(pv, p.shape), float)
        alpha = 1.0 - np.exp(-SUBSTEP / self.tau) if self.tau > 0 else 1.0
        lim = TABLE - self.r_p
        for h in range(H):
            cmd = acts[:, h, :]
            for _ in range(self.nsub):
                pv = pv + (cmd - pv) * alpha
                newp = np.clip(p + pv * SUBSTEP, -lim, lim)
                v = (newp - p) / SUBSTEP
                p = newp
                d = o - p
                dist = np.linalg.norm(d, axis=1)
                c = (dist < rr) & (dist > 1e-9)
                if c.any():
                    n = d[c] / dist[c, None]
                    o[c] = p[c] + n * rr
                    vn = np.einsum("ij,ij->i", v[c], n)
                    von = np.einsum("ij,ij->i", ov[c], n)
                    up = vn > von
                    ovc = ov[c]
                    ovc[up] = ovc[up] + (vn[up] - von[up])[:, None] * n[up]
                    ov[c] = ovc
                sp = np.linalg.norm(ov, axis=1)
                mv = sp > 0
                if mv.any():
                    dec = mu[mv] * self.g * SUBSTEP
                    ov[mv] = ov[mv] * (np.maximum(0.0, sp[mv] - dec) / sp[mv])[:, None]
                o = o + ov * SUBSTEP
            traj[:, h] = o
            oob |= np.any(np.abs(o) > TABLE, axis=1)
        return {"p": p, "o": o, "ov": ov, "pv": pv, "traj": traj, "oob": oob}

    def settle(self, o, ov, mu):
        """Analytic rest position if the pusher stops and does not touch the object again."""
        sp = np.linalg.norm(ov, axis=1)
        dist = sp ** 2 / (2 * np.asarray(mu, float) * self.g)
        dirn = np.where(sp[:, None] > 1e-12, ov / np.maximum(sp, 1e-12)[:, None], 0.0)
        o_rest = o + dirn * dist[:, None]
        t_rest = sp / (np.asarray(mu, float) * self.g)
        return o_rest, t_rest
