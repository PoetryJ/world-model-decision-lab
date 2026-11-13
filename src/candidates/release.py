"""Timed-release fixed library, matched to A4 toy physics.

A4: while in contact the object follows the pusher (mu-independent); mu only acts
after release (slide ≈ v^2 / (2*mu*g)). Therefore a useful small library varies
CONTACT DURATION at a common speed and total horizon, then coasts.

All K segments have the same duration H and the same commanded speed (comparable
magnitude). Direction is the visible (goal - obj) unit vector. Seed unused.
Default H=6, A4's task-segment length.
"""
from __future__ import annotations

import numpy as np

from shared.interface import clip_action

from .utils import DEFAULT_K, NOMINAL_SPEED, as_actions, unit

DEFAULT_RELEASE_H = 6


class TimedReleaseProposer:
    def __init__(self, K: int = DEFAULT_K, H: int = DEFAULT_RELEASE_H, speed: float = NOMINAL_SPEED,
                 n_push: tuple[int, ...] | None = None):
        if H < 1:
            raise ValueError("H must be >= 1")
        self.K = int(K)
        self.H = int(H)
        self.speed = float(speed)
        if n_push is None:
            # Evenly spaced contact durations in 1..H, always including a full-horizon push.
            if self.K == 1:
                n_push = (self.H,)
            else:
                n_push = tuple(sorted({max(1, round(1 + i * (self.H - 1) / (self.K - 1))) for i in range(self.K)}))
                # If rounding collapsed the set, pad with remaining durations.
                n = 1
                while len(n_push) < self.K and n <= self.H:
                    if n not in n_push:
                        n_push = tuple(sorted(n_push + (n,)))
                    n += 1
        self.n_push = tuple(int(n) for n in n_push)[: self.K]
        if len(self.n_push) != self.K:
            raise ValueError(f"n_push has length {len(self.n_push)}, expected K={self.K}")
        self.names = tuple(f"push_{n}_of_{self.H}" for n in self.n_push)

    def propose(self, observation, history, seed: int) -> np.ndarray:
        del history, seed
        g = np.asarray(observation["goal"], dtype=np.float64)
        o = np.asarray(observation["obj"], dtype=np.float64)
        p = np.asarray(observation["pusher"], dtype=np.float64)
        u = unit(g - o, fallback=unit(o - p, fallback=np.array([1.0, 0.0])))
        v = clip_action(self.speed * u)
        acts = np.zeros((self.K, self.H, 2), dtype=np.float64)
        for k, n in enumerate(self.n_push):
            acts[k, :n] = v
        return as_actions(acts, k=self.K, h=self.H)
