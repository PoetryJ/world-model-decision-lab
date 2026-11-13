"""Seeded random candidate proposer. All K segments share duration H and target RMS speed.

Each candidate is a 2-piece constant-velocity path: a random approach heading for
H_approach steps, then a random (or goal-biased) push heading for the rest, both at
the same commanded speed (default NOMINAL_SPEED). This keeps magnitude/duration
comparable with the fixed library while spreading directions.
"""
from __future__ import annotations

import numpy as np

from shared.interface import V_MAX, clip_action

from .utils import DEFAULT_H, DEFAULT_K, NOMINAL_SPEED, as_actions, geometry, unit


class RandomProposer:
    def __init__(
        self,
        K: int = DEFAULT_K,
        H: int = DEFAULT_H,
        speed: float = NOMINAL_SPEED,
        goal_bias: float = 0.5,
    ):
        if K < 1:
            raise ValueError("K must be >= 1")
        if H < 1:
            raise ValueError("H must be >= 1")
        self.K = int(K)
        self.H = int(H)
        self.speed = float(np.clip(speed, 0.0, V_MAX))
        self.goal_bias = float(goal_bias)

    def propose(self, observation, history, seed: int) -> np.ndarray:
        del history
        rng = np.random.default_rng(int(seed))
        geo = geometry(observation)
        gd = geo["goal_dir"]
        H, K, sp = self.H, self.K, self.speed
        # Split so both pieces are nonempty when H>=2.
        h1 = max(1, H // 3)
        acts = np.zeros((K, H, 2), dtype=np.float64)
        for k in range(K):
            # Evenly spaced base headings plus a small jitter so candidates are not clones.
            base = 2.0 * np.pi * (k / K) + float(rng.uniform(-0.15, 0.15))
            approach = np.array([np.cos(base), np.sin(base)])
            if rng.random() < self.goal_bias:
                jitter = rng.normal(0.0, 0.35, size=2)
                push = unit(gd + jitter, fallback=gd)
            else:
                ang = float(rng.uniform(0.0, 2.0 * np.pi))
                push = np.array([np.cos(ang), np.sin(ang)])
            acts[k, :h1] = clip_action(sp * approach)
            acts[k, h1:] = clip_action(sp * push)
        return as_actions(acts, k=K, h=H)
