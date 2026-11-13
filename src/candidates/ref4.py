"""REF4 fixed library as specified by A10's protocol (visible geometry only).

u = (obj - pusher) / |obj - pusher| at the current observation.
Candidate k = speed_k * u, held for H steps (default H=5, 0.5 s).
Speeds = {0.2, 0.3, 0.4, 0.5} m/s. Seed is accepted but unused.
"""
from __future__ import annotations

import numpy as np

from shared.interface import clip_action

from .utils import as_actions, unit

REF4_SPEEDS = (0.2, 0.3, 0.4, 0.5)
REF4_H = 5


class Ref4Proposer:
    def __init__(self, speeds=REF4_SPEEDS, H: int = REF4_H):
        self.speeds = tuple(float(s) for s in speeds)
        self.K = len(self.speeds)
        self.H = int(H)

    def propose(self, observation, history, seed: int) -> np.ndarray:
        del history, seed
        u = unit(
            np.asarray(observation["obj"], dtype=np.float64)
            - np.asarray(observation["pusher"], dtype=np.float64),
            fallback=np.array([1.0, 0.0]),
        )
        acts = np.stack([np.tile(clip_action(s * u), (self.H, 1)) for s in self.speeds], axis=0)
        return as_actions(acts, k=self.K, h=self.H)
