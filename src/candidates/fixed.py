"""Fixed candidate library (Proposer). Strategy set is fixed; directions use visible geometry only.

K primitives of common duration H and (for the default E1 subset) common speed NOMINAL_SPEED,
so magnitude and duration are comparable across the set. Seed is accepted but unused
(deterministic given observation, H, K).
"""
from __future__ import annotations

import numpy as np

from shared.interface import V_MAX

from .utils import (
    DEFAULT_H,
    DEFAULT_K,
    NOMINAL_SPEED,
    as_actions,
    geometry,
    go_to_then_push,
)


PRIMITIVE_NAMES_8 = (
    "direct",
    "offset_plus",
    "offset_minus",
    "angled",
    "direct_fast",
    "direct_slow",
    "side_then_push_plus",
    "side_then_push_minus",
)

# E1 uses the first 4, all at NOMINAL_SPEED (comparable magnitude).
E1_NAMES = PRIMITIVE_NAMES_8[:4]


class FixedLibraryProposer:
    """propose(obs, history, seed) -> (K,H,2) float64, legal, world-frame vx,vy [m/s]."""

    def __init__(self, K: int = DEFAULT_K, H: int = DEFAULT_H, speed: float = NOMINAL_SPEED):
        if K < 1 or K > 8:
            raise ValueError("FixedLibraryProposer supports K in 1..8")
        if H < 1:
            raise ValueError("H must be >= 1")
        self.K = int(K)
        self.H = int(H)
        self.speed = float(speed)
        self.names = PRIMITIVE_NAMES_8[: self.K]

    def propose(self, observation, history, seed: int) -> np.ndarray:
        del history, seed  # deterministic library; seed kept for protocol
        geo = geometry(observation)
        p, gd, tg = geo["pusher"], geo["goal_dir"], geo["tangent"]
        o = geo["obj"]
        H, sp = self.H, self.speed
        # Waypoints relative to the object / goal, using only visible pose.
        behind = geo["behind"]
        behind_plus = o - gd * 0.10 + tg * 0.06
        behind_minus = o - gd * 0.10 - tg * 0.06
        behind_angle = o - gd * 0.10 + tg * 0.03
        side_plus = o + tg * 0.12 - gd * 0.04
        side_minus = o - tg * 0.12 - gd * 0.04

        prims = {
            "direct": lambda: go_to_then_push(p, behind, gd, H, sp),
            "offset_plus": lambda: go_to_then_push(p, behind_plus, gd + 0.25 * tg, H, sp),
            "offset_minus": lambda: go_to_then_push(p, behind_minus, gd - 0.25 * tg, H, sp),
            "angled": lambda: go_to_then_push(p, behind_angle, gd, H, sp),
            "direct_fast": lambda: go_to_then_push(p, behind, gd, H, V_MAX),
            "direct_slow": lambda: go_to_then_push(p, behind, gd, H, 0.25),
            "side_then_push_plus": lambda: go_to_then_push(p, side_plus, gd, H, sp),
            "side_then_push_minus": lambda: go_to_then_push(p, side_minus, gd, H, sp),
        }
        acts = np.stack([prims[name]() for name in self.names], axis=0)
        return as_actions(acts, k=self.K, h=self.H)
