"""Candidate proposer for A9.

Uses A5's frozen set when src/candidates/freeze.py exists; otherwise a deterministic
K=4 geometry-conditioned library (same duration, comparable amplitude, direction from
visible obj->goal). All scorers must be compared on the SAME candidate array.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
from shared.interface import clip_action

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_H = 6
DEFAULT_SPEEDS = (0.15, 0.25, 0.35, 0.45)  # m/s, legal (< V_MAX=0.5), same H
A5_FREEZE = ROOT / "src" / "candidates" / "freeze.py"


def default_task_candidates(observation: dict, H: int = DEFAULT_H, speeds=DEFAULT_SPEEDS) -> np.ndarray:
    """(K,H,2) push along visible obj->goal at each speed for H steps. No hidden params."""
    d = np.asarray(observation["goal"], dtype=np.float64) - np.asarray(observation["obj"], dtype=np.float64)
    n = float(np.linalg.norm(d))
    u = d / n if n > 1e-9 else np.array([1.0, 0.0])
    cands = [clip_action(np.tile(float(s) * u, (H, 1))) for s in speeds]
    return np.stack(cands, axis=0).astype(np.float64)


def load_shared_candidates(observation: dict, seed: int = 0, source: str = "auto") -> tuple[np.ndarray, str]:
    """Return (candidates, source_tag).

    source:
      "auto" | "a5"  — A5 FixedLibraryProposer (K=4, H=10, nominal 0.4 m/s);
                       this is the shared library the world-model route should use.
      "speedgrid"    — A9 diagnostic set: 4 speeds, same H=6, along obj->goal
                       (amplitude varies so hidden friction can flip the best index).
      "default"      — alias of speedgrid.
    freeze.py is A5's save/load helper, not a proposer; we call FixedLibraryProposer
    so every method gets the same geometry-conditioned set at decision time.
    """
    src = (source or "auto").lower()
    if src in ("speedgrid", "default"):
        return default_task_candidates(observation), "a9_default_k4_speeds"
    if src in ("strikegrid", "strike"):
        from src.env.scenarios import strike_grid
        arr, _labels = strike_grid(observation)
        return clip_action(np.asarray(arr, dtype=np.float64)), "a4_strike_grid_K54_H6"
    if src in ("strike4",):
        from src.env.scenarios import strike_grid
        arr, labels = strike_grid(observation)
        want = {(0.2, 6), (0.3, 4), (0.4, 3), (0.5, 2)}  # A10 PROTOCOL section 3 E1 subset
        keep = [i for i, lab in enumerate(labels) if (float(lab[0]), int(lab[1])) in want]
        if len(keep) != 4:
            raise RuntimeError(f"STRIKE4 expected 4 cands, got {len(keep)} from labels={labels}")
        return clip_action(np.asarray(arr, dtype=np.float64)[keep]), "a10_STRIKE4_H6"
    # A5 official library
    try:
        from src.candidates.fixed import FixedLibraryProposer
        prop = FixedLibraryProposer()  # K=4, H=DEFAULT_H=10, speed=NOMINAL_SPEED=0.4
        arr = np.asarray(prop.propose(observation, [], int(seed)), dtype=np.float64)
        return clip_action(arr), "a5_FixedLibraryProposer_K4_H10_v0.4"
    except Exception as exc:
        # fall back, but record why
        cands = default_task_candidates(observation)
        return cands, f"a9_default_k4_speeds_fallback:{type(exc).__name__}:{exc}"


class DirectProposer:
    """Proposer protocol: propose(observation, history, seed) -> (K,H,2).

    Seed is accepted but the default library is a deterministic function of the
    observation so every scorer sees the same set. If A5 freeze.py is present,
    that source is used (still should be the shared frozen set).
    """

    def __init__(self, source: str = "auto"):
        self.source = source

    def propose(self, observation, history, seed: int) -> np.ndarray:
        cands, tag = load_shared_candidates(observation, seed, source=self.source)
        self._last_tag = tag
        return cands
