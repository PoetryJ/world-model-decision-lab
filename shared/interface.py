"""FROZEN shared interface v1.1 (v1 + optional DecisionRecord.candidate_counts) (owner: controller). Do NOT edit from module agents.
Request changes via your report; controller decides and notifies consumers.

UNITS / FRAMES / TIMING (binding for all modules)
- World frame: 2D table, x right, y up, metres, origin at table centre, table = [-0.5,0.5]^2.
- Control step DT = 0.1 s. Physics substep = 0.01 s (env-internal).
- Action = commanded pusher velocity (vx, vy) in m/s, world frame, held constant for one control step.
  Legal: each component clipped to [-V_MAX, V_MAX], V_MAX = 0.5 m/s.
- Action segment ("chunk") = float64 array shape (H, 2), H control steps (H*DT seconds).
- Candidate set = float64 array shape (K, H, 2). All scorers compared on the SAME candidate array.
- Endpoint = object (x, y) position at the END of a segment, shape (2,); batched (K, 2).
- Randomness: every stochastic call takes an explicit seed (int) or np.random.Generator. No global RNG.
  Env stochasticity lives in env-owned Generator, which is part of the snapshot.
- Hidden physical params (e.g. friction mu) live ONLY in env.hidden. Policies/predictors/probers must
  never read env.hidden or evaluator internals. Only env and evaluation/ may access them.
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Any, Protocol
import numpy as np

DT = 0.1
SUBSTEP = 0.01
V_MAX = 0.5
TABLE = 0.5

# Observation: dict with these keys only (idealised structured pose perception, NOT vision)
#  "t": float seconds since reset
#  "pusher": (2,) pusher centre [m]
#  "obj": (2,) object centre [m]
#  "obj_vel": (2,) object velocity [m/s]
#  "goal": (2,) goal centre [m]
#  "goal_radius": float [m]
Observation = dict

@dataclass
class StepResult:
    obs: Observation
    done: bool            # task terminated (success or out of bounds or time limit)
    success: bool
    info: dict = field(default_factory=dict)   # must not contain hidden params

class Env(Protocol):
    hidden: dict          # e.g. {"mu": 0.3}; NEVER passed to policy-side code
    def reset(self, init_state: dict, hidden: dict, seed: int) -> Observation: ...
    def observe(self) -> Observation: ...
    def step(self, action: np.ndarray) -> StepResult: ...           # action shape (2,), one DT
    def step_segment(self, actions: np.ndarray) -> list[StepResult]: ...  # (H,2); stops early if done
    def snapshot(self) -> dict: ...   # FULL state: positions, velocities, contact/controller state, task state, time, RNG state, hidden
    def restore(self, snap: dict) -> None: ...

# History = list of Transition (only things that actually happened, in order)
@dataclass
class Transition:
    obs: Observation
    actions: np.ndarray       # (h,2) executed segment
    next_obs: Observation
    kind: str                 # "probe" | "task"

History = list  # list[Transition]

@dataclass
class Prediction:
    endpoints: np.ndarray                 # (K,2) mean predicted object endpoint
    samples: np.ndarray | None = None     # (K,S,2) optional samples / ensemble members
    traj: np.ndarray | None = None        # (K,H,2) optional per-step object positions
    extra: dict = field(default_factory=dict)

class Proposer(Protocol):
    def propose(self, observation: Observation, history: History, seed: int) -> np.ndarray: ...  # (K,H,2)

class Predictor(Protocol):
    def predict(self, observation: Observation, actions: np.ndarray, context: dict) -> Prediction: ...
    # context: {"history": History, "adapt_state": Any}; no hidden params

class Adapter(Protocol):
    def update(self, history: History) -> Any: ...   # returns adapt_state for predictor context; must be pure fn of history

class Scorer(Protocol):
    def score(self, predictions: Prediction, goal: dict) -> np.ndarray: ...  # (K,), higher = better
    # goal: {"goal": (2,), "goal_radius": float}

class Prober(Protocol):
    def choose_probe(self, observation: Observation, history: History, budget: dict) -> np.ndarray | None: ...
    # returns (h,2) probe segment or None (no probe). budget: {"steps_left": int, "probe_steps_left": int}

# ---- Episode record (one JSON line per decision) ----
@dataclass
class DecisionRecord:
    run_id: str
    episode_id: str
    method: str
    init_state_id: str
    hidden_condition_id: str      # label only; evaluator joins hidden values offline
    seed: int
    step_index: int
    kind: str                     # "probe" | "task"
    candidates: list | None       # (K,H,2) as nested list
    predictions: list | None      # (K,2)
    scores: list | None           # (K,)
    chosen_index: int | None
    executed: list                # (h,2)
    obs_before: dict
    obs_after: dict
    wall_time_s: dict             # {"propose":..,"predict":..,"update":..,"score":..,"probe":..}
    termination: str | None       # None | "success" | "timeout" | "out_of_bounds" | "budget"
    # v1.1 (additive, optional): {"requested":K,"returned":k,"failed":f,"seeds":[...]} actual candidate accounting
    candidate_counts: dict | None = None

def to_jsonable(x):
    if isinstance(x, np.ndarray): return x.tolist()
    if isinstance(x, (np.floating, np.integer)): return x.item()
    if isinstance(x, dict): return {k: to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)): return [to_jsonable(v) for v in x]
    return x

def clip_action(a: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(a, dtype=np.float64), -V_MAX, V_MAX)
