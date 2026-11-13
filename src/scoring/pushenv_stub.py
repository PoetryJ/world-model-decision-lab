"""A4 PushEnv-backed stub predictor (local, A7 exclusive).

Rolls a *private* PushEnv with a caller-supplied assumed mu (or a mu ensemble).
Never reads the real env.hidden. After the action segment, settles with zero
actions (same rule as A4 branch_eval, but implemented here so the predictor
does not import evaluation-only helpers).

This is a misspecified / hypothesis-set world model for scoring studies, not
a deployed dynamics model. Endpoints are resting object positions after settle.
"""
from __future__ import annotations
import numpy as np
from shared.interface import DT, Prediction
from src.env.push_env import PushEnv, REST_SPEED

SETTLE_MAX = 30  # matches src/env/branch_eval.py


class PushEnvStubPredictor:
    """Predictor protocol. assumed_mu or explicit mu_list (hypothesis set).

    mu_list, if given, is a fixed prior over hidden friction — it must not be
    chosen from the current episode's true mu. Using the experiment's known
    pair {0.1, 0.5} is labelled 'hypothesis-set (lo/hi prior)' in the study,
    not an oracle for the current condition.
    """
    name = "a7_pushenv_stub"

    def __init__(
        self,
        assumed_mu: float = 0.25,
        n_samples: int = 1,
        mu_list: list[float] | tuple[float, ...] | None = None,
        mu_perturb: float = 0.0,
        mu_perturb_mode: str = "none",
        seed: int = 0,
        settle_max: int = SETTLE_MAX,
        mu_clip: tuple[float, float] = (0.02, 1.5),
        action_noise: float = 0.0,
        max_time: float = 6.0,
    ):
        self.assumed_mu = float(assumed_mu)
        self.mu_list = None if mu_list is None else [float(m) for m in mu_list]
        if self.mu_list is not None:
            self.n_samples = len(self.mu_list)
        else:
            self.n_samples = int(n_samples)
        self.mu_perturb = float(mu_perturb)
        self.mu_perturb_mode = mu_perturb_mode
        self.seed = int(seed)
        self.settle_max = int(settle_max)
        self.mu_clip = (float(mu_clip[0]), float(mu_clip[1]))
        self.action_noise = float(action_noise)
        self.max_time = float(max_time)

    def sample_mus(self, rng: np.random.Generator) -> np.ndarray:
        if self.mu_list is not None:
            return np.asarray(self.mu_list, dtype=np.float64)
        S = self.n_samples
        lo, hi = self.mu_clip
        if self.mu_perturb_mode == "none" or self.mu_perturb <= 0 or S == 1:
            return np.full(S, self.assumed_mu, dtype=np.float64)
        if self.mu_perturb_mode == "fixed_grid":
            if S == 1:
                mus = np.array([self.assumed_mu], dtype=np.float64)
            else:
                mus = self.assumed_mu + np.linspace(-self.mu_perturb, self.mu_perturb, S)
            return np.clip(mus, lo, hi)
        if self.mu_perturb_mode == "normal":
            return np.clip(self.assumed_mu + rng.normal(0.0, self.mu_perturb, size=S), lo, hi)
        raise ValueError(self.mu_perturb_mode)

    def _rollout_one(self, observation: dict, actions: np.ndarray, mu: float, seed: int):
        env = PushEnv(action_noise=self.action_noise, obs_noise=0.0, max_time=self.max_time)
        init = {
            "pusher": np.asarray(observation["pusher"], dtype=np.float64),
            "obj": np.asarray(observation["obj"], dtype=np.float64),
            "goal": np.asarray(observation["goal"], dtype=np.float64),
            "goal_radius": float(observation["goal_radius"]),
        }
        env.reset(init, {"mu": float(mu)}, seed=int(seed))
        # Mid-episode: patch visible state. Actuator velocity is NOT in the
        # observation (interface v1); left at the reset value 0.
        t0 = float(observation.get("t", 0.0))
        ov = np.asarray(observation["obj_vel"], dtype=np.float64)
        if abs(t0) > 1e-12 or np.linalg.norm(ov) > 1e-12:
            env.p = np.asarray(observation["pusher"], dtype=np.float64).copy()
            env.o = np.asarray(observation["obj"], dtype=np.float64).copy()
            env.ov = ov.copy()
            env.t = t0
            env.n_steps = int(round(t0 / DT))
            env._last_obs = env._make_obs()
        env.step_segment(np.asarray(actions, dtype=np.float64))
        k = 0
        while (not env.done) and k < self.settle_max and np.linalg.norm(env.ov) >= REST_SPEED:
            env.step(np.zeros(2))
            k += 1
        return env.o.copy(), bool(env.success), k

    def predict(self, observation: dict, actions: np.ndarray, context: dict | None = None) -> Prediction:
        context = context or {}
        actions = np.asarray(actions, dtype=np.float64)
        if actions.ndim == 2:
            actions = actions[None, ...]
        if actions.ndim != 3 or actions.shape[-1] != 2:
            raise ValueError(f"actions must be (K,H,2) or (H,2), got {actions.shape}")
        K, H, _ = actions.shape
        seed = int(context.get("seed", self.seed))
        rng = np.random.default_rng(seed)
        mus = self.sample_mus(rng)
        samples = np.zeros((K, len(mus), 2), dtype=np.float64)
        succ = np.zeros((K, len(mus)), dtype=bool)
        settle = np.zeros((K, len(mus)), dtype=np.int32)
        for s, mu in enumerate(mus):
            for k in range(K):
                ep, ok, ns = self._rollout_one(observation, actions[k], float(mu), seed + s * 1009 + k)
                samples[k, s] = ep
                succ[k, s] = ok
                settle[k, s] = ns
        endpoints = samples.mean(axis=1)
        extra = {
            "model": self.name,
            "assumed_mu": self.assumed_mu,
            "mus": mus.tolist(),
            "mu_list": self.mu_list,
            "sample_success": succ,
            "settle_steps": settle,
            "settle_max": self.settle_max,
            "seed": seed,
        }
        return Prediction(endpoints=endpoints, samples=samples, traj=None, extra=extra)
