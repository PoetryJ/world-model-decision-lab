"""Local stub predictor for A7 until / alongside A6.

WrongMuStubPredictor: fixture-identical physics with a FIXED assumed mu
(not the env's hidden mu) plus optional mu-perturbation ensemble and/or
additive endpoint noise. Never reads env.hidden or evaluator internals.

This is an intentionally misspecified world model, used to study scoring,
overconfidence, and gating — not a proposed deployed predictor.
"""
from __future__ import annotations
import numpy as np
from shared.interface import Prediction
from src.scoring.physics_stub import rollout_segment


class WrongMuStubPredictor:
    """Predictor protocol. context may contain history/adapt_state; ignored here
    (no adaptation in the stub). seed comes from context['seed'] if present,
    else self.seed, so calls are reproducible.
    """
    name = "a7_wrong_mu_stub"

    def __init__(
        self,
        assumed_mu: float = 0.3,
        n_samples: int = 9,
        mu_perturb: float = 0.08,
        mu_perturb_mode: str = "normal",  # "normal" | "fixed_grid" | "none"
        endpoint_noise: float = 0.0,
        seed: int = 0,
        stop_on_done: bool = True,
        max_time: float = 6.0,
        mu_clip: tuple[float, float] = (0.02, 1.5),
    ):
        if n_samples < 1:
            raise ValueError("n_samples must be >= 1")
        self.assumed_mu = float(assumed_mu)
        self.n_samples = int(n_samples)
        self.mu_perturb = float(mu_perturb)
        self.mu_perturb_mode = mu_perturb_mode
        self.endpoint_noise = float(endpoint_noise)
        self.seed = int(seed)
        self.stop_on_done = bool(stop_on_done)
        self.max_time = float(max_time)
        self.mu_clip = (float(mu_clip[0]), float(mu_clip[1]))

    def sample_mus(self, rng: np.random.Generator) -> np.ndarray:
        S = self.n_samples
        lo, hi = self.mu_clip
        if self.mu_perturb_mode == "none" or self.mu_perturb <= 0:
            mus = np.full(S, self.assumed_mu, dtype=np.float64)
        elif self.mu_perturb_mode == "fixed_grid":
            # Symmetric grid around assumed_mu (includes the centre if S odd).
            if S == 1:
                mus = np.array([self.assumed_mu], dtype=np.float64)
            else:
                deltas = np.linspace(-self.mu_perturb, self.mu_perturb, S)
                mus = self.assumed_mu + deltas
        elif self.mu_perturb_mode == "normal":
            mus = self.assumed_mu + rng.normal(0.0, self.mu_perturb, size=S)
        else:
            raise ValueError(self.mu_perturb_mode)
        return np.clip(mus, lo, hi)

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
        pusher = np.asarray(observation["pusher"], dtype=np.float64)
        obj = np.asarray(observation["obj"], dtype=np.float64)
        obj_vel = np.asarray(observation["obj_vel"], dtype=np.float64)
        goal = np.asarray(observation["goal"], dtype=np.float64)
        goal_radius = float(observation["goal_radius"])
        t0 = float(observation.get("t", 0.0))

        samples = np.zeros((K, self.n_samples, 2), dtype=np.float64)
        trajs = np.zeros((self.n_samples, K, H, 2), dtype=np.float64)
        succ = np.zeros((K, self.n_samples), dtype=bool)
        used_mu = np.zeros((K, self.n_samples), dtype=np.float64)
        for s, mu in enumerate(mus):
            for k in range(K):
                out = rollout_segment(
                    pusher, obj, obj_vel, actions[k], mu=float(mu),
                    t0=t0, goal=goal, goal_radius=goal_radius,
                    max_time=self.max_time, stop_on_done=self.stop_on_done,
                )
                ep = out["endpoint"]
                if self.endpoint_noise > 0:
                    ep = ep + rng.normal(0.0, self.endpoint_noise, size=2)
                samples[k, s] = ep
                trajs[s, k] = out["traj"]
                succ[k, s] = bool(out["success"])
                used_mu[k, s] = float(mu)
        endpoints = samples.mean(axis=1)
        traj = trajs.mean(axis=0)
        extra = {
            "model": self.name,
            "assumed_mu": self.assumed_mu,
            "mus": mus.tolist(),
            "sample_success": succ,
            "mu_perturb": self.mu_perturb,
            "mu_perturb_mode": self.mu_perturb_mode,
            "endpoint_noise": self.endpoint_noise,
            "seed": seed,
        }
        return Prediction(endpoints=endpoints, samples=samples, traj=traj, extra=extra)
