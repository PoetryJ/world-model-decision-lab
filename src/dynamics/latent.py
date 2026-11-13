"""A6 EXTENSION (after main line): environment-latent predictor (TOY, CPU, numpy).
Same one-step MLP ensemble as model.py but with one extra input z = log(mu). During TRAINING z is the training
condition's mu label (env-side configuration of the TRAINING conditions only -- a supervised system-ID style label;
the RLS main line needs no such label). At test time z is NEVER read from the env: LatentGridAdapter infers it from
the episode's own history by grid search (max likelihood of observed segment endpoints), prior = mean train z.
API:
    from src.dynamics.latent import load_or_train_latent, LatentGridAdapter
    lp = load_or_train_latent(seed=0); la = LatentGridAdapter(lp)
    st = la.update(history)   # {"kind":"latent","z":..., "mu_hat":..., "grid_sse":[...]}
    P = lp.predict(obs, C, {"history": history, "adapt_state": st})
"""
from __future__ import annotations
import os, json, time, pickle
import numpy as np
from shared.interface import DT, Prediction, clip_action
from src.dynamics.model import (DEFAULT_CFG, MODEL_DIR, collect, train_mlp, _fwd, default_env_factory,
                                mixed_init_sampler)

class LatentPredictor:
    name = "a6_latent_mlp_ensemble"
    def __init__(self, members, xm, xs, ym, ys, meta):
        self.members, self.xm, self.xs, self.ym, self.ys, self.meta = members, xm, xs, ym, ys, meta
        self.z_prior = float(np.mean(np.log(meta["cfg"]["train_mus"])))

    def rollout(self, obs, actions, z):
        A = clip_action(np.asarray(actions, float)); K, H, _ = A.shape; M = len(self.members)
        T = np.zeros((M, K, H, 2))
        zc = np.full((K, 1), z)
        for j in range(M):
            p = np.tile(np.asarray(obs["pusher"], float), (K, 1)); o = np.tile(np.asarray(obs["obj"], float), (K, 1))
            ov = np.tile(np.asarray(obs["obj_vel"], float), (K, 1))
            for t in range(H):
                x = (np.concatenate([o - p, ov, A[:, t], zc], -1) - self.xm) / self.xs
                y = _fwd(self.members[j], x)[0] * self.ys + self.ym
                p = p + A[:, t] * DT; o = o + y[:, :2]; ov = y[:, 2:]
                T[j, :, t] = o
        return T

    def predict(self, observation, actions, context=None):
        st = (context or {}).get("adapt_state")
        z = st["z"] if st is not None and st.get("kind") == "latent" else self.z_prior
        T = self.rollout(observation, actions, z)
        samples = T[:, :, -1].transpose(1, 0, 2)
        return Prediction(endpoints=samples.mean(1), samples=samples, traj=T.mean(0), extra={"z": z, "model": self.name})

class LatentGridAdapter:
    name = "latent_grid"
    def __init__(self, lp, grid=None, noise_std=0.005, prior_std_z=1.0):
        self.lp = lp; self.grid = np.log(np.geomspace(0.03, 1.5, 40)) if grid is None else grid
        self.noise_std = noise_std; self.prior_std_z = prior_std_z
    def update(self, history):
        nll = 0.5 * ((self.grid - self.lp.z_prior) / self.prior_std_z) ** 2
        for tr in history:
            if len(tr.actions) == 0: continue
            for i, z in enumerate(self.grid):
                e = self.lp.rollout(tr.obs, np.asarray(tr.actions, float)[None], z)[:, 0, -1].mean(0)
                nll[i] += 0.5 * np.sum((e - np.asarray(tr.next_obs["obj"])) ** 2) / self.noise_std ** 2
        i = int(np.argmin(nll))
        return {"kind": "latent", "z": float(self.grid[i]), "mu_hat": float(np.exp(self.grid[i])), "n": len(history),
                "nll": nll.tolist()}

def load_or_train_latent(seed=0, cfg=None, env_factory=None, retrain=False):
    cfg = dict(DEFAULT_CFG, **(cfg or {})); cfg["M"] = 3
    path = os.path.join(MODEL_DIR, f"latent_{cfg['env']}_{cfg['version']}_s{seed}.pkl")
    if os.path.exists(path) and not retrain:
        with open(path, "rb") as f: return LatentPredictor(**pickle.load(f))
    env_factory = env_factory or default_env_factory(cfg["env"])
    t0 = time.time()
    X, Y, C = collect(env_factory, [{"mu": m} for m in cfg["train_mus"]], cfg["n_train_eps"], seed=seed * 1000 + 1,
                      init_sampler=mixed_init_sampler("train"))     # same data as model.py (same seed)
    X = np.c_[X, np.log(np.asarray(cfg["train_mus"]))[C]]
    xm, xs = X.mean(0), X.std(0) + 1e-6; ym, ys = Y.mean(0), Y.std(0) + 1e-6
    Xn, Yn = (X - xm) / xs, (Y - ym) / ys
    rng = np.random.default_rng(seed); members = []; log = []
    for j in range(cfg["M"]):
        bi = rng.integers(0, len(Xn), len(Xn))
        members.append(train_mlp(Xn[bi], Yn[bi], seed=seed * 100 + 50 + j, hidden=cfg["hidden"], epochs=cfg["epochs"], log=log))
    meta = {"cfg": cfg, "seed": seed, "n_train_steps": int(len(X)), "train_s": time.time() - t0, "log": log,
            "uses_train_condition_labels": True}
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump({"members": members, "xm": xm, "xs": xs, "ym": ym, "ys": ys, "meta": meta}, f)
    return LatentPredictor(members, xm, xs, ym, ys, meta)
