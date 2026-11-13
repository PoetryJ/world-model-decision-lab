"""A6 learned dynamics predictor (TOY, CPU, numpy only).

Model: ensemble of M small MLPs, one-step transition  f(rel, obj_vel, action) -> (d_obj, obj_vel_next)
  rel = obj - pusher [m], obj_vel [m/s], action [m/s] (clipped to V_MAX). DT = 0.1 s per step.
Pusher is kinematic: pusher_{t+1} = pusher_t + clip(a_t)*DT (known robot command, no learning).
Rollout over a segment (K,H,2) gives per-step object positions; endpoint = object pos after H steps.
Training data: per-step transitions collected ONLY on training hidden conditions (env side collects;
predictor never sees hidden params). Ensemble members differ by init seed + bootstrap resample.

API (for A7/A8/runner):
    from src.dynamics.model import load_or_train
    pred = load_or_train(seed=0)                     # cached in runs/a6/models/
    P = pred.predict(obs, actions_KH2, {"history": hist, "adapt_state": st})
    P.endpoints (K,2) mean endpoint; P.samples (K,M,2) per-member endpoints; P.traj (K,H,2) mean traj
    P.extra["base_endpoints"] (K,2) before residual; P.extra["residual_std"] (K,) RLS predictive std (if adapted)
"""
from __future__ import annotations
import os, json, time, pickle
import numpy as np
from shared.interface import DT, V_MAX, TABLE, Prediction, clip_action

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_DIR = os.path.join(ROOT, "runs", "a6", "models")
CONTACT_DIST = 0.08 + 0.005   # pusher r + object r (+margin); known robot/object geometry, used only for residual features


# ---------------- tiny numpy MLP -----------------
def _init(rng, sizes):
    ps = []
    for i, o in zip(sizes[:-1], sizes[1:]):
        ps.append([rng.normal(0, np.sqrt(2.0 / i), (i, o)), np.zeros(o)])
    return ps

def _fwd(ps, x):
    hs = [x]
    for j, (W, b) in enumerate(ps):
        x = x @ W + b
        if j < len(ps) - 1:
            x = np.tanh(x)
        hs.append(x)
    return x, hs

def _bwd(ps, hs, g):
    grads = [None] * len(ps)
    for j in range(len(ps) - 1, -1, -1):
        W, b = ps[j]
        grads[j] = [hs[j].T @ g, g.sum(0)]
        if j > 0:
            g = (g @ W.T) * (1 - hs[j] ** 2)
    return grads

def train_mlp(X, Y, seed, hidden=64, epochs=60, bs=256, lr=3e-3, wd=1e-5, Xv=None, Yv=None, log=None):
    rng = np.random.default_rng(seed)
    ps = _init(rng, [X.shape[1], hidden, hidden, Y.shape[1]])
    m = [[np.zeros_like(W), np.zeros_like(b)] for W, b in ps]
    v = [[np.zeros_like(W), np.zeros_like(b)] for W, b in ps]
    t = 0; b1, b2 = 0.9, 0.999
    n = len(X)
    for ep in range(epochs):
        idx = rng.permutation(n)
        lr_e = lr * (0.5 * (1 + np.cos(np.pi * ep / epochs)))
        for s in range(0, n, bs):
            bi = idx[s:s + bs]
            out, hs = _fwd(ps, X[bi])
            g = 2 * (out - Y[bi]) / len(bi)
            grads = _bwd(ps, hs, g)
            t += 1
            for j in range(len(ps)):
                for k in range(2):
                    gr = grads[j][k] + (wd * ps[j][k] if k == 0 else 0)
                    m[j][k] = b1 * m[j][k] + (1 - b1) * gr
                    v[j][k] = b2 * v[j][k] + (1 - b2) * gr * gr
                    mh = m[j][k] / (1 - b1 ** t); vh = v[j][k] / (1 - b2 ** t)
                    ps[j][k] -= lr_e * mh / (np.sqrt(vh) + 1e-8)
        if log is not None and (ep % 10 == 9 or ep == epochs - 1):
            tr = float(np.mean((_fwd(ps, X[:5000])[0] - Y[:5000]) ** 2))
            va = float(np.mean((_fwd(ps, Xv)[0] - Yv) ** 2)) if Xv is not None else None
            log.append({"seed": seed, "epoch": ep + 1, "train_mse_norm": tr, "val_mse_norm": va})
    return ps


# ---------------- data collection (env side; hidden used only to configure env) -----------------
def sample_init(rng):
    """Random push scene. Object somewhere on table, pusher 0.09-0.2 m from it, goal elsewhere."""
    obj = rng.uniform(-0.3, 0.3, 2)
    ang = rng.uniform(-np.pi, np.pi); r = rng.uniform(0.09, 0.2)
    pusher = obj + r * np.array([np.cos(ang), np.sin(ang)])
    goal = rng.uniform(-0.35, 0.35, 2)
    return {"pusher": pusher.tolist(), "obj": obj.tolist(), "goal": goal.tolist(), "goal_radius": 0.05}

def _behaviour_action(obs, rng, mode_state):
    """Mixture: push toward object (with angle/speed noise), stop (let slide), or uniform random."""
    mode = mode_state["mode"]
    if mode == "push":
        d = obs["obj"] - obs["pusher"]; d = d / (np.linalg.norm(d) + 1e-9)
        th = mode_state["ang"]; c, s = np.cos(th), np.sin(th)
        d = np.array([c * d[0] - s * d[1], s * d[0] + c * d[1]])
        return clip_action(mode_state["speed"] * d + rng.normal(0, 0.03, 2))
    if mode == "stop":
        return np.zeros(2)
    return rng.uniform(-V_MAX, V_MAX, 2)

def mixed_init_sampler(split):
    """Half generic random scenes, half A10 frozen init states of `split` ('train' or 'val', pos+neg), loaded via
    evaluation.audit.load_frozen. Never the test split. Visible init only."""
    assert split in ("train", "val")
    from evaluation.audit import load_frozen
    pool = [e["init"] for e in load_frozen(split, who="A6") + load_frozen(split + "_neg", who="A6")]
    cnt = [0]
    def f(rng):
        cnt[0] += 1
        if cnt[0] % 2:
            return sample_init(rng)
        return dict(pool[(cnt[0] // 2) % len(pool)])
    return f

def collect(env_factory, hiddens, n_episodes, seed, steps_per_ep=40, init_sampler=sample_init):
    """Returns per-step arrays X (N,6) = [rel, obj_vel, action], Y (N,4) = [d_obj, obj_vel_next], plus cond ids."""
    rng = np.random.default_rng(seed)
    X, Y, C = [], [], []
    for e in range(n_episodes):
        ci = e % len(hiddens)
        env = env_factory()
        obs = env.reset(init_sampler(rng), hiddens[ci], seed=int(rng.integers(1 << 30)))
        ms = None
        for t in range(steps_per_ep):
            if ms is None or rng.random() < 0.15:
                ms = {"mode": rng.choice(["push", "push", "push", "stop", "rand"]),
                      "ang": rng.normal(0, 0.35), "speed": rng.uniform(0.05, V_MAX)}
            a = _behaviour_action(obs, rng, ms)
            r = env.step(a)
            o2 = r.obs
            X.append(np.r_[obs["obj"] - obs["pusher"], obs["obj_vel"], a])
            Y.append(np.r_[o2["obj"] - obs["obj"], o2["obj_vel"]])
            C.append(ci)
            obs = o2
            if r.done:
                break
    return np.array(X), np.array(Y), np.array(C)


# ---------------- predictor -----------------
class EnsemblePredictor:
    """Implements shared Predictor protocol. Never reads hidden params."""
    name = "a6_mlp_ensemble"

    def __init__(self, members, xm, xs, ym, ys, meta):
        self.members, self.xm, self.xs, self.ym, self.ys, self.meta = members, xm, xs, ym, ys, meta

    def _step(self, j, rel, ov, a):
        x = (np.concatenate([rel, ov, a], -1) - self.xm) / self.xs
        y = _fwd(self.members[j], x)[0] * self.ys + self.ym
        return y[..., :2], y[..., 2:]

    def rollout(self, obs, actions):
        """actions (K,H,2) -> obj traj (M,K,H,2), obj vel traj (M,K,H,2), contact mask (M,K,H)"""
        A = clip_action(np.asarray(actions, float))
        K, H, _ = A.shape; M = len(self.members)
        T = np.zeros((M, K, H, 2)); V = np.zeros((M, K, H, 2)); Cm = np.zeros((M, K, H), bool)
        for j in range(M):
            p = np.tile(np.asarray(obs["pusher"], float), (K, 1))
            o = np.tile(np.asarray(obs["obj"], float), (K, 1))
            ov = np.tile(np.asarray(obs["obj_vel"], float), (K, 1))
            for t in range(H):
                d, ov2 = self._step(j, o - p, ov, A[:, t])
                p = p + A[:, t] * DT
                o = o + d; ov = ov2
                T[j, :, t] = o; V[j, :, t] = ov
                Cm[j, :, t] = np.linalg.norm(o - p, axis=1) < CONTACT_DIST
        return T, V, Cm

    def base_predict(self, obs, actions):
        T, V, Cm = self.rollout(obs, actions)
        return T, V, Cm

    def predict(self, observation, actions, context=None):
        context = context or {}
        actions = np.asarray(actions, float)
        T, V, Cm = self.rollout(observation, actions)
        samples = T[:, :, -1].transpose(1, 0, 2).copy()          # (K,M,2)
        base = samples.mean(1)
        extra = {"base_endpoints": base.copy(), "model": self.name}
        st = context.get("adapt_state") if context else None
        if st is not None and st.get("kind") == "rls_residual":
            from src.adaptation.rls import residual_features
            o0 = np.asarray(observation["obj"], float); v0 = np.asarray(observation["obj_vel"], float)
            Kk = actions.shape[0]
            res_std = np.zeros(Kk)
            for j in range(T.shape[0]):
                F = residual_features(o0, v0, T[j], V[j], Cm[j], actions)[..., st.get("feat_idx", [0, 1, 2, 3])]   # (K,2,J)
                samples[:, j] += F @ st["w"]
            Fm = residual_features(o0, v0, T.mean(0), V.mean(0), Cm.mean(0) > 0.5, actions)[..., st.get("feat_idx", [0, 1, 2, 3])]
            # predictive std of residual (RLS posterior), averaged over 2 dims
            for k in range(Kk):
                res_std[k] = np.sqrt(st["sigma2"] + np.mean(np.einsum("dj,jl,dl->d", Fm[k], st["P"], Fm[k])))
            extra["residual_std"] = res_std
        endpoints = samples.mean(1)
        traj = T.mean(0)
        if st is not None and st.get("kind") == "rls_residual":
            traj = traj + (endpoints - base)[:, None, :] * (np.arange(1, traj.shape[1] + 1) / traj.shape[1])[None, :, None]
        return Prediction(endpoints=endpoints, samples=samples, traj=traj, extra=extra)


# Split grouped by hidden condition AND episode (stricter than A10 PROTOCOL §1, which allows seen mus):
#  train mus {0.2,0.4,0.5} on A10 train/train_neg init states (+ random scenes); val mus {0.15 (outside train range,
#  proxy for extrapolation), 0.5 (in range)} on A10 val/val_neg states; binding test mus (A10 frozen, PROTOCOL A-7)
#  0.1 / 0.3 / 0.6 are ALL unseen by the model (0.1, 0.6 = extrapolation; 0.3 = interpolation).
# Superseded: v1 (18:02) train {0.15,0.2,0.4,0.5}; v2 (18:08) train {0.15,0.2,0.35,0.6} (A4 grid) - not used for final results.
DEFAULT_CFG = {"train_mus": [0.2, 0.4, 0.5], "val_mus": [0.15, 0.5], "test_mus": [0.1, 0.3, 0.6],
               "n_train_eps": 1600, "n_val_eps": 200, "M": 5, "hidden": 64, "epochs": 40, "env": "push_env",
               "version": "v3"}

def default_env_factory(name="fixture"):
    if name == "fixture":
        from shared.fixture_env import FixturePushEnv
        return lambda: FixturePushEnv()
    if name == "push_env":
        from src.env.push_env import PushEnv  # A4 env (same protocol)
        return lambda: PushEnv()
    raise ValueError(name)

def train(seed=0, cfg=None, env_factory=None, verbose=True):
    cfg = dict(DEFAULT_CFG, **(cfg or {}))
    env_factory = env_factory or default_env_factory(cfg["env"])
    t0 = time.time()
    X, Y, _ = collect(env_factory, [{"mu": m} for m in cfg["train_mus"]], cfg["n_train_eps"], seed=seed * 1000 + 1,
                      init_sampler=mixed_init_sampler('train'))
    Xv, Yv, _ = collect(env_factory, [{"mu": m} for m in cfg["val_mus"]], cfg["n_val_eps"], seed=seed * 1000 + 2,
                        init_sampler=mixed_init_sampler('val'))
    t_collect = time.time() - t0
    xm, xs = X.mean(0), X.std(0) + 1e-6; ym, ys = Y.mean(0), Y.std(0) + 1e-6
    Xn, Yn = (X - xm) / xs, (Y - ym) / ys; Xvn, Yvn = (Xv - xm) / xs, (Yv - ym) / ys
    log = []; members = []
    rng = np.random.default_rng(seed)
    for j in range(cfg["M"]):
        bi = rng.integers(0, len(Xn), len(Xn))   # bootstrap
        members.append(train_mlp(Xn[bi], Yn[bi], seed=seed * 100 + j, hidden=cfg["hidden"], epochs=cfg["epochs"],
                                 Xv=Xvn, Yv=Yvn, log=log))
    n_params = sum(W.size + b.size for W, b in members[0]) * cfg["M"]
    meta = {"cfg": cfg, "seed": seed, "n_train_steps": int(len(X)), "n_val_steps": int(len(Xv)),
            "collect_s": t_collect, "train_s": time.time() - t0 - t_collect, "n_params_total": int(n_params), "log": log}
    if verbose:
        print(json.dumps({k: v for k, v in meta.items() if k != "log"}), flush=True)
    return EnsemblePredictor(members, xm, xs, ym, ys, meta)

def load_or_train(seed=0, cfg=None, env_factory=None, retrain=False):
    cfg = dict(DEFAULT_CFG, **(cfg or {}))
    os.makedirs(MODEL_DIR, exist_ok=True)
    path = os.path.join(MODEL_DIR, f"ens_{cfg['env']}_{cfg['version']}_s{seed}.pkl")
    if os.path.exists(path) and not retrain:
        with open(path, "rb") as f:
            d = pickle.load(f)
        return EnsemblePredictor(**d)
    p = train(seed, cfg, env_factory)
    with open(path, "wb") as f:
        pickle.dump({"members": p.members, "xm": p.xm, "xs": p.xs, "ym": p.ym, "ys": p.ys, "meta": p.meta}, f)
    with open(path.replace(".pkl", "_meta.json"), "w") as f:
        json.dump(p.meta, f, indent=1)
    return p
