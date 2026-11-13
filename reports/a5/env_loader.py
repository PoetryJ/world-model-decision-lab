"""Injectable env + E1 init-set loader. Prefer A10 frozen E1 split, then A4 e1_set, else fallback.

This is experiment/eval glue (reports/a5), not policy-side code.
Loads only the e1 split (and condition/offline metadata). Does not load the held-out
E2/E3 split.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/workspace/wmprobe")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from shared.fixture_env import FIXTURE_HIDDEN, FixturePushEnv

FALLBACK_INIT_SEED = 512
FALLBACK_N = 20
FALLBACK_PATH = ROOT / "runs/a5/fallback_e1_inits.json"

# Controller 18:10 (UTC+8): main pair mu_lo=0.1 vs mu_hi=0.5; mu_mid=0.25 secondary
# (A4 scenarios.hidden_conditions(3)). Mid/high often share the best action.
A4_E1_CONDITIONS = {
    "mu_lo": {"mu": 0.1},
    "mu_mid": {"mu": 0.25},
    "mu_hi": {"mu": 0.5},
}
A10_OFFLINE = {
    "select_rep_seeds": [11, 12, 13],
    "eval_rep_seeds": [21, 22, 23],
    "settle_steps": 20,
    "tie_tol_m": 0.001,
}


def _as_init(d: dict, default_id: str, seed: int | None = None) -> dict:
    src = d.get("init", d)
    out = {
        "id": d.get("id") or d.get("init_state_id") or d.get("name") or default_id,
        "pusher": np.asarray(src["pusher"], dtype=np.float64).tolist(),
        "obj": np.asarray(src["obj"], dtype=np.float64).tolist(),
        "goal": np.asarray(src["goal"], dtype=np.float64).tolist(),
        "goal_radius": float(src.get("goal_radius", d.get("goal_radius", 0.05))),
    }
    if seed is not None:
        out["seed"] = int(seed)
    elif "seed" in d:
        out["seed"] = int(d["seed"])
    return out


def generate_fallback_inits(n: int = FALLBACK_N, seed: int = FALLBACK_INIT_SEED) -> list[dict]:
    rng = np.random.default_rng(int(seed))
    states = []
    for i in range(n):
        ok = False
        for _ in range(2000):
            obj = rng.uniform(-0.22, 0.22, size=2)
            theta = float(rng.uniform(0.0, 2.0 * np.pi))
            dist = float(rng.uniform(0.18, 0.32))
            goal = obj + dist * np.array([np.cos(theta), np.sin(theta)])
            if np.any(np.abs(goal) > 0.35):
                continue
            gdir = goal - obj
            gdir = gdir / np.linalg.norm(gdir)
            pusher = obj - gdir * float(rng.uniform(0.11, 0.18))
            if np.any(np.abs(pusher) > 0.45):
                continue
            if np.linalg.norm(pusher - obj) < 0.09:
                continue
            states.append(
                {
                    "id": f"fb_{i:02d}",
                    "pusher": pusher.tolist(),
                    "obj": obj.tolist(),
                    "goal": goal.tolist(),
                    "goal_radius": 0.05,
                    "seed": int(seed) + i,
                }
            )
            ok = True
            break
        if not ok:
            raise RuntimeError(f"failed to sample init {i}")
    return states


def _load_a10_e1() -> list[dict] | None:
    try:
        from evaluation.audit import load_frozen
    except Exception:
        return None
    try:
        raw = load_frozen("e1", who="A5")
    except Exception:
        return None
    if not isinstance(raw, list) or not raw:
        return None
    return [_as_init(d, d.get("init_state_id", f"e1_{i:03d}")) for i, d in enumerate(raw)]


def _try_a4_e1() -> tuple[list[dict] | None, list[dict] | None, str]:
    scen_path = ROOT / "src/env/scenarios.py"
    if not scen_path.exists():
        return None, None, "a4-scenarios-absent"
    try:
        import env.scenarios as mod  # type: ignore
    except Exception:
        try:
            spec = importlib.util.spec_from_file_location("a4_scenarios", scen_path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
        except Exception as e:
            return None, None, f"a4-scenarios-import-failed:{e}"
    inits = None
    if hasattr(mod, "e1_set"):
        try:
            pairs = mod.e1_set(negative_control=False)
            inits = []
            for i, item in enumerate(pairs):
                if isinstance(item, tuple) and len(item) == 2:
                    sid, st = item
                    d = dict(st)
                    d["id"] = sid
                    inits.append(_as_init(d, sid))
                elif isinstance(item, dict):
                    inits.append(_as_init(item, item.get("id", f"a4_{i:03d}")))
        except Exception as e:
            return None, None, f"a4-e1_set-failed:{e}"
    hiddens = None
    if hasattr(mod, "hidden_conditions"):
        try:
            hiddens = [{"id": hid, **dict(h)} for hid, h in mod.hidden_conditions(3)]
        except Exception:
            hiddens = None
    elif hasattr(mod, "HIDDEN_CONDITIONS"):
        raw = mod.HIDDEN_CONDITIONS
        if raw and isinstance(raw[0], tuple):
            hiddens = [{"id": hid, **dict(h)} for hid, h in raw]
    note = "a4-e1_set" if inits else "a4-scenarios-no-e1"
    return inits, hiddens, note


def load_init_states(n: int = 20) -> tuple[list[dict], str]:
    """Return (states, source_label). Prefers A10 e1 freeze, then A4 e1_set, then fallback."""
    a10 = _load_a10_e1()
    if a10:
        return a10[:n], "a10:load_frozen(e1)"
    a4_inits, _, note = _try_a4_e1()
    if a4_inits:
        return a4_inits[:n], note
    if FALLBACK_PATH.exists():
        data = json.loads(FALLBACK_PATH.read_text())
        raw = data.get("init_states", data) if isinstance(data, dict) else data
        loaded = [_as_init(d, d.get("id", f"fb_{i:02d}")) for i, d in enumerate(raw)]
        if loaded:
            return loaded[:n], f"fallback-file:{FALLBACK_PATH}"
    states = generate_fallback_inits(n=n, seed=FALLBACK_INIT_SEED)
    FALLBACK_PATH.parent.mkdir(parents=True, exist_ok=True)
    FALLBACK_PATH.write_text(json.dumps({"seed": FALLBACK_INIT_SEED, "init_states": states}, indent=2))
    return states, f"fallback-generated-seed{FALLBACK_INIT_SEED}"


def hidden_id(h: dict) -> str:
    if "id" in h:
        return str(h["id"])
    mu = float(h["mu"])
    s = f"{mu:.2f}".replace(".", "p")
    return f"mu_{s}"


def load_hidden_conditions() -> tuple[list[dict], str]:
    # Prefer the A10 E1/test condition table (ids mu_lo / mu_mid / mu_hi).
    # These values are published in PROTOCOL.md; we do not open the holdout split.
    return [{"id": k, **v} for k, v in A4_E1_CONDITIONS.items()], "controller-A4-mu_lo0.1_mid0.25_hi0.5"


def make_env(action_noise: float = 0.0, max_time: float = 6.0, obs_noise: float = 0.0):
    """Return (env, source_label). Prefers A4 PushEnv if present and constructible."""
    push_path = ROOT / "src/env/push_env.py"
    if push_path.exists():
        try:
            try:
                import env.push_env as pe  # type: ignore
            except Exception:
                spec = importlib.util.spec_from_file_location("a4_push_env", push_path)
                pe = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(pe)
            cls = getattr(pe, "PushEnv", None)
            if cls is not None:
                try:
                    env = cls(action_noise=action_noise, obs_noise=obs_noise, max_time=max_time)
                except TypeError:
                    env = cls(action_noise=action_noise, max_time=max_time)
                return env, f"a4:{cls.__name__}"
        except Exception as e:
            return FixturePushEnv(action_noise=action_noise, max_time=max_time), f"fixture-A4importfailed:{e}"
    return FixturePushEnv(action_noise=action_noise, max_time=max_time), "fixture"


def env_is_noisy(env) -> bool:
    return float(getattr(env, "action_noise", 0.0) or 0.0) > 0.0


if __name__ == "__main__":
    env, elab = make_env()
    inits, ilab = load_init_states()
    hids, hlab = load_hidden_conditions()
    print(f"env={elab} type={type(env).__name__}")
    print(f"inits={len(inits)} source={ilab} ids={[s['id'] for s in inits[:3]]}...")
    print(f"hiddens={hids} source={hlab}")
