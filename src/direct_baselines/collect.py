"""OFFLINE data collection and oracle labels for A9.

This module is evaluation-side: it may call env.reset(..., hidden, ...) and may
branch via snapshots / src.env.branch_eval. Policy-side modules must not import
evaluate_candidates or read env.hidden.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shared.interface import Transition
from src.env.push_env import PushEnv
from src.env.branch_eval import evaluate_candidates, SETTLE_MAX
from src.env.scenarios import (
    train_set, e1_set, hidden_conditions, probe_set, PROBE_NAMES,
)
from .candidates import default_task_candidates, load_shared_candidates


def episode_seed(init_id: str, hid_id: str, phase: str, base: int = 10_000) -> int:
    h = hashlib.sha256(f"{init_id}|{hid_id}|{phase}|{base}".encode()).hexdigest()
    return base + (int(h, 16) % 1_000_000)


def make_env(max_time: float = 6.0, action_noise: float = 0.0, obs_noise: float = 0.0) -> PushEnv:
    return PushEnv(action_noise=action_noise, obs_noise=obs_noise, max_time=max_time)


CANDIDATE_SOURCE = "auto"

def set_candidate_source(source: str) -> None:
    global CANDIDATE_SOURCE
    CANDIDATE_SOURCE = source


def propose_candidates(obs: dict, seed: int = 0) -> tuple[np.ndarray, str]:
    return load_shared_candidates(obs, seed, source=CANDIDATE_SOURCE)


def apply_probe(env, obs, probe_name: str) -> tuple[list, dict]:
    """Execute a named A4 probe from the current state. Returns (history, next_obs)."""
    probes = probe_set(obs)
    acts = probes[probe_name]
    res = env.step_segment(acts)
    nxt = env.observe()
    hist = [Transition(obs=obs, actions=np.asarray(acts, dtype=np.float64), next_obs=nxt, kind="probe")]
    return hist, nxt


@dataclass
class LabeledRow:
    init_id: str
    hid_id: str
    context: str          # "no_probe" | "tap_fast" | "push_slow" | "retreat"
    set_name: str         # "pos" | "neg"
    cand_index: int
    y_dist: float         # settled distance (lower better)
    y_success: int
    y_score: float        # training target, higher better = -dist
    group: str            # init_id|hid_id|context  (for listwise val / metrics)


def _meta_set(init_state: dict) -> str:
    return str((init_state.get("meta") or {}).get("set", "unknown"))


def collect_split(
    inits: list[tuple[str, dict]],
    hiddens: list[tuple[str, dict]],
    contexts: tuple[str, ...] = ("no_probe", "tap_fast", "push_slow", "retreat"),
    max_time: float = 6.0,
    settle_max: int = SETTLE_MAX,
    env_seed_base: int = 10_000,
) -> tuple[list[dict], np.ndarray, dict]:
    """Return (rows_jsonable, nothing) plus a list of per-row dicts with X-side refs.

    Each row stores observation/history/candidates/labels. Features are built later
    so feature code can change without re-rolling the env.
    """
    env = make_env(max_time=max_time)
    rows: list[dict] = []
    t0 = time.perf_counter()
    n_roll = 0
    cand_source = None
    for item in inits:
        if len(item) == 3:
            init_id, init, seed0 = item[0], item[1], int(item[2])
        else:
            init_id, init = item[0], item[1]
            seed0 = None
        set_name = _meta_set(init)
        # strip meta before reset? PushEnv only reads pusher/obj/goal/goal_radius, meta is ok.
        init_use = {k: v for k, v in init.items() if k != "meta"}
        for hid_id, hidden in hiddens:
            seed = int(seed0) if seed0 is not None else episode_seed(init_id, hid_id, "reset", env_seed_base)
            env.reset(init_use, hidden, seed)
            obs0 = env.observe()
            snap0 = env.snapshot()
            cands, tag = propose_candidates(obs0, seed)
            if cand_source is None:
                cand_source = tag
            for ctx in contexts:
                env.restore(snap0)
                if ctx == "no_probe":
                    hist: list = []
                    obs = env.observe()
                else:
                    hist, obs = apply_probe(env, obs0, ctx)
                    if env.done:
                        # probe already ended the episode; skip candidate labels
                        continue
                # Re-propose at the *decision* observation so the shared set matches
                # what a runner would ask the proposer for at decision time.
                cands_dec, _ = propose_candidates(obs, seed)
                snap = env.snapshot()
                dist, succ, _final = evaluate_candidates(env, snap, cands_dec, settle_max=settle_max)
                n_roll += int(cands_dec.shape[0])
                group = f"{init_id}|{hid_id}|{ctx}"
                for k in range(cands_dec.shape[0]):
                    row = {
                        "init_id": init_id,
                        "hid_id": hid_id,
                        "context": ctx,
                        "set_name": set_name,
                        "cand_index": k,
                        "y_dist": float(dist[k]),
                        "y_success": int(bool(succ[k])),
                        "y_score": float(-dist[k]),
                        "group": group,
                        "obs": _json_obs(obs) if k == 0 else None,
                        "history": _json_hist(hist) if k == 0 else None,
                        "actions": cands_dec[k].tolist(),
                    }
                    # full candidate array once per group (k==0) for eval replay
                    if k == 0:
                        row["candidates"] = cands_dec.tolist()
                    rows.append(row)
    elapsed = time.perf_counter() - t0
    meta = {
        "n_rows": len(rows),
        "n_groups": len({r["group"] for r in rows}),
        "n_inits": len(inits),
        "n_hiddens": len(hiddens),
        "hidden_ids": [h[0] for h in hiddens],
        "contexts": list(contexts),
        "candidate_source": cand_source,
        "collect_seconds": elapsed,
        "n_oracle_rollouts": n_roll,
        "settle_max": settle_max,
        "max_time": max_time,
        "env": "src.env.push_env.PushEnv [toy]",
        "env_seed_base": env_seed_base,
    }
    return rows, meta


def _json_obs(obs: dict) -> dict:
    out = {}
    for k, v in obs.items():
        if isinstance(v, np.ndarray):
            out[k] = v.tolist()
        else:
            out[k] = v
    return out


def _json_hist(hist: list) -> list:
    rec = []
    for tr in hist:
        rec.append({
            "obs": _json_obs(tr.obs),
            "actions": np.asarray(tr.actions, dtype=np.float64).tolist(),
            "next_obs": _json_obs(tr.next_obs),
            "kind": tr.kind,
        })
    return rec


def hist_from_json(blob: list) -> list:
    out = []
    for tr in blob:
        out.append(Transition(
            obs=_arrays_obs(tr["obs"]),
            actions=np.asarray(tr["actions"], dtype=np.float64),
            next_obs=_arrays_obs(tr["next_obs"]),
            kind=tr["kind"],
        ))
    return out


def _arrays_obs(obs: dict) -> dict:
    out = dict(obs)
    for k in ("pusher", "obj", "obj_vel", "goal"):
        if k in out:
            out[k] = np.asarray(out[k], dtype=np.float64)
    return out


def fill_group_fields(rows: list[dict]) -> list[dict]:
    """Reattach obs/history/candidates stored only on cand_index==0."""
    last = {}
    out = []
    for r in rows:
        if r.get("obs") is not None:
            last[r["group"]] = (r["obs"], r["history"], r.get("candidates"))
        obs, hist, cands = last[r["group"]]
        rr = dict(r)
        rr["obs"], rr["history"] = obs, hist
        if cands is not None:
            rr["candidates"] = cands
        out.append(rr)
    return out


def rows_to_xy(rows: list[dict], use_history: bool):
    from .features import build_features
    rows = fill_group_fields(rows)
    X, y, groups = [], [], []
    for r in rows:
        obs = _arrays_obs(r["obs"])
        hist = hist_from_json(r["history"])
        if "actions" in r:
            actions = np.asarray(r["actions"], dtype=np.float64)
        else:
            actions = np.asarray(r["candidates"], dtype=np.float64)[r["cand_index"]]
        feat = build_features(obs, actions, hist, use_history=use_history)
        X.append(feat)
        y.append(r["y_score"])
        groups.append(r["group"])
    return np.stack(X), np.asarray(y, dtype=np.float64), np.asarray(groups)
