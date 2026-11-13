"""Freeze candidate arrays per (init_state_id, seed) so every scorer uses the same set.

Saves to runs/a5/<proposer>__<init_state_id>__seed<seed>.npz
and a manifest at runs/a5/manifest.json.

Does not read live-env parameters. Observation is built from the visible init pose only
(t=0, obj_vel=0) unless an explicit observation is provided.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

DEFAULT_OUT = Path("/workspace/wmprobe/runs/a5")


def observation_from_init(init_state: dict) -> dict:
    return {
        "t": 0.0,
        "pusher": np.asarray(init_state["pusher"], dtype=np.float64),
        "obj": np.asarray(init_state["obj"], dtype=np.float64),
        "obj_vel": np.zeros(2, dtype=np.float64),
        "goal": np.asarray(init_state["goal"], dtype=np.float64),
        "goal_radius": float(init_state.get("goal_radius", 0.05)),
    }


def candidate_path(out_dir: Path, proposer_name: str, init_state_id: str, seed: int) -> Path:
    safe_p = "".join(c if c.isalnum() or c in "-_" else "_" for c in proposer_name)
    safe_i = "".join(c if c.isalnum() or c in "-_" else "_" for c in init_state_id)
    return Path(out_dir) / f"{safe_p}__{safe_i}__seed{int(seed)}.npz"


def save_candidates(
    path: Path,
    actions: np.ndarray,
    *,
    init_state_id: str,
    seed: int,
    proposer_name: str,
    init_state: dict | None = None,
    extra: dict | None = None,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    actions = np.asarray(actions, dtype=np.float64)
    payload = {
        "actions": actions,
        "init_state_id": np.asarray(init_state_id),
        "seed": np.int64(seed),
        "proposer_name": np.asarray(proposer_name),
        "K": np.int64(actions.shape[0]),
        "H": np.int64(actions.shape[1]),
    }
    if init_state is not None:
        payload["init_state_json"] = np.asarray(json.dumps(init_state))
    if extra:
        for k, v in extra.items():
            payload[f"extra_{k}"] = np.asarray(v)
    np.savez_compressed(path, **payload)
    return path


def load_candidates(path: Path) -> dict:
    path = Path(path)
    with np.load(path, allow_pickle=False) as z:
        out = {
            "actions": np.array(z["actions"], dtype=np.float64),
            "init_state_id": str(z["init_state_id"]),
            "seed": int(z["seed"]),
            "proposer_name": str(z["proposer_name"]),
            "K": int(z["K"]),
            "H": int(z["H"]),
        }
        if "init_state_json" in z:
            out["init_state"] = json.loads(str(z["init_state_json"]))
    return out


def freeze_set(
    proposer,
    init_states: list[dict],
    seed: int,
    proposer_name: str,
    out_dir: Path | str = DEFAULT_OUT,
    history=None,
) -> dict:
    """Propose and save one (K,H,2) array per init state. Returns the manifest dict."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    history = [] if history is None else history
    entries = []
    for st in init_states:
        sid = st["id"]
        obs = observation_from_init(st)
        actions = proposer.propose(obs, history, int(seed))
        path = candidate_path(out_dir, proposer_name, sid, seed)
        save_candidates(
            path,
            actions,
            init_state_id=sid,
            seed=int(seed),
            proposer_name=proposer_name,
            init_state={k: v for k, v in st.items()},
        )
        entries.append(
            {
                "init_state_id": sid,
                "seed": int(seed),
                "proposer_name": proposer_name,
                "path": str(path),
                "K": int(actions.shape[0]),
                "H": int(actions.shape[1]),
                "shape": list(actions.shape),
            }
        )
    manifest = {
        "out_dir": str(out_dir),
        "proposer_name": proposer_name,
        "seed": int(seed),
        "n": len(entries),
        "entries": entries,
    }
    man_path = out_dir / f"manifest_{proposer_name}_seed{int(seed)}.json"
    man_path.write_text(json.dumps(manifest, indent=2))
    # Also maintain a combined pointer for consumers.
    index_path = out_dir / "manifest.json"
    if index_path.exists():
        try:
            index = json.loads(index_path.read_text())
        except json.JSONDecodeError:
            index = {"sets": []}
    else:
        index = {"sets": []}
    # replace same proposer+seed if re-frozen
    index["sets"] = [
        s
        for s in index.get("sets", [])
        if not (s.get("proposer_name") == proposer_name and s.get("seed") == int(seed))
    ]
    index["sets"].append({"manifest": str(man_path), "proposer_name": proposer_name, "seed": int(seed), "n": len(entries)})
    index_path.write_text(json.dumps(index, indent=2))
    return manifest
