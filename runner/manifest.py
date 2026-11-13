"""Frozen-artifact hash manifest (configs/frozen_manifest.json). Refuse to run if any hash changed."""
from __future__ import annotations
import hashlib, json, os, datetime
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFEST = os.path.join(ROOT, "configs", "frozen_manifest.json")
FROZEN = {
  "A6": ["runs/a6/models/ens_push_env_v3_s0.pkl", "src/dynamics/model.py", "src/adaptation/rls.py", "reports/a6/FROZEN.md"],
  "A7": ["reports/a7/frozen_scorer_config.json", "src/scoring/__init__.py", "src/scoring/config.py", "src/scoring/scorers.py",
         "src/scoring/gate.py"],
  "A8": ["src/probing/frozen_config.json", "src/probing/loop.py", "src/probing/probers.py", "src/probing/belief.py",
         "src/probing/physics_model.py", "src/probing/stubs.py", "src/probing/experiments.py", "src/probing/probe_library.py"],
  "A9": ["runs/a9/frozen_strike54/models/obs_only.npz", "runs/a9/frozen_strike54/models/history_direct.npz",
         "src/direct_baselines/api.py", "src/direct_baselines/selectors.py", "src/direct_baselines/features.py",
         "src/direct_baselines/models.py", "src/direct_baselines/candidates.py", "src/direct_baselines/physics_heuristic.py",
         "reports/a9/config_frozen.json", "reports/a9/API.md"],
  "A4_env": ["src/env/push_env.py", "src/env/scenarios.py"],
  "shared": ["shared/interface.py"],
  "A10_eval": ["evaluation/frozen_sets.json", "evaluation/audit.py", "evaluation/analyze.py", "evaluation/metrics.py",
               "evaluation/online_regret.py", "evaluation/offline_rank.py", "evaluation/PROTOCOL.md"],
  "runner": ["runner/run_e2e3.py", "runner/methods.py", "runner/manifest.py", "runner/analyze_ctrl.py"],
}
EXPECTED = {"runs/a6/models/ens_push_env_v3_s0.pkl": "eb221f9c804e1a8e383a700d5fb620e1038127a4a798c9c87263d0026187dbc0",
            "src/dynamics/model.py": "d044f0dfad8bd74513d523a7950cdfe5af693c9b11f4ffb46b27a1ccf90d025b",
            "src/adaptation/rls.py": "0500c3f84a22df5b0ae0e7da592cd7910d81645b229207c6950cb94bff74e6e6",
            "reports/a7/frozen_scorer_config.json": "281f07faf27da901aab4ec733064bbd4e189ee37ed88367d22d251f33e046a3a",  # file bytes (== git caa9508)
            "reports/a9/config_frozen.json": "efea60ae33eb617254b8d1cf4d546e982f1a55dfe148b916f99759998c72151e",     # file bytes
            "evaluation/frozen_sets.json": "0c2873b508f7d19e190b1e142be9a7c2dd44975db0daf3845ea7bb648733d5d4"}

# A7/A9 declare a CONTENT hash embedded in the file: sha256(json.dumps(cfg minus "config_sha256", **kw)).
SELF_HASH = {"reports/a7/frozen_scorer_config.json": ({"indent": 2}, "a4aea5da78fca714fee2ed76b1084eb674df02f16c6a31a5fd695c82dadb0466"),
             "reports/a9/config_frozen.json": ({"sort_keys": True}, "19a28a25928e24c0246cec93de1ad31d035b49e4bf836d67230f1237428605ac")}

def self_hash_issues():
    bad = []
    for p, (kw, want) in SELF_HASH.items():
        d = json.load(open(os.path.join(ROOT, p))); emb = d.pop("config_sha256", None)
        got = hashlib.sha256(json.dumps(d, **kw).encode()).hexdigest()
        if got != want or emb != want: bad.append(f"{p}: content hash {got} / embedded {emb} != declared {want}")
    return bad

def sha(p):
    full = os.path.join(ROOT, p)
    if not os.path.exists(full): return None
    with open(full, "rb") as f: return hashlib.sha256(f.read()).hexdigest()

def current(groups=None):
    return {g: {p: sha(p) for p in fs} for g, fs in FROZEN.items() if groups is None or g in groups}

def init_manifest(force_runner_refresh=False):
    cur = current()
    bad = [f"{p}: {cur_h} != owner-declared {EXPECTED[p]}" for g in cur.values() for p, cur_h in g.items() if p in EXPECTED and cur_h != EXPECTED[p]]
    missing = [p for g, d in cur.items() if g != "runner" for p, h in d.items() if h is None]
    bad += self_hash_issues()
    if bad or missing: raise SystemExit(f"REFUSE manifest init: owner hash mismatch {bad} / missing {missing}")
    if os.path.exists(MANIFEST):
        old = json.load(open(MANIFEST))["sha256"]
        diff = [p for g in cur if g != "runner" for p in cur[g] if old.get(g, {}).get(p) != cur[g][p]]
        if diff: raise SystemExit(f"REFUSE: module artifacts changed since manifest: {diff}")
        if not force_runner_refresh: return json.load(open(MANIFEST))
    m = {"written": datetime.datetime.now().astimezone().isoformat(timespec="seconds"), "sha256": cur,
         "sealed_split_expected_sha256": "b6e9c90364fe3c9a109d51dc54c4283a2898a22ad4b2b41f1211195f3d9a66bd (verified by evaluation.audit.load_sealed)",
         "owner_declared_content_hashes": {p: v[1] for p, v in SELF_HASH.items()},
         "note": "Module artifacts (A4/A6/A7/A8/A9/A10/shared) strict; runner/ hashes refreshable only via --refresh-runner-hashes before the sealed run."}
    os.makedirs(os.path.dirname(MANIFEST), exist_ok=True)
    json.dump(m, open(MANIFEST, "w"), indent=1)
    return m

def verify():
    if not os.path.exists(MANIFEST): raise SystemExit("REFUSE: configs/frozen_manifest.json missing (run with --init-manifest first)")
    m = json.load(open(MANIFEST)); cur = current()
    diff = [f"{p}: {m['sha256'].get(g, {}).get(p)} -> {h}" for g, d in cur.items() for p, h in d.items() if m["sha256"].get(g, {}).get(p) != h]
    diff += self_hash_issues()
    if diff: raise SystemExit("REFUSE to run: frozen artifact hash changed:\n  " + "\n  ".join(diff))
    return m
