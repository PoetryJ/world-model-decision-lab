"""A10 audit tools: hidden-param leakage (static + runtime), split hygiene, snapshot exactness,
online budget/continuity accounting, test-set access logging. Read-only w.r.t. other agents' code.
CLI: python -m evaluation.audit static|splits|snapshot [--env fixture|push]
"""
from __future__ import annotations
import copy, datetime, hashlib, inspect, json, os, re, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
POLICY_DIRS = ["src/candidates", "src/planning", "src/dynamics", "src/adaptation", "src/scoring",
               "src/probing", "src/direct_baselines"]
ALLOWED_OBS_KEYS = {"t", "pusher", "obj", "obj_vel", "goal", "goal_radius"}
FROZEN = os.path.join(ROOT, "evaluation", "frozen_sets.json")
ACCESS_LOG = os.path.join(ROOT, "reports", "a10", "test_access_log.jsonl")

# ------------------------------------------------------------------ 1. static leak scan
LEAK_PATTERNS = [
    (r"\.hidden\b", "reads .hidden"), (r"\[\s*['\"]mu['\"]\s*\]", "indexes ['mu']"),
    (r"['\"]hidden['\"]", "string 'hidden'"), (r"\bsnapshot\s*\(", "calls snapshot()"),
    (r"\brestore\s*\(", "calls restore()"), (r"frozen_sets", "reads frozen_sets"),
    (r"\bevaluation\b", "imports/mentions evaluation"), (r"offline_rank", "uses offline oracle"),
    (r"FIXTURE_HIDDEN|CONDITIONS\b", "hidden condition table"), (r"\benv\.(o|ov|p|pv|rng)\b", "env internals"),
    (r"\bstep_segment\s*\(|\benv\.step\s*\(", "drives an env"), (r"test_\d{3}", "test init id literal"),
]

def static_scan(dirs=POLICY_DIRS, root=ROOT):
    hits = []
    for d in dirs:
        base = os.path.join(root, d)
        for dp, _, fs in os.walk(base):
            for f in fs:
                if not f.endswith(".py"): continue
                p = os.path.join(dp, f)
                for i, line in enumerate(open(p, errors="replace"), 1):
                    s = line.split("#", 1)[0]
                    for pat, why in LEAK_PATTERNS:
                        if re.search(pat, s):
                            hits.append({"file": os.path.relpath(p, root), "line": i, "why": why, "code": line.strip()[:160]})
    return hits

# ------------------------------------------------------------------ 2. runtime leak guard
class HiddenAccessError(RuntimeError):
    pass

POLICY_METHODS = {"propose", "predict", "update", "score", "choose_probe", "act", "select"}

def _policy_frame_in_stack():
    """Violation if (a) the code that directly touches hidden (nearest repo frame, skipping this file) is in a policy dir,
    or (b) hidden is touched (e.g. via env.step/snapshot) while a policy-side protocol method is executing.
    Runner loops that live in policy dirs (e.g. src/probing/loop.py calling env.step) are NOT violations."""
    f = sys._getframe(2); nearest = None
    def pol(fn):
        return any(fn.startswith(os.path.join(ROOT, d) + os.sep) for d in POLICY_DIRS)
    while f is not None:
        fn = os.path.abspath(f.f_code.co_filename)
        if fn.startswith(ROOT + os.sep) and fn != os.path.abspath(__file__) and nearest is None:
            nearest = fn
            if pol(fn): return f"direct:{os.path.relpath(fn, ROOT)}:{f.f_lineno}"
        if pol(fn) and f.f_code.co_name in POLICY_METHODS:
            return f"inside {f.f_code.co_name}:{os.path.relpath(fn, ROOT)}:{f.f_lineno}"
        f = f.f_back
    return None

class TrackedHidden(dict):
    """dict that records any access made while a policy-side frame is on the stack."""
    violations: list = []   # shared class-level log
    strict = False
    def _chk(self, op):
        where = _policy_frame_in_stack()
        if where:
            TrackedHidden.violations.append({"op": op, "where": where})
            if TrackedHidden.strict: raise HiddenAccessError(f"hidden accessed from {where}")
    def __getitem__(self, k): self._chk(f"getitem {k}"); return dict.__getitem__(self, k)
    def get(self, k, d=None): self._chk(f"get {k}"); return dict.get(self, k, d)
    def __iter__(self): self._chk("iter"); return dict.__iter__(self)
    def items(self): self._chk("items"); return dict.items(self)
    def values(self): self._chk("values"); return dict.values(self)
    def keys(self): self._chk("keys"); return dict.keys(self)
    def __contains__(self, k): self._chk("contains"); return dict.__contains__(self, k)
    def __reduce_ex__(self, proto): return (TrackedHidden, (dict(self),))
    def __deepcopy__(self, memo): return TrackedHidden(dict(self))
    def copy(self): return TrackedHidden(dict(self))

def track_env_hidden(env):
    """Call after env.reset(): replaces env.hidden with a TrackedHidden (same values)."""
    env.hidden = TrackedHidden(dict.items(env.hidden) if isinstance(env.hidden, TrackedHidden) else env.hidden.items())
    return env

def scan_args(obj, path="arg", depth=0, out=None):
    """Find env-like objects / hidden keys inside call arguments to policy-side code."""
    out = [] if out is None else out
    if depth > 6: return out
    if hasattr(obj, "snapshot") and hasattr(obj, "restore"):
        out.append(f"{path}: env-like object {type(obj).__name__}")
    elif hasattr(obj, "hidden") and not isinstance(obj, type):
        out.append(f"{path}: object with .hidden")
    elif isinstance(obj, dict):
        for k, v in obj.items():
            if k in ("hidden", "mu", "rng", "_cls", "snapshot"):
                out.append(f"{path}[{k!r}]: forbidden key")
            scan_args(v, f"{path}[{k!r}]", depth + 1, out)
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj[:50]):
            scan_args(v, f"{path}[{i}]", depth + 1, out)
    elif hasattr(obj, "__dataclass_fields__"):
        for k in obj.__dataclass_fields__:
            scan_args(getattr(obj, k), f"{path}.{k}", depth + 1, out)
    return out

def check_observation(ob, path="obs"):
    extra = set(ob) - ALLOWED_OBS_KEYS
    return [f"{path}: extra keys {sorted(extra)}"] if extra else []

class GuardedPolicy:
    """Wrap any policy-side object; every method call has its args scanned. Also scans the wrapped
    object's own attributes for env references (once at wrap time and at each call)."""
    METHODS = {"propose", "predict", "update", "score", "choose_probe", "act", "select", "fit"}
    def __init__(self, inner, name="policy"):
        object.__setattr__(self, "_inner", inner); object.__setattr__(self, "_name", name)
        object.__setattr__(self, "violations", [])
        self._scan_self()
    def _scan_self(self):
        try: attrs = vars(self._inner)
        except TypeError: return
        for k, v in attrs.items():
            for m in scan_args(v, f"{self._name}.{k}", 0):
                if "forbidden key" in m and "rng" in m:   # a policy may own its own rng-named dict entry
                    continue
                self.violations.append(m)
    def __getattr__(self, k):
        a = getattr(self._inner, k)
        if k in self.METHODS and callable(a):
            def wrapped(*args, **kw):
                for i, x in enumerate(args):
                    self.violations.extend(scan_args(x, f"{self._name}.{k}.arg{i}"))
                    if isinstance(x, dict) and "obj" in x and "pusher" in x:
                        self.violations.extend(check_observation(x, f"{self._name}.{k}.arg{i}"))
                for kk, x in kw.items():
                    self.violations.extend(scan_args(x, f"{self._name}.{k}.{kk}"))
                return a(*args, **kw)
            return wrapped
        return a

# ------------------------------------------------------------------ 3. split hygiene
def split_hygiene(sets=None, training_ids=None):
    sets = sets or load_frozen("__audit__", who="audit.split_hygiene", log=False)
    issues = []
    splits = ["train", "train_neg", "val", "val_neg", "e0", "e0_neg", "e1", "test", "test_neg"]
    seeds = {s: {x["seed"] for x in sets[s]} for s in splits}
    ids = {s: {x["init_state_id"] for x in sets[s]} for s in splits}
    geo = {s: {json.dumps(x["init"], sort_keys=True) for x in sets[s]} for s in splits}
    for i, a in enumerate(splits):
        for b in splits[i + 1:]:
            if seeds[a] & seeds[b]: issues.append(f"seed overlap {a}/{b}")
            if ids[a] & ids[b]: issues.append(f"id overlap {a}/{b}")
            if geo[a] & geo[b]: issues.append(f"identical init geometry {a}/{b}: {len(geo[a] & geo[b])}")
    tv = {json.dumps(v, sort_keys=True) for v in sets["conditions"]["train_val"].values()}
    if json.dumps({"mu": 0.3}, sort_keys=True) in tv: issues.append("held-out mu=0.3 present in train_val")
    if os.path.exists(SEALED):
        ss = load_sealed("A10-hygiene", "__all__")
        for k in ("test_sealed", "test_sealed_neg"):
            sg = {json.dumps(x["init"], sort_keys=True) for x in ss[k]}; sd = {x["seed"] for x in ss[k]}
            for a in splits:
                if sg & geo[a]: issues.append(f"identical init geometry {k}/{a}")
                if sd & seeds[a]: issues.append(f"seed overlap {k}/{a}")
    if training_ids is not None:
        bad = set(training_ids) & (ids["test"] | seeds["test"] | ids["test_neg"] | seeds["test_neg"])
        if bad: issues.append(f"training used test items: {sorted(bad)[:10]}")
    return issues

# ------------------------------------------------------------------ 4. snapshot exactness
def _deep_equal(a, b, path="s"):
    diffs = []
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        if not (isinstance(a, np.ndarray) and isinstance(b, np.ndarray) and a.shape == b.shape and np.array_equal(a, b)):
            diffs.append(path)
    elif isinstance(a, dict) and isinstance(b, dict):
        for k in set(a) | set(b):
            if k not in a or k not in b: diffs.append(f"{path}.{k} missing")
            else: diffs += _deep_equal(a[k], b[k], f"{path}.{k}")
    elif isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b): diffs.append(f"{path} len")
        else:
            for i, (x, y) in enumerate(zip(a, b)): diffs += _deep_equal(x, y, f"{path}[{i}]")
    else:
        if a != b and not (isinstance(a, float) and isinstance(b, float) and np.isnan(a) and np.isnan(b)):
            diffs.append(path)
    return diffs

def snapshot_exactness(env_factory, init, hidden, seed=0, warm_steps=5, test_steps=15, action_seed=123):
    """Checks: (a) restore->replay gives bit-identical obs AND full snapshots after each step;
    (b) restore into a FRESH env instance gives the same; (c) snapshot is not aliased to live state;
    (d) RNG state included (stochastic env gives same noise after restore)."""
    rng = np.random.default_rng(action_seed)
    A_warm = rng.uniform(-0.5, 0.5, (warm_steps, 2)); A = rng.uniform(-0.5, 0.5, (test_steps, 2))
    A[: test_steps // 2] = np.abs(A[: test_steps // 2])  # some pushing
    env = env_factory(); env.reset(init, hidden, seed); env.step_segment(A_warm)
    snap = env.snapshot(); snap_copy = copy.deepcopy(snap)
    def run(e):
        obs, snaps = [], []
        for a in A:
            r = e.step(a); obs.append(r.obs); snaps.append(e.snapshot())
            if r.done: break
        return obs, snaps
    o1, s1 = run(env)
    issues = []
    if _deep_equal(snap, snap_copy): issues.append("snapshot aliased to live state (mutated by later steps)")
    env.restore(snap); o2, s2 = run(env)
    d = _deep_equal(o1, o2, "obs") + _deep_equal(s1, s2, "snap")
    if d: issues.append(f"same-instance replay mismatch: {d[:5]}")
    env2 = env_factory(); env2.reset({"pusher": [0, 0.3], "obj": [0.2, -0.2], "goal": [-0.2, 0.2], "goal_radius": 0.05},
                                     {"mu": 0.9 if "mu" in dict(hidden) else 0}, seed + 999)
    env2.restore(snap); o3, s3 = run(env2)
    d = _deep_equal(o1, o3, "obs") + _deep_equal(s1, s3, "snap")
    if d: issues.append(f"fresh-instance replay mismatch: {d[:5]}")
    return {"issues": issues, "n_steps_compared": len(o1), "snapshot_keys": sorted(snap.keys())}

# ------------------------------------------------------------------ 5. online budget + continuity
def online_accounting(records, budget, dt=0.1, tol=1e-9):
    """records: DecisionRecord dicts. Checks per episode: step budget, probe budget, contiguous
    step_index, obs continuity (obs_before[i+1] == obs_after[i]: no hidden restore/reset in online run),
    physical time = executed steps * dt, termination present on last record, no records after done."""
    issues = []
    by = {}
    for r in records: by.setdefault((r["method"], r["episode_id"]), []).append(r)
    for key, rs in by.items():
        rs = sorted(rs, key=lambda r: r["step_index"])
        idx = [r["step_index"] for r in rs]
        if any(b <= a for a, b in zip(idx[:-1], idx[1:])): issues.append(f"{key}: step_index not strictly increasing")
        steps = sum(len(r["executed"]) for r in rs); probe = sum(len(r["executed"]) for r in rs if r["kind"] == "probe")
        if steps > budget["max_steps_total"]: issues.append(f"{key}: steps {steps} > {budget['max_steps_total']}")
        if probe > budget["max_probe_steps"]: issues.append(f"{key}: probe steps {probe} > {budget['max_probe_steps']}")
        for a, b in zip(rs[:-1], rs[1:]):
            for k in ("obj", "pusher", "obj_vel", "t"):
                if not np.allclose(np.asarray(a["obs_after"][k], float), np.asarray(b["obs_before"][k], float), atol=tol, rtol=0):
                    issues.append(f"{key}: discontinuity at step_index {b['step_index']} key {k} (restore/reset?)"); break
            if a.get("termination"): issues.append(f"{key}: record after termination at {b['step_index']}")
        t0 = float(rs[0]["obs_before"]["t"]); t1 = float(rs[-1]["obs_after"]["t"])
        if abs((t1 - t0) - steps * dt) > 1e-6:
            issues.append(f"{key}: physical time {t1 - t0:.3f}s != executed steps*dt {steps * dt:.3f}s")
        if not rs[-1].get("termination"): issues.append(f"{key}: last record has no termination")
        # E2 structure (PROTOCOL A-3): <=max_probes probes of <=max_probe_seg steps, exactly max_task_decisions decisions,
        # task segment <= max_task_segment_steps, any later (settle) records must be all-zero actions (no corrective pushes)
        probes = [r for r in rs if r["kind"] == "probe"]
        dec = [r for r in rs if r["kind"] == "task" and r.get("candidates") is not None]
        if "max_probes" in budget and len(probes) > budget["max_probes"]: issues.append(f"{key}: {len(probes)} probes > {budget['max_probes']}")
        tdp = terminated_during_probe(rs)
        if "max_task_decisions" in budget and len(dec) != budget["max_task_decisions"] and not (tdp and not dec):
            issues.append(f"{key}: {len(dec)} task decisions != {budget['max_task_decisions']}")
        if tdp and len(rs) > len(probes):   # A-9: nothing may follow a terminating probe
            issues.append(f"{key}: records after env terminated during probe")
        for r in dec:
            if "max_task_segment_steps" in budget and len(r["executed"]) > budget["max_task_segment_steps"]:
                issues.append(f"{key}: task segment {len(r['executed'])} steps > {budget['max_task_segment_steps']}")
            cc = r.get("candidate_counts")   # interface v1.1
            K = len(r["candidates"])
            if not cc: issues.append(f"{key}: candidate_counts missing at {r['step_index']}")
            else:
                if cc.get("requested") != cc.get("returned"): issues.append(f"{key}: candidate_counts requested {cc.get('requested')} != returned {cc.get('returned')}")
                if cc.get("returned") != K: issues.append(f"{key}: candidate_counts returned {cc.get('returned')} != len(candidates) {K}")
                if cc.get("failed", 0): issues.append(f"{key}: {cc['failed']} candidate workers failed")
            if r.get("scores") is not None and len(r["scores"]) != K: issues.append(f"{key}: len(scores) != K")
            if r.get("chosen_index") is None or not (0 <= r["chosen_index"] < K): issues.append(f"{key}: bad chosen_index")
            elif not np.allclose(np.asarray(r["candidates"][r["chosen_index"]], float)[:len(r["executed"])], np.asarray(r["executed"], float), atol=1e-12):
                issues.append(f"{key}: executed != candidates[chosen_index]")
        if dec and "max_task_decisions" in budget:
            after = [r for r in rs if r["step_index"] > dec[-1]["step_index"]]
            if any(np.abs(np.asarray(r["executed"], float)).max(initial=0) > 0 for r in after):
                issues.append(f"{key}: non-zero actions after the single task segment (corrective push)")
        for r in rs:
            issues += [f"{key}: {m}" for m in check_observation(r["obs_before"], "obs_before")]
            ex = np.asarray(r["executed"], float)
            if ex.size and np.any(np.abs(ex) > 0.5 + 1e-12): issues.append(f"{key}: illegal action |a|>V_MAX at {r['step_index']}")
    return issues

TDP_TERMS = ("success", "out_of_bounds")

def terminated_during_probe(rs):
    """A-9: episode validly ends with 0 task decisions if env terminated (success / out_of_bounds) on a probe record."""
    rs = sorted(rs, key=lambda r: r["step_index"])
    return bool(rs) and rs[-1]["kind"] == "probe" and rs[-1].get("termination") in TDP_TERMS and \
        not any(r["kind"] == "task" for r in rs)

def episode_classes(records):
    """Per method: n_episodes, n_probed (>=1 probe), probe_rate, n_terminated_during_probe (+ by termination)."""
    by = {}
    for r in records: by.setdefault((r["method"], r["episode_id"]), []).append(r)
    out = {}
    for (m, _), rs in by.items():
        d = out.setdefault(m, {"n_episodes": 0, "n_probed": 0, "n_terminated_during_probe": 0, "tdp_by_termination": {}})
        d["n_episodes"] += 1
        d["n_probed"] += any(r["kind"] == "probe" for r in rs)
        if terminated_during_probe(rs):
            d["n_terminated_during_probe"] += 1
            t = sorted(rs, key=lambda r: r["step_index"])[-1]["termination"]
            d["tdp_by_termination"][t] = d["tdp_by_termination"].get(t, 0) + 1
    for d in out.values(): d["probe_rate"] = d["n_probed"] / d["n_episodes"]
    return out

# ------------------------------------------------------------------ 5b. LCB / threshold hygiene (A2 findings)
def lcb_sample_check(prediction, name="pred"):
    """An LCB/std-based scorer needs >=2 samples per candidate; S==1 gives std 0 or NaN (ddof=1)."""
    smp = getattr(prediction, "samples", None)
    if smp is None: return [f"{name}: no samples (LCB undefined)"]
    smp = np.asarray(smp)
    if smp.ndim != 3 or smp.shape[1] < 2: return [f"{name}: samples shape {smp.shape}, need (K,S>=2,2)"]
    if not np.all(np.isfinite(smp)): return [f"{name}: non-finite samples"]
    return []

def static_lcb_threshold_scan(root=ROOT):
    """List LCB uses and hard-coded numeric thresholds in policy dirs for manual provenance review (must be val-tuned)."""
    hits = []
    for d in POLICY_DIRS:
        for dp, _, fs in os.walk(os.path.join(root, d)):
            for f in fs:
                if not f.endswith(".py") or f.startswith("test_"): continue
                p = os.path.join(dp, f)
                for i, line in enumerate(open(p, errors="replace"), 1):
                    code = line.split("#", 1)[0]
                    if re.search(r"lcb|ddof\s*=\s*1", code, re.I):
                        hits.append(f"LCB {os.path.relpath(p, root)}:{i}: {line.strip()[:120]}")
                    if re.search(r"(thresh\w*|thr|tau|beta)\s*=\s*-?\d*\.\d+", code, re.I):
                        hits.append(f"THR {os.path.relpath(p, root)}:{i}: {line.strip()[:120]}")
    return hits

# ------------------------------------------------------------------ 6. frozen-set loading with access log
def frozen_sha():
    with open(FROZEN, "rb") as f: return hashlib.sha256(f.read()).hexdigest()

def load_frozen(split, who="unknown", log=True):
    with open(FROZEN) as f: s = json.load(f)
    if log:
        caller = inspect.stack()[1]
        os.makedirs(os.path.dirname(ACCESS_LOG), exist_ok=True)
        with open(ACCESS_LOG, "a") as f:
            f.write(json.dumps({"time": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
                                "split": split, "who": who, "caller": f"{os.path.relpath(caller.filename, ROOT)}:{caller.lineno}",
                                "sha256": frozen_sha()}) + "\n")
    return s if split == "__audit__" else s[split]

def scan_test_readers(root=ROOT):
    """Find code outside evaluation/ that mentions the frozen test split or test ids."""
    hits = []
    for dp, dn, fs in os.walk(root):
        if any(x in dp for x in (".venv", "adapters", "__pycache__", os.path.join(root, "evaluation"), "tests/evaluation")): continue
        for f in fs:
            if f.endswith(".py"):
                p = os.path.join(dp, f)
                for i, line in enumerate(open(p, errors="replace"), 1):
                    if re.search(r"frozen_sets|['\"]test['\"]\]|test_\d{3}|SEED_BASE", line):
                        hits.append(f"{os.path.relpath(p, root)}:{i}: {line.strip()[:140]}")
    return hits

# ------------------------------------------------------------------ 7. sealed test split (PROTOCOL A-8)
SEALED = os.path.join(ROOT, "evaluation", "sealed", "test_sealed.json")
SEALED_SHA = "b6e9c90364fe3c9a109d51dc54c4283a2898a22ad4b2b41f1211195f3d9a66bd"
SEALED_ALLOWED = ("controller", "A10-hygiene")

def load_sealed(who: str, split: str = "test_sealed"):
    """Only the controller's final E2/E3 runner may call this (who='controller...'). Every call (allowed or not) is logged."""
    caller = inspect.stack()[1]
    with open(SEALED, "rb") as f: raw = f.read()
    sha = hashlib.sha256(raw).hexdigest()
    ok = who.startswith(SEALED_ALLOWED)
    os.makedirs(os.path.dirname(ACCESS_LOG), exist_ok=True)
    with open(ACCESS_LOG, "a") as f:
        f.write(json.dumps({"time": datetime.datetime.now().astimezone().isoformat(timespec="seconds"), "split": split,
                            "who": who, "caller": f"{os.path.relpath(caller.filename, ROOT)}:{caller.lineno}", "sha256": sha,
                            "sealed": True, "allowed": ok, "sha_ok": sha == SEALED_SHA}) + "\n")
    if not ok: raise PermissionError(f"sealed split is controller-only (who={who!r}); access logged")
    if sha != SEALED_SHA: raise RuntimeError("sealed split hash mismatch")
    s = json.loads(raw)
    return s if split == "__all__" else s[split]

def sealed_access_audit():
    """Sealed-split reads by anyone other than the controller (or A10 structural hygiene) + code that reads the file directly."""
    issues = []
    if os.path.exists(ACCESS_LOG):
        for l in open(ACCESS_LOG):
            r = json.loads(l)
            if r.get("sealed") and not str(r.get("who", "")).startswith(SEALED_ALLOWED) \
                    and not str(r.get("caller", "")).startswith("tests/evaluation/"):   # A10's own denial test
                issues.append(f"sealed read by {r['who']} at {r['time']} from {r['caller']}")
    for dp, dn, fs in os.walk(ROOT):
        if any(x in dp for x in (".venv", "adapters", "__pycache__")) or dp.startswith(os.path.join(ROOT, "evaluation")): continue
        for f in fs:
            if f.endswith((".py", ".sh")):
                p = os.path.join(dp, f)
                for i, line in enumerate(open(p, errors="replace"), 1):
                    code = line.split("#", 1)[0]
                    if "load_sealed" in code or "sealed/" in code or "test_sealed.json" in code:
                        rel = os.path.relpath(p, ROOT)
                        if not rel.startswith(("runner/", "tests/evaluation/")):
                            issues.append(f"non-controller code mentions sealed split: {rel}:{i}")
    return issues

def runner_logs_audit(paths=None, budget=None):
    """Audit controller runner/ decision logs: A-3 structure, candidate_counts, single task decision, no corrective pushes,
    plus sealed-access audit. Returns {path: issues}."""
    import glob
    budget = budget or {"max_steps_total": 60, "max_probe_steps": 5, "max_probes": 1, "max_task_decisions": 1, "max_task_segment_steps": 12}
    paths = paths or sorted(glob.glob(os.path.join(ROOT, "runner", "**", "*decisions*.jsonl"), recursive=True) +
                            glob.glob(os.path.join(ROOT, "runs", "controller", "**", "*decisions*.jsonl"), recursive=True))
    out = {}
    for p in paths:
        recs = [json.loads(l) for l in open(p)]
        out[os.path.relpath(p, ROOT)] = online_accounting(recs, budget)
        out[os.path.relpath(p, ROOT) + " :: episode_classes"] = episode_classes(recs)
    out["__sealed_access__"] = sealed_access_audit()
    return out

if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "static"
    if cmd == "static":
        for h in static_scan(): print(json.dumps(h))
        print("test-set readers:", *scan_test_readers(), sep="\n  ")
    elif cmd == "lcbthr":
        print(*static_lcb_threshold_scan(), sep="\n")
    elif cmd == "runner":
        for k, v in runner_logs_audit(sys.argv[2:] or None).items():
            if isinstance(v, dict): print(k, json.dumps(v)); continue
            print(k, "OK" if not v else f"{len(v)} issues", *v[:15], sep="\n  ")
    elif cmd == "splits":
        print(split_hygiene() or "OK")
    elif cmd == "snapshot":
        which = sys.argv[2] if len(sys.argv) > 2 else "fixture"
        if which == "push":
            sys.path.insert(0, ROOT); from src.env.push_env import PushEnv as E
        else:
            from shared.fixture_env import FixturePushEnv as E
        from shared.fixture_env import FIXTURE_INIT
        for noise in (0.0, 0.05):
            print(which, "noise", noise, snapshot_exactness(lambda: E(action_noise=noise), FIXTURE_INIT, {"mu": 0.3}))

