import json, os, numpy as np, pytest
from shared.fixture_env import FixturePushEnv, FIXTURE_INIT
from evaluation import metrics as M
from evaluation.offline_rank import rank_candidates
from evaluation import audit as A

def test_pairwise_ties_and_perfect():
    e = [0.1, 0.2, 0.3, 0.3005]
    acc, n, ties = M.pairwise_accuracy(e, [4, 3, 2, 1])
    assert ties == 1 and n == 5 and acc == 1.0
    acc, _, _ = M.pairwise_accuracy(e, [1, 2, 3, 4]); assert acc == 0.0
    acc, _, _ = M.pairwise_accuracy(e, [0, 0, 0, 0]); assert acc == 0.5

def test_regret_tie_break_expected():
    assert M.top1_regret([0.1, 0.3], [1, 1]) == pytest.approx(0.1)
    assert M.top1_regret([0.1, 0.3], [2, 1]) == 0.0

def test_cluster_bootstrap_not_inflated_by_duplicates():
    # duplicating units within a cluster must not shrink the CI the way independent samples would
    rng = np.random.default_rng(0); base = rng.normal(0, 1, 10)
    _, lo1, hi1 = M.cluster_bootstrap({i: [v] for i, v in enumerate(base)})
    _, lo2, hi2 = M.cluster_bootstrap({i: [v] * 20 for i, v in enumerate(base)})
    assert abs((hi2 - lo2) - (hi1 - lo1)) < 0.25 * (hi1 - lo1)

def test_offline_rank_restores_env_and_crn():
    env = FixturePushEnv(action_noise=0.02); env.reset(FIXTURE_INIT, {"mu": 0.3}, 0); env.step(np.array([0.1, 0]))
    s = env.snapshot()
    C = np.stack([np.tile([v, 0], (4, 1)) for v in [0.2, 0.5]])
    r1 = rank_candidates(env, s, C); r2 = rank_candidates(env, s, C)
    assert r1["eval_err"] == r2["eval_err"]
    assert A._deep_equal(env.snapshot(), s) == []
    assert set(r1["select_rep_seeds"]).isdisjoint(r1["eval_rep_seeds"])

def test_snapshot_exactness_fixture():
    for n in (0.0, 0.05):
        assert A.snapshot_exactness(lambda: FixturePushEnv(action_noise=n), FIXTURE_INIT, {"mu": 0.3})["issues"] == []

def test_snapshot_exactness_detects_partial_snapshot():
    class Bad(FixturePushEnv):
        def snapshot(self):
            s = super().snapshot(); s.pop("rng"); return s
        def restore(self, s):
            self.p, self.o, self.ov = s["p"].copy(), s["o"].copy(), s["ov"].copy(); self.t = s["t"]; self.done = s["done"]
    assert A.snapshot_exactness(lambda: Bad(action_noise=0.05), FIXTURE_INIT, {"mu": 0.3})["issues"]

def test_runtime_guard_flags_env_and_hidden():
    env = FixturePushEnv(); env.reset(FIXTURE_INIT, {"mu": 0.3}, 0)
    class P:
        def propose(self, obs, hist, seed): return np.zeros((1, 1, 2))
    g = A.GuardedPolicy(P())
    g.propose(env.observe(), [], 0); assert g.violations == []
    g.propose({**env.observe(), "mu": 0.3}, [], 0); assert g.violations
    g2 = A.GuardedPolicy(P()); g2.propose(env.observe(), [env], 0); assert g2.violations

def test_tracked_hidden_flags_only_policy_frames(tmp_path, monkeypatch):
    env = FixturePushEnv(); env.reset(FIXTURE_INIT, {"mu": 0.3}, 0); A.track_env_hidden(env)
    A.TrackedHidden.violations = []
    env.step(np.array([0.3, 0.0])); env.snapshot()
    assert A.TrackedHidden.violations == []          # env-internal access is fine
    pdir = os.path.join(A.ROOT, "src", "probing"); fn = os.path.join(pdir, "_a10_fake_policy.py")
    code = compile("def leak(env):\n    return env.hidden['mu']\n", fn, "exec"); ns = {}; exec(code, ns)
    ns["leak"](env)
    assert A.TrackedHidden.violations and "src/probing" in A.TrackedHidden.violations[0]["where"]

def test_online_accounting_detects_restore_and_budget():
    def rec(i, before, after, kind="task", n=1, term=None):
        return {"method": "m", "episode_id": "e", "step_index": i, "kind": kind, "executed": [[0.1, 0]] * n,
                "obs_before": before, "obs_after": after, "termination": term}
    o = lambda x, t: {"t": t, "pusher": [0, 0], "obj": [x, 0], "obj_vel": [0, 0], "goal": [1, 0], "goal_radius": 0.05}
    good = [rec(0, o(0, 0), o(0.1, 0.3), "probe", 3), rec(1, o(0.1, 0.3), o(0.2, 0.5), "task", 2, "timeout")]
    assert A.online_accounting(good, {"max_steps_total": 60, "max_probe_steps": 9}) == []
    bad = [rec(0, o(0, 0), o(0.1, 0.3), "probe", 3), rec(1, o(0, 0), o(0.2, 0.2), "task", 2, "timeout")]
    iss = A.online_accounting(bad, {"max_steps_total": 60, "max_probe_steps": 2})
    assert any("discontinuity" in s for s in iss) and any("probe steps" in s for s in iss)

def test_frozen_sets_hash_and_hygiene():
    proto = open(os.path.join(A.ROOT, "evaluation", "PROTOCOL.md")).read()
    assert A.frozen_sha() in proto
    assert A.split_hygiene() == []

def _e2_eps(cc=True, extra_push=False, requested=4):
    o = lambda x, t: {"t": t, "pusher": [0, 0], "obj": [x, 0], "obj_vel": [0, 0], "goal": [1, 0], "goal_radius": 0.03}
    C = [[[0.1, 0]] * 2, [[0.2, 0]] * 2, [[0.3, 0]] * 2, [[0.4, 0]] * 2]
    base = {"method": "m", "episode_id": "e"}
    r0 = {**base, "step_index": 0, "kind": "probe", "executed": [[0.4, 0]] * 5, "obs_before": o(0, 0), "obs_after": o(.1, .5), "termination": None}
    r1 = {**base, "step_index": 1, "kind": "task", "executed": C[2], "candidates": C, "scores": [0, 1, 2, 1], "chosen_index": 2,
          "obs_before": o(.1, .5), "obs_after": o(.2, .7), "termination": None,
          "candidate_counts": {"requested": requested, "returned": 4, "failed": 0, "seeds": [1]} if cc else None}
    r2 = {**base, "step_index": 2, "kind": "task", "executed": [[0.1 if extra_push else 0, 0]] * 3, "candidates": None,
          "obs_before": o(.2, .7), "obs_after": o(.2, 1.0), "termination": "budget"}
    return [r0, r1, r2]

B3 = {"max_steps_total": 60, "max_probe_steps": 5, "max_probes": 1, "max_task_decisions": 1, "max_task_segment_steps": 12}

def test_e2_structure_and_candidate_counts():
    assert A.online_accounting(_e2_eps(), B3) == []
    assert any("candidate_counts missing" in s for s in A.online_accounting(_e2_eps(cc=False), B3))
    assert any("requested" in s for s in A.online_accounting(_e2_eps(requested=8), B3))
    assert any("corrective" in s for s in A.online_accounting(_e2_eps(extra_push=True), B3))

def test_lcb_sample_check():
    from shared.interface import Prediction
    assert A.lcb_sample_check(Prediction(endpoints=np.zeros((3, 2)), samples=np.zeros((3, 1, 2))))
    assert A.lcb_sample_check(Prediction(endpoints=np.zeros((3, 2)), samples=np.zeros((3, 4, 2)))) == []

def test_tracked_hidden_flags_env_rollout_inside_predict():
    env = FixturePushEnv(); env.reset(FIXTURE_INIT, {"mu": 0.3}, 0); A.track_env_hidden(env)
    A.TrackedHidden.violations = []
    fn = os.path.join(A.ROOT, "src", "dynamics", "_a10_fake_pred.py"); ns = {}
    exec(compile("import numpy as np\ndef predict(env):\n    env.step(np.array([0.1,0.0]))\n", fn, "exec"), ns)
    ns["predict"](env)
    assert A.TrackedHidden.violations and "inside predict" in A.TrackedHidden.violations[0]["where"]
    A.TrackedHidden.violations = []
    exec(compile("import numpy as np\ndef run_loop(env):\n    env.step(np.array([0.1,0.0]))\n", os.path.join(A.ROOT, "src", "probing", "_a10_fake_loop.py"), "exec"), ns)
    ns["run_loop"](env); assert A.TrackedHidden.violations == []

def test_sealed_loader_controller_only_and_logged(tmp_path, monkeypatch):
    monkeypatch.setattr(A, "ACCESS_LOG", str(tmp_path / "log.jsonl")); n0 = 0
    with pytest.raises(PermissionError):
        A.load_sealed("A6")
    rows = [json.loads(l) for l in open(A.ACCESS_LOG)][n0:]
    assert rows and rows[-1]["sealed"] and rows[-1]["allowed"] is False
    assert A.frozen_sha() != A.SEALED_SHA
    with open(A.ACCESS_LOG, "a") as f:
        f.write(json.dumps({"time": "x", "who": "A6", "caller": "src/adaptation/x.py:1", "sealed": True}) + "\n")
    assert any("A6" in s for s in A.sealed_access_audit())

def test_a9_terminated_during_probe_valid_and_counted():
    o = lambda x, t: {"t": t, "pusher": [0, 0], "obj": [x, 0], "obj_vel": [0, 0], "goal": [1, 0], "goal_radius": 0.03}
    ep = lambda eid, term: [{"method": "m", "episode_id": eid, "init_state_id": "s", "hidden_condition_id": "c", "seed": 1, "step_index": 0, "kind": "probe", "executed": [[0.4, 0]] * 3,
                             "obs_before": o(0, 0), "obs_after": o(.1, .3), "termination": term}]
    recs = ep("a", "success") + ep("b", "out_of_bounds") + _e2_eps()
    assert A.online_accounting(recs, B3) == []
    ec = A.episode_classes(recs)["m"]
    assert ec["n_terminated_during_probe"] == 2 and ec["tdp_by_termination"] == {"success": 1, "out_of_bounds": 1}
    assert ec["probe_rate"] == 1.0 and ec["n_episodes"] == 3
    # probe that ends WITHOUT env termination and no task decision is still invalid
    bad = ep("c", "timeout")
    assert any("task decisions" in s for s in A.online_accounting(bad, B3))
    eps = M.records_to_episodes(ep("a", "success"))
    assert eps[0]["success"] and eps[0]["physical_time_s"] == 0.3
