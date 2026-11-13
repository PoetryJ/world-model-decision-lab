#!/usr/bin/env python3
"""Unit tests for A7 scorers / stub / calibration / gate. Run:
  cd /workspace/wmprobe && . .venv/bin/activate
  PYTHONPATH=/workspace/wmprobe python -m src.scoring.test_scoring
"""
from __future__ import annotations
import sys
import numpy as np
from shared.interface import Prediction
from shared.fixture_env import FixturePushEnv, FIXTURE_INIT, FIXTURE_HIDDEN
from src.scoring.physics_stub import rollout_segment
from src.scoring.stub_predictor import WrongMuStubPredictor
from src.scoring.scorers import (
    GoalDistanceScorer, SuccessProbScorer, LCBScorer, official_aggregate,
    score_detailed, choose_index, dist_to_value, VALUE_SCALE, per_sample_values,
)
from src.scoring.calibration import reliability_table, expected_calibration_error, fit_logistic, logistic_prob
from src.scoring.gate import LowConfidenceGate, extract_feature
from src.scoring.metrics import pairwise_ranking_accuracy, top1_regret
from src.scoring.run_study import make_candidates, oracle_rollouts


def _check(cond, msg):
    if not cond:
        raise AssertionError(msg)


def test_stub_matches_fixture_at_assumed_mu():
    hidden = {"mu": 0.3}
    env = FixturePushEnv(action_noise=0.0)
    env.reset(FIXTURE_INIT, hidden, seed=0)
    cand = make_candidates()
    # execute first candidate in env
    snap = env.snapshot()
    rs = env.step_segment(cand[0])
    env_ep = rs[-1].obs["obj"]
    env.restore(snap)
    obs = env.observe()
    # stub with assumed_mu=0.3, no perturb, 1 sample, no noise
    pred = WrongMuStubPredictor(assumed_mu=0.3, n_samples=1, mu_perturb=0.0,
                                mu_perturb_mode="none", endpoint_noise=0.0, seed=0)
    P = pred.predict(obs, cand[0:1], {"seed": 0})
    _check(np.allclose(P.endpoints[0], env_ep, atol=1e-12),
           f"stub/env mismatch: stub={P.endpoints[0]} env={env_ep}")


def test_stub_does_not_read_hidden():
    """Same observation, different env hidden, stub with fixed mu must be identical."""
    cand = make_candidates()
    preds = []
    for h in FIXTURE_HIDDEN:
        env = FixturePushEnv()
        obs = env.reset(FIXTURE_INIT, h, seed=1)
        _check("mu" not in obs, "observation leaked mu")
        pred = WrongMuStubPredictor(assumed_mu=0.3, n_samples=5, mu_perturb=0.1,
                                    mu_perturb_mode="fixed_grid", seed=11)
        # even if someone stuffed hidden into context, predictor must ignore it
        P = pred.predict(obs, cand, {"seed": 11, "hidden": h, "env": env})
        preds.append(P.endpoints.copy())
    for i in range(1, len(preds)):
        _check(np.allclose(preds[0], preds[i], atol=1e-12),
               "stub endpoints changed when env hidden changed")


def test_goal_distance_ranks_closer_higher():
    g = {"goal": np.array([0.0, 0.0]), "goal_radius": 0.05}
    P = Prediction(endpoints=np.array([[0.1, 0.0], [0.3, 0.0], [0.0, 0.0]]))
    s = GoalDistanceScorer().score(P, g)
    _check(s[2] > s[0] > s[1], f"ranking {s}")
    _check(choose_index(s) == 2, "argmax")


def test_success_prob_distinct_from_distance():
    g = {"goal": np.array([0.0, 0.0]), "goal_radius": 0.05}
    # cand 0: all samples just inside; cand 1: one sample at 0, others far
    # mean distance of cand 1 may be worse, but success_prob of cand 0 is 1 vs 1/3
    samples = np.array([
        [[0.01, 0], [0.02, 0], [0.03, 0]],
        [[0.00, 0], [0.40, 0], [0.40, 0]],
    ])
    P = Prediction(endpoints=samples.mean(1), samples=samples)
    bun = score_detailed(P, g, score_name="success_prob")
    _check(bun.success_prob[0] == 1.0 and abs(bun.success_prob[1] - 1 / 3) < 1e-12, bun.success_prob)
    _check(bun.mean_distance[1] > bun.mean_distance[0], "mean dist")
    # distance scorer and success_prob agree here, but values differ
    _check(bun.score_name == "success_prob", "name")


def test_lcb_official_nan_on_one_sample():
    vals = np.array([[0.8], [0.1]])  # K=2,S=1
    off = official_aggregate(vals, "lcb", safe_single_sample=False)
    safe = official_aggregate(vals, "lcb", safe_single_sample=True)
    _check(np.all(np.isnan(off)), f"official S=1 LCB should be NaN, got {off}")
    _check(np.allclose(safe, [0.8, 0.1]), f"safe S=1 should equal mean, got {safe}")


def test_lcb_penalises_disagreement():
    # two candidates, same mean 0.5; second has higher variance
    vals = np.array([
        [0.5, 0.5, 0.5, 0.5],
        [0.9, 0.1, 0.9, 0.1],
    ])
    lcb = official_aggregate(vals, "lcb", safe_single_sample=False)
    _check(lcb[0] > lcb[1], f"LCB should prefer tight cand, got {lcb}")
    # mean - std; cand0 std=0 => 0.5; cand1 std>0 => <0.5
    _check(np.allclose(lcb[0], 0.5), lcb[0])


def test_majority_mean_scale():
    # v>0.05 almost always on shaped distances of 0.1 m
    d = np.array([[0.10, 0.12, 0.08, 0.11]])  # v ≈ 0.8 >> 0.05
    v = dist_to_value(d, VALUE_SCALE)
    maj = official_aggregate(v, "majority_mean", success_threshold=0.05)
    avg = official_aggregate(v, "average")
    _check(np.allclose(maj, avg), f"official 0.05 majority should collapse to mean, {maj} vs {avg}")
    # at radius threshold 0.9, all v=0.8 < 0.9 => majority failure => 0
    thr = float(dist_to_value(np.array([0.05]))[0])
    maj_r = official_aggregate(v, "majority_mean", success_threshold=thr)
    _check(np.allclose(maj_r, [0.0]), f"radius-threshold majority should be 0, got {maj_r} thr={thr}")


def test_scorer_ignores_env_hidden_in_goal_dict():
    """Only goal and goal_radius are used; extra keys must be ignored."""
    P = Prediction(endpoints=np.array([[0.1, 0.0], [0.2, 0.0]]))
    s1 = GoalDistanceScorer().score(P, {"goal": np.array([0.0, 0.0]), "goal_radius": 0.05})
    s2 = GoalDistanceScorer().score(P, {"goal": np.array([0.0, 0.0]), "goal_radius": 0.05, "mu": 0.1, "hidden": {"mu": 0.9}})
    _check(np.allclose(s1, s2), "scorer used unexpected keys")


def test_ece_perfect_and_worst():
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 1, 4000)
    # perfectly calibrated synthetic y ~ Bern(p)
    y = (rng.uniform(0, 1, 4000) < p).astype(float)
    ece = expected_calibration_error(p, y, n_bins=10, strategy="uniform")
    _check(ece < 0.05, f"calibrated ECE too high: {ece}")
    # perfectly miscalibrated: p=0.9 always, y=0
    ece2 = expected_calibration_error(np.full(1000, 0.9), np.zeros(1000), n_bins=10)
    _check(abs(ece2 - 0.9) < 1e-12, ece2)


def test_gate_fits_val_only_and_catches_high_feature():
    # feature = 0 for correct, 1 for wrong — a perfect gate threshold in (0,1]
    f = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0])
    w = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=float)
    g = LowConfidenceGate.fit(f, w, "value_std", probe_cost=0.4, fitted_on="val")
    _check(g.fitted_on == "val", g.fitted_on)
    # abstain iff feature >= t, t should be 1.0 (or just at the wrong cluster)
    _check(g.threshold <= 1.0 and g.threshold > 0.0, g.threshold)
    from src.scoring.gate import evaluate_gate
    ev = evaluate_gate(g, f, w)
    _check(ev["n_missed_wrong"] == 0, ev)
    _check(ev["n_false_abstain"] == 0, ev)


def test_pairwise_ties():
    # equal true distances: ignored in pairwise; top-1 either is correct
    s = np.array([0.5, 0.2, 0.1])
    d = np.array([0.10, 0.10, 0.30])
    pw = pairwise_ranking_accuracy(s, d, tie_tol=1e-4)
    _check(pw["n_oracle_ties"] == 1, pw)
    _check(pw["n_compared"] == 2, pw)  # (0,2) and (1,2)
    _check(abs(pw["pairwise_acc"] - 1.0) < 1e-12, pw)  # both 0 and 1 beat 2, and scores agree
    r0 = top1_regret(0, d)
    r1 = top1_regret(1, d)
    _check(r0["top1_correct"] and r1["top1_correct"], (r0, r1))
    r2 = top1_regret(2, d)
    _check(not r2["top1_correct"] and r2["regret_m"] > 0, r2)


def test_oracle_branching_replay():
    env = FixturePushEnv(action_noise=0.0)
    cand = make_candidates()
    a = oracle_rollouts(env, FIXTURE_INIT, {"mu": 0.1}, seed=3, candidates=cand)
    b = oracle_rollouts(env, FIXTURE_INIT, {"mu": 0.1}, seed=3, candidates=cand)
    _check(np.array_equal(a["true_ep"], b["true_ep"]), "oracle not deterministic")
    # different mu must change at least some endpoints for push-coast cands
    c = oracle_rollouts(env, FIXTURE_INIT, {"mu": 0.6}, seed=3, candidates=cand)
    _check(not np.allclose(a["true_ep"], c["true_ep"], atol=1e-6), "mu should change coast endpoints")


def main():
    tests = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in tests:
        try:
            fn()
            print(f"  PASS {fn.__name__}")
        except Exception as e:
            failed += 1
            print(f"  FAIL {fn.__name__}: {type(e).__name__}: {e}")
    print(f"{len(tests)-failed}/{len(tests)} passed")
    return 1 if failed else 0




def test_worst_case_and_robust_success():
    g = {"goal": np.array([0.0, 0.0]), "goal_radius": 0.05}
    # cand 0: all samples inside; cand 1: mean close but one far miss
    samples = np.array([
        [[0.01, 0], [0.02, 0], [0.00, 0]],
        [[0.00, 0], [0.00, 0], [0.40, 0]],
    ])
    P = Prediction(endpoints=samples.mean(1), samples=samples)
    wc = score_detailed(P, g, score_name="worst_case_distance")
    _check(wc.scores[0] > wc.scores[1], wc.scores)
    rs = score_detailed(P, g, score_name="robust_success")
    _check(rs.scores[0] == 1.0 and rs.scores[1] == 0.0, rs.scores)
    # greedy mean distance may prefer cand 1 if mean is closer... cand1 mean is 0.133, cand0 mean 0.01
    # so distance also prefers 0 here. Build a case where mean prefers the fragile one:
    samples2 = np.array([
        [[0.04, 0], [0.04, 0], [0.04, 0]],          # all just inside, mean 0.04
        [[0.00, 0], [0.00, 0], [0.20, 0]],          # mean 0.0667 worse actually
    ])
    # need fragile with BETTER mean: [0.01, 0.01, 0.20] mean 0.073 vs safe 0.04 — still worse.
    # safe 0.049 all inside (mean 0.049); fragile [0.01, 0.01, 0.10] mean 0.04 better mean, not robust
    samples3 = np.array([
        [[0.049, 0], [0.049, 0], [0.049, 0]],
        [[0.010, 0], [0.010, 0], [0.090, 0]],
    ])
    P3 = Prediction(endpoints=samples3.mean(1), samples=samples3)
    gd = score_detailed(P3, g, score_name="goal_distance")
    rst = score_detailed(P3, g, score_name="robust_success_then_distance")
    _check(choose_index(gd.scores) == 1, f"greedy should pick fragile mean, {gd.scores} {gd.mean_distance}")
    _check(choose_index(rst.scores) == 0, f"robust should pick safe, {rst.scores}")


def test_pushenv_stub_matches_branch_eval():
    from src.env.push_env import PushEnv
    from src.env import scenarios as S
    from src.env.branch_eval import evaluate_candidates
    from src.scoring.pushenv_stub import PushEnvStubPredictor
    sid, init = S.e0_set(True)[0]
    env = PushEnv()
    obs = env.reset(init, {"mu": 0.1}, seed=0)
    cands, _ = S.strike_grid(obs)
    # use a small subset for speed
    cands = cands[::9]
    snap = env.snapshot()
    d, su, fin = evaluate_candidates(env, snap, cands)
    pred = PushEnvStubPredictor(assumed_mu=0.1, n_samples=1, mu_perturb_mode="none", seed=0)
    P = pred.predict(obs, cands, {"seed": 0})
    _check(np.allclose(P.endpoints, fin, atol=1e-12),
           f"stub/oracle endpoint mismatch max {np.max(np.abs(P.endpoints-fin))}")
    # different assumed mu must not use env.hidden (true is 0.1)
    pred_hi = PushEnvStubPredictor(assumed_mu=0.5, n_samples=1, mu_perturb_mode="none", seed=0)
    P2 = pred_hi.predict(obs, cands, {"seed": 0, "hidden": {"mu": 0.1}})
    # at least one candidate should differ after settle (release shots)
    _check(not np.allclose(P.endpoints, P2.endpoints, atol=1e-6),
           "hi-mu stub matched lo-mu endpoints; expected slide difference")

if __name__ == "__main__":
    sys.exit(main())
