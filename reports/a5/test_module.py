"""Lightweight module tests for A5 proposers / planner / freeze (fixture env)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path("/workspace/wmprobe")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "reports/a5"))

from shared.fixture_env import FIXTURE_HIDDEN, FIXTURE_INIT, FixturePushEnv
from shared.interface import V_MAX
from candidates import FixedLibraryProposer, RandomProposer, observation_from_init
from candidates.utils import mean_speed, pairwise_action_distance, segment_rms
from planning import BestOfNPlanner, GoalDistanceScorer, NominalPredictor
from oracle import evaluate_candidates


def test_fixed_shape_legal_comparable():
    prop = FixedLibraryProposer(K=4, H=10)
    obs = observation_from_init(FIXTURE_INIT)
    a = prop.propose(obs, [], seed=0)
    b = prop.propose(obs, [], seed=99)
    assert a.shape == (4, 10, 2)
    assert np.array_equal(a, b), "fixed library must be seed-independent"
    assert np.all(np.abs(a) <= V_MAX + 1e-12)
    spd = mean_speed(a)
    # E1 subset all use NOMINAL_SPEED=0.4; allow tiny drop if a step saturates
    assert np.all(spd > 0.35) and np.all(spd <= 0.5 + 1e-12)
    # not near-identical
    d = pairwise_action_distance(a)
    assert d[0, 1] > 0.02


def test_random_seeded_and_diverse():
    prop = RandomProposer(K=8, H=10)
    obs = observation_from_init(FIXTURE_INIT)
    a = prop.propose(obs, [], seed=510)
    b = prop.propose(obs, [], seed=510)
    c = prop.propose(obs, [], seed=511)
    assert a.shape == (8, 10, 2)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)
    assert np.all(np.abs(a) <= V_MAX + 1e-12)
    d = pairwise_action_distance(a)
    assert np.min(d + np.eye(8)) > 0.05


def test_best_of_n_and_oracle_branching():
    env = FixturePushEnv()
    env.reset(FIXTURE_INIT, FIXTURE_HIDDEN[1], seed=7)
    snap = env.snapshot()
    prop = FixedLibraryProposer(K=4, H=10)
    planner = BestOfNPlanner(prop, NominalPredictor(), GoalDistanceScorer())
    obs = env.observe()
    plan = planner.plan(obs, [], seed=0)
    assert plan["candidates"].shape == (4, 10, 2)
    assert plan["scores"].shape == (4,)
    assert 0 <= plan["chosen_index"] < 4
    env.restore(snap)
    ev = evaluate_candidates(env, plan["candidates"], coast=True, n_repeats=1)
    env.restore(snap)
    ev2 = evaluate_candidates(env, plan["candidates"], coast=True, n_repeats=1)
    assert np.allclose(ev["endpoints"], ev2["endpoints"])
    # env left at snapshot
    assert np.array_equal(env.observe()["obj"], obs["obj"])


def test_k8_library_legal():
    prop = FixedLibraryProposer(K=8, H=5)
    a = prop.propose(observation_from_init(FIXTURE_INIT), [], 0)
    assert a.shape == (8, 5, 2)
    assert np.all(np.abs(a) <= V_MAX + 1e-12)


if __name__ == "__main__":
    test_fixed_shape_legal_comparable()
    test_random_seeded_and_diverse()
    test_best_of_n_and_oracle_branching()
    test_k8_library_legal()
    test_no_overshoot_oscillation()
    test_timed_release_shape()
    print("ok")


def test_no_overshoot_oscillation():
    from candidates.utils import go_to_then_push
    p0 = np.array([-0.25, 0.0])
    waypoint = np.array([-0.23, 0.0])  # 0.02 m away; speed*DT=0.04 would overshoot
    acts = go_to_then_push(p0, waypoint, np.array([1.0, 0.0]), H=6, speed=0.4)
    # no exact reversals
    dots = np.sum(acts[1:] * acts[:-1], axis=-1)
    assert np.all(dots >= -1e-9), acts


def test_timed_release_shape():
    from candidates import TimedReleaseProposer, observation_from_init
    from shared.fixture_env import FIXTURE_INIT
    prop = TimedReleaseProposer(K=4, H=6, speed=0.4)
    a = prop.propose(observation_from_init(FIXTURE_INIT), [], 0)
    assert a.shape == (4, 6, 2)
    # contact durations differ; later steps of shorter ones are zero
    n_nonzero = [int(np.sum(np.linalg.norm(a[k], axis=-1) > 1e-9)) for k in range(4)]
    assert n_nonzero == sorted(n_nonzero)
    assert n_nonzero[0] < n_nonzero[-1]
    assert n_nonzero[-1] == 6
