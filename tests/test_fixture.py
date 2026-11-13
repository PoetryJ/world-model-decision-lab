import numpy as np
from shared.fixture_env import FixturePushEnv, FIXTURE_INIT, FIXTURE_HIDDEN

def run(env, acts):
    return [r.obs["obj"].copy() for r in env.step_segment(acts)]

def test_snapshot_replay_exact():
    env = FixturePushEnv(action_noise=0.05)
    env.reset(FIXTURE_INIT, FIXTURE_HIDDEN[1], seed=7)
    env.step_segment(np.tile([0.3, 0.0], (3, 1)))
    s = env.snapshot()
    a = run(env, np.tile([0.4, 0.05], (5, 1)))
    env.restore(s)
    b = run(env, np.tile([0.4, 0.05], (5, 1)))
    assert np.array_equal(np.array(a), np.array(b))

def test_hidden_changes_outcome_same_obs():
    outs = []
    for h in FIXTURE_HIDDEN:
        env = FixturePushEnv(); o0 = env.reset(FIXTURE_INIT, h, seed=0)
        assert "mu" not in o0
        env.step_segment(np.tile([0.4, 0.0], (4, 1))); env.step_segment(np.zeros((15, 2)))
        outs.append(env.observe()["obj"][0])
    assert outs[0] > outs[1] > outs[2]
