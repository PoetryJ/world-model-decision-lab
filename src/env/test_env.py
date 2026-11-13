"""A4 tests for PushEnv (TOY). Run: cd /workspace/wmprobe && . .venv/bin/activate && pytest -q src/env/test_env.py"""
import pickle
import numpy as np
import pytest
from shared.interface import V_MAX, StepResult
from src.env.push_env import PushEnv, OBS_KEYS
from src.env import scenarios as S

SID, ST = S.e0_set(False)[0]
NOISY = dict(action_noise=0.05, obs_noise=0.003)


def full_trace(env, acts):
    """Everything observable AND internal after each step (bitwise comparison)."""
    tr = []
    for a in acts:
        r = env.step(a)
        o = r.obs
        tr.append(np.concatenate([[o["t"]], o["pusher"], o["obj"], o["obj_vel"], env.p, env.o, env.ov,
                                  env.pv, env.cmd, [env.contact_substeps, env.in_contact, r.done,
                                                    r.success, env.n_steps]]))
        if r.done:
            break
    return np.array(tr), env.rng.random(4)   # also compare RNG continuation


def mixed_actions(seed, n=25):
    g = np.random.default_rng(seed)
    u = np.asarray(ST["goal"]) - np.asarray(ST["obj"]); u /= np.linalg.norm(u)
    acts = np.where(g.random((n, 1)) < 0.5, 0.45 * u, 0.0) + g.normal(0, 0.05, (n, 2))
    return acts


@pytest.mark.parametrize("cond", [c for c, _ in S.hidden_conditions(3)])
@pytest.mark.parametrize("k_pre", [0, 1, 3, 7])   # snapshot before contact, at impact, mid-slide, etc.
def test_snapshot_bitwise_continuation(cond, k_pre):
    h = dict(S.hidden_conditions(3))[cond]
    env = PushEnv(**NOISY)
    env.reset(ST, h, seed=11)
    pre = np.tile(0.4 * (np.asarray(ST["goal"]) - ST["obj"]) / np.linalg.norm(np.asarray(ST["goal"]) - ST["obj"]), (k_pre, 1))
    env.step_segment(pre)
    snap = env.snapshot()
    acts = mixed_actions(5)
    a1, r1 = full_trace(env, acts)
    # same instance restore
    env.restore(snap)
    a2, r2 = full_trace(env, acts)
    # fresh instance restore (no shared state), via pickle round trip
    env3 = PushEnv(); env3.restore(pickle.loads(pickle.dumps(snap)))
    a3, r3 = full_trace(env3, acts)
    assert a1.tobytes() == a2.tobytes() == a3.tobytes()
    assert r1.tobytes() == r2.tobytes() == r3.tobytes()


def test_snapshot_is_deep_copy_and_has_full_state():
    env = PushEnv(**NOISY); env.reset(ST, {"mu": 0.25}, seed=3)
    env.step_segment(np.tile([0.3, 0.0], (3, 1)))
    snap = env.snapshot()
    for k in ("p", "o", "ov", "pv", "cmd", "in_contact", "contact_substeps", "t", "n_steps", "done",
              "success", "termination", "rng", "hidden", "goal", "_last_obs"):
        assert k in snap
    before = pickle.dumps(snap)
    env.step_segment(np.tile([0.3, 0.1], (5, 1)))
    assert pickle.dumps(snap) == before          # stepping does not mutate the snapshot


def test_velocity_and_controller_state_matter():
    """Restoring only positions (not velocities / actuator state / rng) gives a DIFFERENT continuation,
    i.e. the extra snapshot fields are necessary for fair counterfactuals."""
    env = PushEnv(**NOISY); env.reset(ST, {"mu": 0.1}, seed=3)
    u = np.asarray(ST["goal"]) - ST["obj"]; u = u / np.linalg.norm(u)
    env.step_segment(np.tile(0.45 * u, (2, 1)))   # object now sliding fast
    snap = env.snapshot()
    a_full, _ = full_trace(env, np.zeros((5, 2)))
    env.restore(snap); env.ov = np.zeros(2); env.pv = np.zeros(2)
    a_pos, _ = full_trace(env, np.zeros((5, 2)))
    assert not np.array_equal(a_full[:, 5:7], a_pos[:, 5:7])


def test_same_visible_init_obs_across_hidden():
    for seed in (0, 1):
        obs = []
        for _, h in S.hidden_conditions(3):
            env = PushEnv(**NOISY); obs.append(env.reset(ST, h, seed=seed))
        for o in obs[1:]:
            for k in OBS_KEYS:
                assert np.asarray(o[k]).tobytes() == np.asarray(obs[0][k]).tobytes()


def test_no_hidden_leak_in_obs_or_info():
    env = PushEnv(**NOISY); o = env.reset(ST, {"mu": 0.5}, seed=0)
    assert tuple(sorted(o)) == tuple(sorted(OBS_KEYS))
    for r in env.step_segment(np.tile([0.4, 0.0], (10, 1))):
        assert set(r.obs) == set(OBS_KEYS)
        assert "mu" not in r.info and 0.5 not in [v for v in r.info.values() if isinstance(v, float)]


def test_protocol_and_legality():
    env = PushEnv(); env.reset(ST, {"mu": 0.25}, seed=0)
    r = env.step(np.array([5.0, -5.0]))            # clipped
    assert isinstance(r, StepResult) and np.all(np.abs(env.cmd) <= V_MAX)
    res = env.step_segment(np.zeros((3, 2)))
    assert len(res) == 3 and abs(env.observe()["t"] - 0.4) < 1e-12
    for name, p in S.probe_set(env.observe()).items():
        assert p.shape == (S.PROBE_H, 2) and np.all(np.abs(p) <= V_MAX)


def test_contact_kinematics_mu_independent():
    """During continuous contact obj_vel == pusher normal velocity regardless of mu (documented toy property)."""
    outs = []
    for _, h in S.hidden_conditions(3):
        env = PushEnv(); o = env.reset(ST, h, seed=0)
        outs.append(np.array([np.r_[r.obs["obj"], r.obs["obj_vel"]] for r in env.step_segment(S.probe_set(o)["push_slow"])]))
    # equal up to float rounding (~1e-16); mu enters only via rounding of ov reduce-then-raise
    assert np.allclose(outs[0], outs[1], atol=1e-12, rtol=0) and np.allclose(outs[0], outs[2], atol=1e-12, rtol=0)


def test_friction_ordering_after_release():
    finals = []
    for _, h in S.hidden_conditions(3):
        env = PushEnv(); o = env.reset(ST, h, seed=0)
        env.step_segment(S.probe_set(o)["tap_fast"]); env.step_segment(np.zeros((20, 2)))
        finals.append(np.linalg.norm(env.o - np.asarray(ST["obj"])))
    assert finals[0] > finals[1] > finals[2]


def test_terminations():
    env = PushEnv(max_time=0.5); env.reset(ST, {"mu": 0.25}, seed=0)
    res = env.step_segment(np.zeros((10, 2)))
    assert len(res) == 5 and res[-1].info["termination"] == "timeout"
    st = dict(ST); st = {**ST, "goal": ST["obj"]}
    env = PushEnv(); env.reset(st, {"mu": 0.25}, seed=0)
    r = env.step(np.zeros(2)); assert r.success and r.info["termination"] == "success"
    env = PushEnv(); env.reset({"pusher": [0.3, 0.0], "obj": [0.4, 0.0], "goal": [0, 0]}, {"mu": 0.1}, 0)
    res = env.step_segment(np.tile([0.5, 0], (10, 1)))
    assert res[-1].info["termination"] == "out_of_bounds"
