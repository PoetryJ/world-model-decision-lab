"""A8 tests: python -m pytest -q src/probing/test_probing.py"""
import numpy as np
from src.env.push_env import PushEnv
from src.env import scenarios as SC
from shared.fixture_env import FixturePushEnv, FIXTURE_INIT
from shared.interface import Transition
from src.probing.stubs import BeliefPredictor, BeliefAdapter, StubScorer
from src.probing.probers import make_prober, ALL
from src.probing.loop import run_episode
from src.probing.belief import MuGridBelief
from src.probing.experiments import StrikeGridProposer

class GuardEnv:
    """Wraps env; counts reset/restore/snapshot; raises if anyone reads .hidden after reset."""
    def __init__(self, env): self._e = env; self.n_reset = self.n_restore = self.n_snap = 0
    @property
    def hidden(self): raise AssertionError("policy-side read of env.hidden")
    def reset(self, init, hidden, seed): self.n_reset += 1; return self._e.reset(init, hidden, seed)
    def observe(self): return self._e.observe()
    def step(self, a): return self._e.step(a)
    def step_segment(self, a): return self._e.step_segment(a)
    def snapshot(self): self.n_snap += 1; return self._e.snapshot()
    def restore(self, s): self.n_restore += 1; return self._e.restore(s)

def _st(neg=False, i=0):
    sid, st = SC.e0_set(negative_control=neg)[i]
    return {k: v for k, v in st.items() if k != "meta"}

def _run(name, env, st, mu=0.25, seed=3, budget=60, pb=9, one_shot=False):
    prop = StrikeGridProposer()
    pr = make_prober(name, proposer=prop, seed=seed)
    return run_episode(env, st, {"mu": mu}, seed, prop, BeliefPredictor(), BeliefAdapter(), StubScorer(), pr,
                       step_budget=budget, probe_budget=pb, max_task_decisions=1 if one_shot else None)

def test_no_hidden_no_reset_budget():
    for name in ALL:
        g = GuardEnv(PushEnv())
        s, recs = _run(name, g, _st(), budget=20, one_shot=True)
        assert g.n_reset == 1 and g.n_restore == 0 and g.n_snap == 0
        assert s["steps_used"] == sum(len(r["executed"]) for r in recs) <= 20
        assert s["probe_steps"] == sum(len(r["executed"]) for r in recs if r["kind"] == "probe") <= 9
        # continuity: each decision starts where the previous one ended (no reset after probe)
        for a, b in zip(recs, recs[1:]):
            assert np.allclose(a["obs_after"]["obj"], b["obs_before"]["obj"])

def test_deterministic():
    a = _run("task_voi_probe", PushEnv(), _st(), one_shot=True)[1]
    b = _run("task_voi_probe", PushEnv(), _st(), one_shot=True)[1]
    assert [r["executed"] for r in a] == [r["executed"] for r in b]
    ra = _run("random_probe", PushEnv(), _st(), seed=5)[1]; rb = _run("random_probe", PushEnv(), _st(), seed=5)[1]
    assert [r["executed"] for r in ra] == [r["executed"] for r in rb]

def test_belief_identifies_mu_after_tap():
    st = _st(); bel = MuGridBelief()
    for mu in (0.1, 0.25, 0.5):
        env = PushEnv(); o = env.reset(st, {"mu": mu}, 0)
        seg = SC.probe_set(o)["tap_fast"]; env.step_segment(seg)
        w = bel.posterior([Transition(o, seg, env.observe(), "probe")])
        pm = float(w @ bel.grid)
        assert abs(pm - mu) / mu < 0.25, (mu, pm)  # grid is geometric, 25 pts in [0.05,1]

def test_uninformative_probes_zero_gain():
    st = _st(); env = PushEnv(); o = env.reset(st, {"mu": 0.25}, 0)
    pr = make_prober("infogain_probe"); pr.choose_probe(o, [], {"steps_left": 60, "probe_steps_left": 9})
    assert pr.last["gains"]["retreat"] < 1e-6 and pr.last["gains"]["push_slow"] < 0.05
    assert pr.last["gains"]["tap_fast"] > 0.5

def test_fixture_env_runs():
    s, _ = _run("task_voi_probe", FixturePushEnv(), FIXTURE_INIT, one_shot=True)
    assert s["steps_used"] > 0

def test_candidate_counts_recorded():
    s, recs = _run("no_probe", PushEnv(), _st(), one_shot=True)
    t = [r for r in recs if r["kind"] == "task" and r["candidates"] is not None]
    assert t and all(r["candidate_counts"]["returned"] == len(r["candidates"]) for r in t)
    assert all(r["candidate_counts"]["failed"] == 0 and len(r["candidate_counts"]["seeds"]) >= 1 for r in t)
