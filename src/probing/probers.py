"""Five probe selectors implementing shared.interface.Prober.choose_probe(obs, history, budget).
budget = {"steps_left": int, "probe_steps_left": int}. Return (h,2) segment or None.
None of them reads env.hidden; belief uses visible history only.
Classic problems: (4) active system identification / max expected information gain (Bayesian experimental
design); (5) value of information / dual control (preposterior analysis, myopic 1-step lookahead).
Each prober exposes .last = dict with its internal decision quantities (logged by loop.py).
"""
from __future__ import annotations
import numpy as np
from .belief import MuGridBelief, entropy, last_command
from .physics_model import BatchPushModel
from . import probe_library as PL

def _fits(seg, budget):
    h = len(seg)
    return budget.get("probe_steps_left", 0) >= h and budget.get("steps_left", 0) > h

def default_library():
    """A4's probe_set (tap_fast / push_slow / retreat, h=5) if available, else local library."""
    try:
        from src.env.scenarios import probe_set
        return probe_set
    except Exception:
        return PL.library

class NoProbe:
    name = "no_probe"
    def __init__(self): self.last = {}
    def choose_probe(self, observation, history, budget):
        self.last = {"reason": "never"}; return None

class FixedProbe:
    """Always tap_fast once at the start of the episode (if budget allows)."""
    name = "fixed_probe"
    def __init__(self, probe_name="tap_fast", max_probes=1, library=None):
        self.probe_name, self.max_probes, self.last = probe_name, max_probes, {}
        self.lib = library or default_library()
    def choose_probe(self, observation, history, budget):
        n = sum(t.kind == "probe" for t in history)
        seg = self.lib(observation)[self.probe_name]
        if n >= self.max_probes or not _fits(seg, budget):
            self.last = {"reason": "done_or_budget"}; return None
        self.last = {"reason": "fixed", "probe": self.probe_name}; return seg

class RandomProbe:
    """Uniform over library (incl. uninformative no_contact), one probe at episode start. Seeded."""
    name = "random_probe"
    def __init__(self, seed: int, max_probes=1, library=None):
        self.seed, self.max_probes, self.last = seed, max_probes, {}
        self.lib = library or default_library()
    def choose_probe(self, observation, history, budget):
        n = sum(t.kind == "probe" for t in history)
        lib = self.lib(observation); names = sorted(lib)
        # seed derived from (self.seed, #transitions so far) -> reproducible, independent of call count
        rng = np.random.default_rng([self.seed, len(history)])
        name = names[int(rng.integers(len(names)))]
        if n >= self.max_probes or not _fits(lib[name], budget):
            self.last = {"reason": "done_or_budget"}; return None
        self.last = {"reason": "random", "probe": name}; return lib[name]

class _BeliefProber:
    def __init__(self, belief: MuGridBelief | None = None, top_m: int = 12, library=None):
        self.belief = belief or MuGridBelief(); self.model = self.belief.model
        self.lib = library or default_library()
        self.top_m, self.last = top_m, {}
    def _hyp(self, history):
        w = self.belief.posterior(history)
        idx = np.sort(np.argsort(-w)[: self.top_m])
        return self.belief.grid[idx], w[idx] / w[idx].sum(), w
    def _sim_probe(self, obs, seg, mus, pv0):
        M = len(mus)
        return self.model.rollout(np.tile(obs["pusher"], (M, 1)), np.tile(obs["obj"], (M, 1)),
                                  np.tile(obs["obj_vel"], (M, 1)), mus, np.tile(seg, (M, 1, 1)), pv=pv0)
    def _post_after(self, obs, seg, mus, ww, pv0):
        """For each hypothetical true mu_i: noise-free outcome y_i; posterior over mus given y_i. (M,M)"""
        r = self._sim_probe(obs, seg, mus, pv0)
        ll = self.belief.loglik_obs(r["o"], r["ov"], r["o"], r["ov"])  # (N=M outcomes, M hyps)
        lp = ll + np.log(ww)[None]
        lp -= lp.max(axis=1, keepdims=True)
        post = np.exp(lp); post /= post.sum(axis=1, keepdims=True)
        return r, post

class InfoGainProbe(_BeliefProber):
    """(4) Pick probe maximising expected entropy reduction of mu posterior; probe if gain > min_gain nats.
    Task-agnostic: will probe even when mu does not matter for the task (expected failure on negative control)."""
    name = "infogain_probe"
    def __init__(self, min_gain: float = 0.1, **kw):
        super().__init__(**kw); self.min_gain = min_gain
    def choose_probe(self, observation, history, budget):
        mus, ww, _ = self._hyp(history); pv0 = last_command(history)
        H0 = entropy(ww); gains = {}
        for name, seg in self.lib(observation).items():
            if not _fits(seg, budget): continue
            _, post = self._post_after(observation, seg, mus, ww, pv0)
            gains[name] = H0 - float(sum(ww[i] * entropy(post[i]) for i in range(len(mus))))
        if not gains:
            self.last = {"reason": "budget", "H0": H0}; return None
        best = max(gains, key=gains.get)
        self.last = {"H0": H0, "gains": gains, "best": best}
        if gains[best] <= self.min_gain:
            self.last["reason"] = "gain_below_threshold"; return None
        self.last["reason"] = "infogain"; return self.lib(observation)[best]

class TaskVOIProbe(_BeliefProber):
    """(5) Myopic value of information (dual control, preposterior analysis).
    V_now   = min_k E_{mu~b}[cost_k(mu | s_now)]                         (act now under current belief)
    V_probe = E_{mu_i~b}[ min_k E_{mu~b'_i}[cost_k(mu | s_p(mu_i))] ]   (probe, observe y_i, re-plan from the
              ACTUAL post-probe state s_p(mu_i) -- includes object displacement caused by the probe)
    probe iff V_now - V_probe > lam * h  (lam = cost per probe control step in metres-of-goal-error units).
    Candidates come from the injected proposer (same one the task uses), truncated to remaining steps.
    Cost = max(||rest - goal|| - r, 0) + 1[out of bounds]; ties inside goal give zero VOI (negative control)."""
    name = "task_voi_probe"
    def __init__(self, proposer, lam: float = 0.01, seed: int = 0, value: str = "success", hinge_w: float = 0.1,
                 tie_w: float = 0.0, **kw):
        # value="success": realised cost = 1[rest outside goal or oob] + hinge_w*hinge; lam in P(success) per step.
        # value="hinge"  : realised cost = hinge metres (v0, abandoned after E0 dev run: VOI ~5 mm << lam*h).
        super().__init__(**kw); self.proposer, self.lam, self.seed = proposer, lam, seed
        self.value, self.hinge_w, self.tie_w = value, hinge_w, tie_w
        self.probe_tie_tol = 1e-6
    def _cost_matrix(self, obs, cands, mus, pv0, goal, r_goal):
        """cost (K,M) for each candidate under each mu from observed state obs."""
        K, M = len(cands), len(mus); B = K * M
        r = self.model.rollout(np.tile(obs["pusher"], (B, 1)), np.tile(obs["obj"], (B, 1)),
                               np.tile(obs["obj_vel"], (B, 1)), np.tile(mus, K), np.repeat(cands, M, axis=0), pv=pv0)
        rest, _ = self.model.settle(r["o"], r["ov"], np.tile(mus, K))
        dist = np.linalg.norm(rest - goal, axis=1)
        hinge = np.maximum(dist - r_goal, 0.0) + r["oob"]
        # decision cost = hinge (same objective as StubScorer); value cost = task failure (+ small hinge)
        fail = (hinge > 0).astype(float) + self.hinge_w * hinge if self.value == "success" else hinge
        dec = hinge + self.tie_w * dist     # must equal the task scorer's objective
        return dec.reshape(K, M), fail.reshape(K, M)
    def _cands(self, obs, history, steps_left):
        C = np.asarray(self.proposer.propose(obs, history, self.seed), float)
        return C[:, :max(1, min(C.shape[1], steps_left))]
    def choose_probe(self, observation, history, budget):
        mus, ww, _ = self._hyp(history); pv0 = last_command(history)
        goal, rg = np.asarray(observation["goal"]), float(observation["goal_radius"])
        C = self._cands(observation, history, budget["steps_left"])
        dm, vm = self._cost_matrix(observation, C, mus, pv0, goal, rg)
        k_now = int((dm @ ww).argmin()); V_now = float(vm[k_now] @ ww)
        res = {}
        for name, seg in self.lib(observation).items():
            if not _fits(seg, budget): continue
            r, post = self._post_after(observation, seg, mus, ww, pv0)
            vals = []
            for i in range(len(mus)):
                o_i = {"pusher": r["p"][i], "obj": r["o"][i], "obj_vel": r["ov"][i],
                       "goal": goal, "goal_radius": rg, "t": observation["t"] + len(seg) * 0.1}
                Ci = self._cands(o_i, history, budget["steps_left"] - len(seg))
                dmi, vmi = self._cost_matrix(o_i, Ci, mus, r["pv"][i], goal, rg)
                k = int((dmi @ post[i]).argmin())
                vals.append(vmi[k, i])          # realised cost if truth is mu_i (decision used b'_i only)
            V_p = float(np.dot(ww, vals))
            res[name] = {"V_probe": V_p, "voi": V_now - V_p, "net": V_now - V_p - self.lam * len(seg)}
        self.last = {"V_now": V_now, "k_now": k_now, "probes": res}
        if not res:
            self.last["reason"] = "budget"; return None
        # v3 (tuned on VAL, A-8): nets equal within tie_tol -> keep library order (tap_fast first); previously float noise
        # picked push_slow on exact ties (val_pos_012), whose apparent value comes from a substep-friction velocity artefact.
        top = max(r["net"] for r in res.values())
        best = next(n for n in res if res[n]["net"] >= top - self.probe_tie_tol)
        self.last["best"] = best
        if res[best]["net"] <= 0:
            self.last["reason"] = "voi_below_cost"; return None
        self.last["reason"] = "voi"; return self.lib(observation)[best]

class GoalDistanceProbe:
    """Simple hand rule (strong cheap competitor to task_voi): tap once iff visible goal distance > d0.
    d0 = 0.15 m fixed a priori (~ tap_fast displacement under the lowest plausible friction + goal radius), NOT fit."""
    name = "goaldist_probe"
    def __init__(self, d0: float = 0.15, probe_name: str = "tap_fast", library=None):
        self.d0, self.probe_name, self.last = d0, probe_name, {}
        self.lib = library or default_library()
    def choose_probe(self, observation, history, budget):
        d = float(np.linalg.norm(np.asarray(observation["goal"]) - np.asarray(observation["obj"])))
        seg = self.lib(observation)[self.probe_name]
        if any(t.kind == "probe" for t in history) or not _fits(seg, budget) or d <= self.d0:
            self.last = {"reason": "rule_no", "goal_dist": d}; return None
        self.last = {"reason": "rule_yes", "goal_dist": d}; return seg

def make_prober(name, proposer=None, seed=0, **kw):
    if name == "no_probe": return NoProbe()
    if name == "fixed_probe": return FixedProbe(library=kw.get("library"))
    if name == "random_probe": return RandomProbe(seed=seed, library=kw.get("library"))
    if name == "infogain_probe": return InfoGainProbe(**{k: v for k, v in kw.items() if k in ("belief", "min_gain", "top_m", "library")})
    if name == "task_voi_probe": return TaskVOIProbe(proposer=proposer, seed=seed, **kw)
    if name == "goaldist_probe": return GoalDistanceProbe()
    raise KeyError(name)

ALL = ("no_probe", "fixed_probe", "random_probe", "infogain_probe", "task_voi_probe", "goaldist_probe")
