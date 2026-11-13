"""A4 experiment env: PushEnv  --  *** TOY *** CPU 2D mechanism env (NOT a physics simulator, NOT vision).

Owner: A4 (src/env). Implements shared.interface.Env protocol (FROZEN v1) exactly.
Observation = idealised structured pose perception (true poses, optional Gaussian obs noise); it is NOT
a vision/perception result.

Units/frames (from shared/interface.py): world frame metres, x right, y up, origin table centre,
table [-0.5,0.5]^2. Control step DT=0.1 s, physics substep 0.01 s. Action = commanded pusher velocity
(vx,vy) m/s, clipped per-component to [-0.5,0.5]. Segment = (H,2).

Mechanism (toy):
- Pusher: disc r=0.03, velocity-controlled with first-order actuator lag (tau=0.02 s, internal state
  `pv`, part of snapshot). Pusher is "infinitely strong" (kinematic): friction does NOT slow it.
  Pusher centre clamped to the table.
- Object: disc r=0.05, point-mass sliding with Coulomb friction: |a| = mu*g opposite velocity.
- Contact: if discs overlap, object is projected out along contact normal and its normal velocity is
  raised to at least the pusher's normal velocity (perfectly inelastic, object light vs pusher;
  no tangential contact friction).  => While in continuous contact the object follows the pusher
  kinematically, independent of mu.  mu only matters in free-sliding phases (after release / taps).
- Hidden factor (v1): ONLY friction coefficient mu (env.hidden = {"mu": float}).
- Stochasticity (optional): Gaussian action noise (std action_noise m/s per component, drawn once
  per control step) and Gaussian obs noise (std obs_noise m on pusher/obj, m/s on obj_vel), both from
  the env-owned Generator `rng` (seeded in reset, part of snapshot). Default: both 0 (deterministic).
- Termination: success = obj within goal_radius of goal AND |obj_vel| < 1e-3 (at rest in goal);
  out_of_bounds = obj centre leaves table; timeout = t >= max_time.
"""
from __future__ import annotations
import copy
import math
import numpy as np
from shared.interface import DT, SUBSTEP, TABLE, StepResult, clip_action

G = 9.81
R_P, R_O = 0.03, 0.05
TAU_ACT = 0.02            # actuator lag time constant [s]
REST_SPEED = 1e-3         # [m/s] object considered at rest below this
N_SUB = int(round(DT / SUBSTEP))
OBS_KEYS = ("t", "pusher", "obj", "obj_vel", "goal", "goal_radius")


class PushEnv:
    """TOY 2D push env. hidden = {"mu": float}. See module docstring."""

    def __init__(self, action_noise: float = 0.0, obs_noise: float = 0.0, max_time: float = 6.0,
                 tau_act: float = TAU_ACT):
        self.action_noise = float(action_noise)
        self.obs_noise = float(obs_noise)
        self.max_time = float(max_time)
        self.tau_act = float(tau_act)
        self.hidden = {}

    # ------------------------------------------------------------------ protocol
    def reset(self, init_state: dict, hidden: dict, seed: int):
        assert set(hidden) == {"mu"}, "v1 env varies only mu"
        self.hidden = {"mu": float(hidden["mu"])}
        self.p = np.array(init_state["pusher"], dtype=np.float64)
        self.o = np.array(init_state["obj"], dtype=np.float64)
        self.ov = np.zeros(2)
        self.pv = np.zeros(2)                 # actuator (controller) state: actual pusher velocity
        self.cmd = np.zeros(2)                # last executed (noisy, clipped) command
        self.goal = np.array(init_state["goal"], dtype=np.float64)
        self.goal_radius = float(init_state.get("goal_radius", 0.03))
        self.t = 0.0
        self.n_steps = 0
        self.in_contact = False
        self.contact_substeps = 0             # cumulative substeps in contact (task/contact state)
        self.done = False
        self.success = False
        self.termination = None
        self.rng = np.random.default_rng(seed)
        self._last_obs = self._make_obs()
        return self.observe()

    def observe(self):
        o = self._last_obs
        return {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in o.items()}

    def step(self, action):
        if self.done:
            return StepResult(self.observe(), True, self.success,
                              {"termination": self.termination, "note": "already done"})
        v = clip_action(np.asarray(action, dtype=np.float64).reshape(2))
        if self.action_noise > 0:
            v = clip_action(v + self.rng.normal(0.0, self.action_noise, 2))
        self.cmd = v
        for _ in range(N_SUB):
            self._substep(v)
        self.n_steps += 1
        self.t = round(self.n_steps * DT, 10)
        reason = None
        if np.any(np.abs(self.o) > TABLE):
            reason = "out_of_bounds"
        elif (np.linalg.norm(self.o - self.goal) < self.goal_radius
              and np.linalg.norm(self.ov) < REST_SPEED):
            reason = "success"; self.success = True
        elif self.t >= self.max_time - 1e-9:
            reason = "timeout"
        if reason is not None:
            self.done, self.termination = True, reason
        self._last_obs = self._make_obs()
        return StepResult(self.observe(), self.done, self.success,
                          {"termination": reason, "in_contact": self.in_contact})

    def step_segment(self, actions):
        out = []
        for a in np.asarray(actions, dtype=np.float64).reshape(-1, 2):
            out.append(self.step(a))
            if out[-1].done:
                break
        return out

    _STATE = ("p", "o", "ov", "pv", "cmd", "goal", "goal_radius", "t", "n_steps", "in_contact",
              "contact_substeps", "done", "success", "termination", "hidden", "action_noise",
              "obs_noise", "max_time", "tau_act", "_last_obs")

    def snapshot(self):
        s = {k: copy.deepcopy(getattr(self, k)) for k in self._STATE}
        s["rng"] = copy.deepcopy(self.rng.bit_generator.state)
        s["_cls"] = "PushEnv-v1"
        return s

    def restore(self, snap):
        assert snap.get("_cls") == "PushEnv-v1"
        s = copy.deepcopy(snap)
        for k in self._STATE:
            setattr(self, k, s[k])
        self.rng = np.random.default_rng()
        self.rng.bit_generator.state = s["rng"]

    # ------------------------------------------------------------------ internals
    def _make_obs(self):
        p, o, ov = self.p.copy(), self.o.copy(), self.ov.copy()
        if self.obs_noise > 0:
            p = p + self.rng.normal(0.0, self.obs_noise, 2)
            o = o + self.rng.normal(0.0, self.obs_noise, 2)
            ov = ov + self.rng.normal(0.0, self.obs_noise, 2)
        return {"t": self.t, "pusher": p, "obj": o, "obj_vel": ov,
                "goal": self.goal.copy(), "goal_radius": self.goal_radius}

    def _substep(self, v):
        """Order: (1) Coulomb friction on object velocity, (2) pusher actuator + kinematics,
        (3) contact projection + normal-velocity raise, (4) object position integration.
        Friction is applied BEFORE the contact raise so that, while in contact, the observed obj_vel
        equals the pusher normal velocity exactly (no mu-dependent artefact in obj_vel during contact)."""
        mu = self.hidden["mu"]
        sp = float(np.linalg.norm(self.ov))
        if sp > 0:
            self.ov = self.ov * (max(0.0, sp - mu * G * SUBSTEP) / sp)
        if self.tau_act > 0:
            self.pv = self.pv + (v - self.pv) * (1.0 - math.exp(-SUBSTEP / self.tau_act))
        else:
            self.pv = v.copy()
        lim = TABLE - R_P
        newp = np.clip(self.p + self.pv * SUBSTEP, -lim, lim)
        pvel = (newp - self.p) / SUBSTEP
        self.p = newp
        d = self.o - self.p
        dist = float(np.linalg.norm(d))
        self.in_contact = False
        if dist < R_P + R_O - 1e-12 and dist > 1e-12:
            n = d / dist
            self.o = self.p + n * (R_P + R_O)
            vn = float(pvel @ n)
            ovn = float(self.ov @ n)
            if vn > ovn:
                self.ov = self.ov + (vn - ovn) * n
            self.in_contact = True
            self.contact_substeps += 1
        self.o = self.o + self.ov * SUBSTEP
