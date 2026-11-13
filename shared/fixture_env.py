"""Reference FIXTURE env (TOY, CPU, 2D). Owner: controller. Frozen for interface testing.
A4 builds the real experiment env in src/env implementing the same Env protocol.
Physics: kinematic disc pusher (r=0.03) pushes disc object (r=0.05). On contact the object's velocity
component along the contact normal is set to at least the pusher's normal velocity. Object decelerates
with Coulomb friction a = mu*g opposite velocity. Optional Gaussian action noise from env RNG.
"""
from __future__ import annotations
import copy
import numpy as np
from .interface import DT, SUBSTEP, TABLE, StepResult, clip_action

G = 9.81
R_P, R_O = 0.03, 0.05

class FixturePushEnv:
    def __init__(self, action_noise: float = 0.0, max_time: float = 6.0):
        self.action_noise = action_noise
        self.max_time = max_time

    def reset(self, init_state: dict, hidden: dict, seed: int):
        self.hidden = dict(hidden)
        self.p = np.array(init_state["pusher"], float)
        self.o = np.array(init_state["obj"], float)
        self.ov = np.zeros(2)
        self.goal = np.array(init_state["goal"], float)
        self.goal_radius = float(init_state.get("goal_radius", 0.05))
        self.t = 0.0
        self.done = False
        self.success = False
        self.rng = np.random.default_rng(seed)
        return self.observe()

    def observe(self):
        return {"t": self.t, "pusher": self.p.copy(), "obj": self.o.copy(), "obj_vel": self.ov.copy(),
                "goal": self.goal.copy(), "goal_radius": self.goal_radius}

    def _substep(self, v):
        mu = self.hidden["mu"]
        self.p = self.p + v * SUBSTEP
        d = self.o - self.p
        dist = np.linalg.norm(d)
        if dist < R_P + R_O and dist > 1e-9:
            n = d / dist
            self.o = self.p + n * (R_P + R_O)
            vn = float(v @ n)
            if vn > float(self.ov @ n):
                self.ov = self.ov + (vn - float(self.ov @ n)) * n
        sp = np.linalg.norm(self.ov)
        if sp > 0:
            dec = mu * G * SUBSTEP
            self.ov = self.ov * max(0.0, sp - dec) / sp
        self.o = self.o + self.ov * SUBSTEP

    def step(self, action):
        if self.done:
            return StepResult(self.observe(), True, self.success, {"note": "already done"})
        v = clip_action(action)
        if self.action_noise > 0:
            v = clip_action(v + self.rng.normal(0, self.action_noise, 2))
        for _ in range(int(round(DT / SUBSTEP))):
            self._substep(v)
        self.t = round(self.t + DT, 10)
        reason = None
        if np.any(np.abs(self.o) > TABLE):
            self.done, reason = True, "out_of_bounds"
        elif np.linalg.norm(self.o - self.goal) < self.goal_radius and np.linalg.norm(self.ov) < 1e-3:
            self.done, self.success, reason = True, True, "success"
        elif self.t >= self.max_time - 1e-9:
            self.done, reason = True, "timeout"
        return StepResult(self.observe(), self.done, self.success, {"termination": reason})

    def step_segment(self, actions):
        out = []
        for a in np.asarray(actions):
            out.append(self.step(a))
            if out[-1].done:
                break
        return out

    def snapshot(self):
        return copy.deepcopy({"p": self.p, "o": self.o, "ov": self.ov, "goal": self.goal,
                "goal_radius": self.goal_radius, "t": self.t, "done": self.done, "success": self.success,
                "rng": self.rng.bit_generator.state, "hidden": self.hidden,
                "action_noise": self.action_noise, "max_time": self.max_time})

    def restore(self, s):
        s = copy.deepcopy(s)
        self.p, self.o, self.ov, self.goal = s["p"], s["o"], s["ov"], s["goal"]
        self.goal_radius, self.t, self.done, self.success = s["goal_radius"], s["t"], s["done"], s["success"]
        self.rng = np.random.default_rng(); self.rng.bit_generator.state = s["rng"]
        self.hidden, self.action_noise, self.max_time = s["hidden"], s["action_noise"], s["max_time"]

FIXTURE_INIT = {"pusher": [-0.25, 0.0], "obj": [-0.15, 0.0], "goal": [0.15, 0.0], "goal_radius": 0.05}
FIXTURE_HIDDEN = [{"mu": 0.1}, {"mu": 0.3}, {"mu": 0.6}]
