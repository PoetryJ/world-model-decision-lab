"""Legal probe segments built from the CURRENT VISIBLE observation only.
All probes: per-component |v| <= V_MAX, fixed length PROBE_H control steps (PROBE_H*DT s), world frame.
A probe = approach/push along a direction at speed v for k steps, then hold still (v=0) to watch the
object slide (free-sliding deceleration = mu*g is what makes mu visible).
"""
from __future__ import annotations
import numpy as np
from shared.interface import DT, V_MAX, clip_action

PROBE_H = 4
RR = 0.08  # assumed pusher+object radius sum (agent-side constant)

def _unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else np.array([1.0, 0.0])

def tap(obs, speed, direction="obj", extra_push=0.01, h=PROBE_H):
    p, o, g = np.asarray(obs["pusher"]), np.asarray(obs["obj"]), np.asarray(obs["goal"])
    u = _unit(o - p) if direction == "obj" else _unit(g - o)
    gap = max(np.linalg.norm(o - p) - RR, 0.0)
    k = int(np.clip(np.ceil((gap + extra_push) / (speed * DT)), 1, h - 1))
    seg = np.zeros((h, 2))
    seg[:k] = u * speed
    return clip_action(seg)

def no_contact(obs, h=PROBE_H):
    """Legal but (by construction) uninformative probe: move pusher away from the object, then stop."""
    p, o = np.asarray(obs["pusher"]), np.asarray(obs["obj"])
    seg = np.zeros((h, 2)); seg[:2] = -_unit(o - p) * 0.1
    return seg

def library(obs):
    """dict name -> (PROBE_H,2) segment. Same 3 probes as used for E0 discussion."""
    return {"tap_fast": tap(obs, 0.4), "tap_slow": tap(obs, 0.15), "no_contact": no_contact(obs)}
