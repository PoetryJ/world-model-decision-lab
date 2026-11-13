"""Wiring A6 (src/dynamics EnsemblePredictor + src/adaptation RLSAdapter) into the A8 loop, read-only use.
PaddedPredictor: predicts the SETTLED outcome by appending `settle` zero-action steps (one-shot regime: the env
also holds still after the strike). Prediction.endpoints/samples are settled positions; extra['settled'] = samples.
DisagreementProbe (generic method 4b): pick probe with max ensemble spread of predicted probe outcome; probe iff the
spread (RMS member std, metres) > thr. thr = 0.005 m fixed a priori (1/6 of A4 goal radius), not tuned.
"""
from __future__ import annotations
import numpy as np
from shared.interface import Prediction
from .probers import default_library, _fits

class PaddedPredictor:
    def __init__(self, base, settle: int = 20):
        self.base, self.settle = base, settle
    def predict(self, observation, actions, context):
        A = np.asarray(actions, float)
        A = np.concatenate([A, np.zeros((A.shape[0], self.settle, 2))], axis=1)
        P = self.base.predict(observation, A, context)
        ex = dict(P.extra); ex["settled"] = P.samples; ex["weights"] = np.full(P.samples.shape[1], 1.0 / P.samples.shape[1])
        return Prediction(endpoints=P.endpoints, samples=P.samples, traj=None, extra=ex)

class DisagreementProbe:
    name = "disagree_probe"
    def __init__(self, predictor, adapter, thr: float = 0.005, library=None):
        self.pred, self.adapter, self.thr, self.last = predictor, adapter, thr, {}
        self.lib = library or default_library()
    def choose_probe(self, observation, history, budget):
        st = self.adapter.update(history); spread = {}
        for name, seg in self.lib(observation).items():
            if not _fits(seg, budget): continue
            P = self.pred.predict(observation, seg[None], {"history": history, "adapt_state": st})
            spread[name] = float(np.sqrt(np.mean(np.var(P.samples[0], axis=0))))
        if not spread:
            self.last = {"reason": "budget"}; return None
        best = max(spread, key=spread.get)
        self.last = {"spread": spread, "best": best}
        if spread[best] <= self.thr:
            self.last["reason"] = "spread_below_thr"; return None
        self.last["reason"] = "disagree"; return self.lib(observation)[best]
