"""A6 self-checks (python -m src.adaptation.selfcheck): API shapes, adapter purity / fresh state, empty history ==
no adaptation, predictor never touches env.hidden (env not passed at all; obs keys checked), latent API."""
import numpy as np
from shared.interface import Transition
from src.env.push_env import PushEnv
from src.env import scenarios as S
from src.dynamics.model import load_or_train
from src.adaptation.rls import RLSAdapter
from src.dynamics.latent import load_or_train_latent, LatentGridAdapter
from evaluation.audit import load_frozen

pred = load_or_train(0); ad = RLSAdapter(pred)
e = load_frozen("val", who="A6-selfcheck")[0]
env = PushEnv(); obs = env.reset(e["init"], {"mu": 0.15}, e["seed"])
assert set(obs) == {"t", "pusher", "obj", "obj_vel", "goal", "goal_radius"}
a = S.probe_set(obs)["tap_fast"]; res = env.step_segment(a)
h = [Transition(obs=obs, actions=a, next_obs=res[-1].obs, kind="probe")]
C, _ = S.strike_grid(env.observe())
P0 = pred.predict(env.observe(), C, {"history": h, "adapt_state": None})
assert P0.endpoints.shape == (54, 2) and P0.samples.shape == (54, 5, 2) and P0.traj.shape == (54, 6, 2)
s1 = ad.update(h); s2 = ad.update(list(h))
assert np.array_equal(s1["w"], s2["w"]), "update must be pure"
s0 = ad.update([]); assert np.all(s0["w"] == 0) and s0["n"] == 0
Pe = pred.predict(env.observe(), C, {"history": [], "adapt_state": s0})
assert np.allclose(Pe.endpoints, P0.endpoints), "empty history must equal no adaptation"
P1 = pred.predict(env.observe(), C, {"history": h, "adapt_state": s1})
assert P1.extra["residual_std"].shape == (54,)
lp = load_or_train_latent(0); la = LatentGridAdapter(lp); ls = la.update(h)
assert lp.predict(env.observe(), C, {"adapt_state": ls}).endpoints.shape == (54, 2)
print("A6 selfcheck OK | rls w =", s1["w"], "| latent mu_hat =", ls["mu_hat"],
      "| mean |adapted-base| endpoint shift [m] =", float(np.mean(np.linalg.norm(P1.endpoints - P0.endpoints, axis=1))))
