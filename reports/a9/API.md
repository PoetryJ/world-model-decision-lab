# A9 importable API for the sealed E2/E3 run

Time zone: Asia/Shanghai (UTC+8).

A9 will **not** load `test`, `test_neg`, or `test_sealed*` (controller decision A-8).
The controller runs the sealed test with this API.

## Load

```python
import sys
sys.path.insert(0, "/workspace/wmprobe")
from src.direct_baselines.api import load_frozen_a9, decide

bundle = load_frozen_a9("/workspace/wmprobe/runs/a9/frozen_strike54/models")
# bundle["proposer"]            # DirectProposer(source="strikegrid") -> A4 strike_grid (54, 6, 2)
# bundle["obs_only"]            # learned ridge, current observation only
# bundle["history_direct"]      # learned ridge + probe-feedback features (no dynamics model)
# bundle["physics_obs"]         # mechanical scorer, fixed a prior (mu=0.35 midpoint of 0.1/0.6)
# bundle["physics_history"]     # same, but estimates deceleration from probe release (tap_fast)
# bundle["heuristic_obs"]       # geometric (obj + commanded displacement)
# bundle["heuristic_history"]   # geometric scaled by |obj_disp|/|pusher_disp|
```

Weights: `runs/a9/frozen_strike54/models/{obs_only,history_direct}.npz`
Frozen config: `reports/a9/config_frozen.json` (written after train/val fit).

## A-3 episode (runner-side; A9 policy never sees physical parameters)

1. `obs = env.reset(init, <condition>, seed)` with the **frozen seed**. `action_noise=0`, `obs_noise=0`.
2. Optional **one** probe, ≤5 steps, from `src.env.scenarios.probe_set(obs)` (`tap_fast` / `push_slow` / `retreat`), executed on the live env. Append `Transition(obs, actions, next_obs, kind="probe")`. No restore, no reset.
3. Task decision at the actual post-probe (or initial) observation:

```python
cands = bundle["proposer"].propose(obs, history, seed)   # (54, 6, 2), H=6 <= 12
out = decide(bundle["history_direct"], obs, history, cands)
# out["executed"] == cands[out["chosen_index"]]
# out["candidate_counts"] = {requested, returned, failed}
# out["score_time_s"]  -> DecisionRecord.wall_time_s["score"]
```

4. `env.step_segment(out["executed"])` fully, unless the env terminates.
5. Settle: zero actions until `|obj_vel| < 1e-3` or budget. **No non-zero action after the task segment.** Hard cap 60 control steps (6.0 s) including probe + task + settle.
6. Termination: success (in goal and at rest), out_of_bounds, timeout (6.0 s), budget (settled outside goal).

Use the **same** candidate array for every scorer at a given decision (call `propose` once).

## Methods to run (same budget, same probe, same candidates)

| method | key | uses history | future rollout |
|---|---|---|---|
| obs-only learned | `obs_only` | no | no |
| history-direct learned | `history_direct` | yes (displacement, mobility, residual speed) | no |
| physics obs | `physics_obs` | no (nominal a=0.35 g) | cheap kinematic heuristic |
| physics history | `physics_history` | yes (a estimated from tap release) | cheap kinematic heuristic |
| random | `RandomSelector(seed)` | no | no |

`physics_*` are direct baselines with a 1-scalar mechanical inductive bias, **not** A6's learned/residual world model.

## Conditions (controller)

Test conditions in the sealed set: `mu_lo=0.1`, `mu_mid=0.3` (held out of A9 training), `mu_hi=0.6`.
Main pair: lo vs hi=0.6. Report mid separately.

## Relabel of earlier A9 numbers

`runs/a9/logs` and `runs/a9/speedgrid` used **A4 `e1_set` at mu 0.1/0.25/0.5**, not the frozen protocol (A-7). Do not pool with sealed-test results.

## Val recommendation (not a test result)

On official val (train_val mus, strike_grid, A-3 labels): run **all four** of `physics_obs`, `physics_history`, `history_direct`, `obs_only` in the sealed comparison. `physics_obs` was the strongest overall val method (success 0.720 vs oracle 0.939). `physics_history` is the one that almost matches the oracle on **mu_lo after tap_fast** and can fail on **mu_hi after tap_fast** (a_est lower bound). Do not send only the learned ridge.

Config sha256: `19a28a25928e24c0246cec93de1ad31d035b49e4bf836d67230f1237428605ac`.
Frozen 2026-11-12 18:24 Asia/Shanghai (UTC+8).
