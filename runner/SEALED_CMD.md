# Sealed E2/E3 run — exact command (controller only; NOT yet executed)
Prepared 2026-11-12 ~18:45 UTC+8. Do not run until every module hyperparameter is frozen. Single-shot: the runner refuses
if `runs/controller/sealed*/decisions.jsonl` already exists, refuses `--limit/--methods` on sealed, and refuses if any hash in
`configs/frozen_manifest.json` changed (A4 env, A6 pkl+code, A7 config+code, A8 probing code+config, A9 npz+code+config,
A10 evaluation code + frozen_sets.json, shared/interface.py, runner/*). Sealed split sha is checked by `evaluation.audit.load_sealed`.

## One command (deterministic primary + stochastic variant + audit + analysis)
```bash
cd /workspace/wmprobe && . .venv/bin/activate && set -e && \
python -m runner.run_e2e3 --split sealed --out runs/controller/sealed --workers 6 && \
python -m runner.run_e2e3 --split sealed --out runs/controller/sealed_stoch --action_noise 0.02 --reps 0,1,2 --workers 6 && \
python -m evaluation.audit runner runs/controller/sealed/decisions.jsonl runs/controller/sealed_stoch/decisions.jsonl | tee runs/controller/sealed/audit.txt && \
python -m evaluation.analyze runs/controller/sealed/decisions.jsonl --split test \
  --pairs wm_tap_update:wm_tap_noupdate,wm_tap_update:direct_history_tap,wm_tap_update:wm_noprobe,wm_taskvoi_update:wm_goalrule_update \
  | tee runs/controller/sealed/a10_analyze.txt && \
python -m runner.analyze_ctrl runs/controller/sealed/decisions.jsonl --split sealed --regret --workers 6 \
  --pairs wm_tap_update:wm_tap_noupdate,wm_tap_update:direct_history_tap,wm_tap_update:wm_noprobe,wm_taskvoi_update:wm_goalrule_update,wm_tap_update:direct_physics_history_tap,wm_tap_update:direct_physics_obs,belief_tap_update:belief_tap_noupdate \
  | tee runs/controller/sealed/ctrl_analyze.txt && \
python -m runner.analyze_ctrl runs/controller/sealed_stoch/decisions.jsonl --split sealed --action_noise 0.02 --workers 6 \
  --pairs wm_tap_update:wm_tap_noupdate,wm_tap_update:direct_history_tap,wm_tap_update:wm_noprobe,wm_taskvoi_update:wm_goalrule_update,wm_tap_update:direct_physics_history_tap \
  | tee runs/controller/sealed_stoch/ctrl_analyze.txt
```
Outputs: `runs/controller/sealed/{decisions.jsonl,episodes.csv,episodes.jsonl,run_config.json,stage_times.json,ctrl_summary.json}`
(+ `decisions.a10_summary.json`), same under `runs/controller/sealed_stoch/` (env seed = seed*10+r, r∈{0,1,2}, action_noise 0.02, CRN across methods).

## Why two analyzers
- `evaluation.analyze` (A10, unmodified): accounting/validity, per-method + per-condition summaries, paired cluster-bootstrap diffs.
  With `--split test` its `coverage_missing_states` lists frozen *test* ids (sealed ids differ) -> ignore that field.
  `--regret` is NOT passed: `evaluation.analyze --regret` looks ids up only in frozen_sets.json -> KeyError on `test_sealed_*` ids.
- `runner.analyze_ctrl` (controller wrapper calling A10's `online_accounting`, `metrics`, `online_regret.analyse` unmodified):
  feeds sealed ids, gives per (pos/neg subset) x condition (mu_lo 0.1 / mu_mid 0.3 / mu_hi 0.6) breakdown, probe rate by subset,
  online top-1 regret + pairwise acc (deterministic only; stochastic regret would need CRN replay of noise -> not claimed).
- Clusters = init_state_id (conditions/reps of one state are not independent samples).

## Estimated runtime (from val 60 states x 3 cond, 21 methods, 6 workers: 52.5 s run, ~3.5 min regret)
sealed = 80 states x 3 = 240 episodes/method -> det run ~70 s; stoch (x3) ~3.5 min; regret (det) ~4-5 min; rest <1 min.
Total ~10 min wall on 6 of 8 CPUs, <100 MB RSS per worker.

## Hooks / caveats
- A9 direct_* methods auto-skip with a clear WARNING + `run_config.json:skipped` if the A9 API/weights are absent (currently present).
- wm_taskvoi_update: probe/no-probe decision uses A8's mu-grid belief VOI (privileged model class); task choice is WM stack.
- belief_* = PRIVILEGED structured physics belief route (A8 analytic model knows the env's functional form); secondary.
