# Controller final report — 3h world-model probe (20:15 checkpoint, written 2026-11-12 ~20:30 UTC+8)
Question: when should a robot trust a world model, and can a few interactions improve later action choice?
Scope: TOY 2-D PushEnv (A4), hidden friction mu ∈ {0.1, 0.3 (held-out), 0.6}; CPU only; NO official Cosmos run (no CUDA GPU). All numbers [toy-run].

## 1. Commits (git log, newest first)
```
4e57ac8 A3 source errata; deterministic sealed repro
ae355ec controller 19:15 checkpoint
71a980b sealed E2/E3 run outputs (single run, audit OK)
c2513f7 A9 frozen + controller runner + val pilot
b3997dd A10: A-9 audit support
5fe9bb3 A8 deliverables; ruling A-9
caa9508 A7 deliverables (frozen scorer)
90efc52 A10: sealed test split + E1 reconcile + runner audit
3be500b A-8 test exposure decision
e9e5164 A10 deliverables
d704c25 controller decision A-7 conditions
eda1fd8 A5 deliverables
b3a4064 A2 deliverables; interface v1.1 candidate_counts
9a5aeda ledger decisions
6158b29 ignore cosmos clone (pinned 18a2acc, see ledger)
aadad89 A1,A3,A4 deliverables
0b63ab4 controller: frozen interface v1 + fixture + ledger
```

## 2. Component map
| component | path | owner | status |
|---|---|---|---|
| frozen interface v1.1 + fixture | shared/interface.py, shared/fixture_env.py | controller | frozen |
| Cosmos source map / smoke gap | reports/a1/component_map.md, adapters/cosmos/src @18a2acc | A1 | done; smoke BLOCKED (39 CUDA sites, TE/flash/natten) |
| Cosmos suspicion probes (LCB n=1 NaN, majority_mean threshold, seeds, partial failure) | probes/source, reports/a2 | A2 | done, 4/4 confirmed |
| prior work + 3 directions | reports/a3 | A3 | done (+ source errata 19:16) |
| env + E0 gate | src/env (push_env, scenarios, e0_gate) | A4 | E0 PASS |
| proposers / best-of-N / oracle | src/candidates, src/planning, reports/a5 | A5 | done |
| ensemble dynamics + RLS adapter | src/dynamics, src/adaptation, runs/a6/models | A6 | frozen 18:22 |
| scorer / uncertainty gate | src/scoring, reports/a7/frozen_scorer_config.json | A7 | frozen |
| probers (tap, random, goalrule, taskvoi) | src/probing/frozen_config.json | A8 | frozen |
| direct baselines (obs, history, physics) | src/direct_baselines, reports/a9/config_frozen.json | A9 | frozen |
| protocol, sealed split, metrics, audit | evaluation/, reports/a10 | A10 | frozen (PROTOCOL v2 + A-1..A-9) |
| closed-loop E2/E3 runner | runner/ (run_e2e3, methods, analyze_ctrl, manifest) | controller | sealed run 18:45–18:54 |

## 3. One-command repro
`runner/SEALED_CMD.md` (det + stoch + audit + both analyzers, ~10 min, 6 CPU). Runner refuses to overwrite `runs/controller/sealed*`, so a re-run goes to a new dir:
`. .venv/bin/activate && python -m runner.run_e2e3 --split sealed --out runs/controller/repro_X --workers 6 && python reports/controller/indep_check.py`
Tests: `python -m pytest -q --ignore=adapters` (71 pass; plain `pytest` collects the Cosmos clone and errors — use --ignore). E0: `python -m src.env.e0_gate`.

## 4. Independent re-runs (20:15)
- Deterministic sealed re-run (runs/controller/repro_sealed_det) vs sealed: 5040/5040 episodes, 0 mismatches in success/termination/goal error/steps.
- pytest 71/71 pass; E0 gate re-run reproduces (C2 pos det: lo vs hi share 0 good candidates; best-known-mu 1.0 vs best-fixed 0.667).
- Stochastic pairs recomputed from episodes.csv with own bootstrap (no A10 code) — match the 19:15 ledger exactly (reports/controller/indep_check.log).

## 5. Per-episode results
Per-episode rows: runs/controller/sealed/episodes.csv (det, 21 methods × 80 states × 3 mu = 5040) and runs/controller/sealed_stoch/episodes.csv (noise 0.02, ×3 reps). Audit valid, 0 issues, 0 replay mismatches.
Summary, deterministic success (all / pos / neg; tdp = episodes that end in success during the probe):
| method | all | pos | neg | tdp | stoch all |
|---|---|---|---|---|---|
| wm_noprobe | 0.758 | 0.67 | 0.85 | 0 | 0.711 |
| wm_tap_noupdate (probe, no update) | 0.662 | 0.70 | 0.62 | 75 | 0.550 |
| wm_tap_update | 0.771 | 0.92 | 0.62 | 75 | 0.607 |
| wm_random_update | 0.708 | 0.66 | 0.76 | 25 | 0.558 |
| wm_goalrule_update (task-relevant) | 0.883 | 0.92 | 0.85 | 0 | 0.733 |
| wm_taskvoi_update (infogain/VOI) | 0.871 | 0.89 | 0.85 | 0 | 0.718 |
| direct_obs_only / history_noprobe (learned) | 0.229 | 0.46 | 0.00 | 0 | 0.221 |
| direct_history_tap | 0.333 | 0.04 | 0.62 | 75 | 0.317 |
| direct_physics_obs (A9 analytic) | 0.758 | 0.65 | 0.87 | 0 | 0.717 |
| direct_physics_history_tap | 0.533 | 0.44 | 0.62 | 75 | 0.461 |
| belief_goalrule_update (PRIVILEGED) | 1.000 | 1.00 | 1.00 | 0 | 0.883 |
Key pairs (cluster bootstrap by init_state, 95% CI):
- update vs same-probe-no-update: det +0.108 [0.071,0.146] (pos +0.217); stoch +0.057 [0.035,0.079] → adaptation from one probe is real.
- tap+update vs no probe: det +0.012 [-0.046,0.075]; pos +0.25 but neg −0.225 → an always-on probe hurts when the best action doesn't depend on mu.
- goalrule vs no probe: det +0.125 [0.092,0.162] (post-hoc); stoch +0.022 [-0.006,0.05] → NOT robust to 0.02 action noise.
- taskvoi vs goalrule: −0.012 [-0.029,0.0] → probe-choice sophistication gives nothing here.
- wm_tap_update vs learned direct_history_tap: +0.438 det / +0.29 stoch — but the learned direct baseline is weak (ridge, K=54); vs analytic direct_physics_obs only +0.012.

## 6. Best next direction
D2-as-measurement (A3/A10 "LIMITED REWORK"): measure *when one probe + adaptation changes the top-1 decision*, with mandatory same-probe-no-update arm and negative controls, under noise. Concretely: (1) side-effect-free probes (75/240 tap episodes succeed during the probe = physical confound); (2) a probe gate that stays positive under action noise 0.02 (currently CI crosses 0; G4 noisy identifiability FAIL 0.80); (3) only then move off the toy env to a real WM (needs GPU).

## 7. Excluded directions (evidence)
- Sophisticated probe selection (infogain/VOI): taskvoi − goalrule −0.012 [-0.029,0]; random probe < no probe (0.708 vs 0.758).
- Always-probe: neg-subset −0.225 vs no-probe; tap-no-update < no-probe (0.662 vs 0.758).
- Learned history-direct ridge as primary baseline: 0.23–0.33, far below analytic physics (0.758); not a fair WM-vs-direct test.
- Novelty claims for probe-with-stop-rule / test-time latent adaptation / fail-gating: direct 2025–26 prior art (A3 §5c: Poke and Strike, DISaM, PhysCaP, T2-z).
- mu_mid identifiability: E0 G5 pre-registered FAIL 1/12; mid/hi share best action.
- Official Cosmos smoke test on this hardware: blocked (A1 §3).

## 8. Decisions needed from the lead
1. GPU (≥8 GB VRAM CUDA) for the official Cosmos Policy smoke test — or accept toy-only conclusions for this round.
2. Fix A9 a_est high-mu lower-bound bug? It is post-freeze → requires a NEW sealed split; current sealed set is spent.
3. Approve re-scope to D2 measurement (side-effect-free probes + noise-robust gate) as next round.
4. Treat goalrule/taskvoi gains as post-hoc only (not pre-registered); confirm reporting language.
Caveats: toy env (kinematic contact, true-pose obs, only mu varies); test/test_neg dev-exposed (A-8), only test_sealed reported; belief_* routes privileged.
