# A6 FROZEN artefacts — 2026-11-12 18:24 (UTC+8). Do not overwrite; any new version gets a new file name.
Files are chmod a-w. Hyper-parameters were frozen at 18:22 (runs/a6/FROZEN_CONFIG.json); nothing changed since.

## Binding (use these for sealed E2/E3)
| artefact | sha256 |
|---|---|
| runs/a6/models/ens_push_env_v3_s0.pkl (main predictor, written 18:13:16) | eb221f9c804e1a8e383a700d5fb620e1038127a4a798c9c87263d0026187dbc0 |
| src/dynamics/model.py | d044f0dfad8bd74513d523a7950cdfe5af693c9b11f4ffb46b27a1ccf90d025b |
| src/adaptation/rls.py | 0500c3f84a22df5b0ae0e7da592cd7910d81645b229207c6950cb94bff74e6e6 |
Usage: `pred = load_or_train(seed=0)` (loads the pkl; never retrains if file exists); `ad = RLSAdapter(pred)`
(defaults = frozen: feat_idx=(2,), prior_std=5.0, noise_std=0.005, use_kinds=("probe","task")).
Training config (DEFAULT_CFG v3): train_mus [0.2,0.4,0.5]; val_mus [0.15,0.5] (monitor only); n_train_eps 1600 (54,222 steps),
n_val_eps 200; M=5 bootstrap members; MLP 6→64→64→4 tanh; 40 epochs Adam lr 3e-3 cosine, bs 256, wd 1e-5; env PushEnv();
init states = A10 train+train_neg (load_frozen) alternating with random scenes. Seeds: model seed 0 → collection seed 1, val
collection seed 2, member init seeds 0..4, bootstrap rng seed 0.

## Secondary (robustness / extension only)
| runs/a6/models/ens_push_env_v3_s1.pkl (18:18:59) | 87448c80519e97cf0156616e72cf55c1f5080bf6218c7e4b4e9af4f24e6f720b |
| runs/a6/models/ens_push_env_v3_s2.pkl (18:18:42) | da48d86b42389fbe72aabd29ed60421565ee0486f7f1e493621a73b82465c321 |
| runs/a6/models/latent_push_env_v3_s0.pkl (18:19:00) | 56eb27e8874a556dcd598dbf9d1e20fb84e3d97856f3c77034a2e115fa08bb32 |
| src/dynamics/latent.py | 2afdb43efd25859d8c5e7ce9d8b8982b9f75b5f2adbf2d77e60efa9b6182c106 |

## Why A8 may have seen a change (history of A6 versions)
- ens_push_env_s0.pkl (18:02, v1 cfg) → ens_push_env_v2_s0.pkl (18:08) → ens_push_env_v3_s0.pkl (18:13): load_or_train(0)
  returned a different file as DEFAULT_CFG["version"] changed. v3 pkl unchanged since 18:13:16.
- RLSAdapter defaults changed at ~18:05 (feat (1,2), prior 1.0) and 18:22 (feat (2,), prior 5.0 — frozen). A8 runs spanning
  either time would see different adaptation with identical probes.
