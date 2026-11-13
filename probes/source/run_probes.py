"""A2 probes: mechanics (M1-M5) and suspicions (Sa-Sd). Writes runs/a2/probe_results.json.
Run: cd /workspace/wmprobe && . .venv/bin/activate && python probes/source/run_probes.py
All randomness: np.random.default_rng(SEED) with SEED listed per probe. [toy-run] of ported code.
"""
from __future__ import annotations

import json
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import cosmos_port as cp  # noqa: E402
import fixes  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "..", "..", "runs", "a2")
R = {}


def m1_pack_unpack(seed=0):
    rng = np.random.default_rng(seed)
    res = {}
    for name, (C, a_dim) in {"libero": (16, 7), "robocasa": (32, 7), "aloha": (50, 14)}.items():
        B, Cl, T, H, W = 2, 16, 11, 28, 28
        x0 = rng.standard_normal((B, Cl, T, H, W)).astype(np.float32)
        act = rng.uniform(-1, 1, (B, C, a_dim)).astype(np.float32)
        idx = np.array([4, 4])
        x = cp.replace_latent_with_action_chunk(x0, act, idx)
        back = cp.extract_action_chunk_from_latent_sequence(x, (C, a_dim), idx)
        L = Cl * H * W
        n = C * a_dim
        # noisy-latent averaging: extraction averages k = L//n copies
        sig = 0.5
        xn = x.copy()
        xn[:, :, 4] += sig * rng.standard_normal(xn[:, :, 4].shape).astype(np.float32)
        noisy_back = cp.extract_action_chunk_from_latent_sequence(xn, (C, a_dim), idx)
        res[name] = dict(chunk=C, action_dim=a_dim, latent_elems=L, action_elems=n, full_copies=L // n,
                         partial_tail_elems=L % n, roundtrip_max_abs_err=float(np.abs(back - act).max()),
                         noise_sigma=sig, extracted_err_std=float((noisy_back - act).std()),
                         expected_std_sigma_over_sqrt_k=sig / math.sqrt(L // n),
                         other_frames_untouched=bool(np.array_equal(np.delete(x, 4, axis=2), np.delete(x0, 4, axis=2))))
    # value: scalar broadcast + mean
    x0 = rng.standard_normal((3, 16, 9, 28, 28)).astype(np.float32)
    v = np.array([-1.0, 0.2, 1.0], dtype=np.float32)
    xv = cp.replace_latent_with_value(x0, v, np.array([8, 8, 8]))
    res["value_roundtrip_max_abs_err"] = float(np.abs(cp.extract_value_from_latent_sequence(xv, np.array([8, 8, 8])) - v).max())
    R["M1_pack_unpack"] = res


def m2_normalization(seed=1):
    rng = np.random.default_rng(seed)
    a = rng.uniform(-3, 3, (500, 7))
    mn, mx = a.min(0), a.max(0)
    z = cp.rescale(a, mn, mx)
    back = cp.unnormalize_actions(z.reshape(50, 10, 7), mn, mx).reshape(500, 7)
    a_const = a.copy(); a_const[:, 6] = 1.0
    mn2, mx2 = a_const.min(0), a_const.max(0)
    with np.errstate(all="ignore"):
        zc = cp.rescale(a_const, mn2, mx2)
    # out-of-range inference proprio
    z_oor = cp.rescale(np.array([mx[0] + 0.5 * (mx[0] - mn[0])]), mn[0:1], mx[0:1])
    R["M2_normalization"] = dict(range=[float(z.min()), float(z.max())], roundtrip_max_abs_err=float(np.abs(back - a).max()),
                                 constant_dim_output_nan_count=int(np.isnan(zc[:, 6]).sum()), constant_dim_rows=500,
                                 out_of_range_value_not_clipped=float(z_oor[0]))


def m3_condition_mask():
    T, A, V = 9, 4, 8  # LIBERO layout cosmos_utils.py:1297
    cur = (1, 2, 3)
    R["M3_condition_mask_libero"] = {
        "layout": "0 blank,1 proprio,2 wrist,3 primary,4 action,5 fut proprio,6 fut wrist,7 fut primary,8 value",
        "policy_demo(k=4)": cp.condition_mask(T, 4, "policy", A, V).tolist(),
        "world_model(k=4)": cp.condition_mask(T, 4, "world_model", A, V).tolist(),
        "value V(s,a,s')": cp.condition_mask(T, 4, "value", A, V, cur, False).tolist(),
        "value V(s') masked": cp.condition_mask(T, 4, "value", A, V, cur, True).tolist(),
    }


def m4_time_alignment(seed=2):
    """Synthetic episode: proprio s_t = t (scalar), action a_t = +1 moves s_t -> s_{t+1}. Check endpoint vs chunk."""
    T, C = 40, 16
    actions = np.ones((T, 1)) * np.arange(T)[:, None]  # a_t encodes its own t
    rows = []
    for t in range(T):
        ix = cp.libero_sample_indices(t, C, T)
        chunk = cp.get_action_chunk_with_padding(actions, t, C, T)
        real = min(C, T - t)
        transitions_in_endpoint = ix["future_frame_idx"] - t  # obs index difference = # actions whose effect is visible
        rows.append(dict(t=t, future_frame_idx=ix["future_frame_idx"], real_actions=real, padded=C - real,
                         transitions_visible=transitions_in_endpoint,
                         chunk_actions_without_visible_effect=int(C - transitions_in_endpoint),
                         pad_value_is_last_action=bool(np.all(chunk[real:] == actions[-1])) if C > real else None))
    mis = [r for r in rows if r["chunk_actions_without_visible_effect"] > 0]
    R["M4_time_alignment"] = dict(T=T, chunk=C, n_samples=T, n_misaligned=len(mis), first_misaligned_t=mis[0]["t"],
                                  example_tail=[r for r in rows if r["t"] in (T - C - 1, T - C, T - 5, T - 1)],
                                  frac_misaligned=len(mis) / T,
                                  interior_ok=all(r["transitions_visible"] == C for r in rows if r["t"] + C <= T - 1))


def m5_return_labels():
    out = {}
    for name, gamma, T in [("libero", 0.99, 300), ("aloha", 0.998, 2000)]:
        g = cp.compute_monte_carlo_returns(T, 1.0, gamma)
        f = cp.compute_monte_carlo_returns(T, 0.0, gamma)
        v01 = (g + 1) / 2  # cosmos_utils.py:1504 unnormalization
        out[name] = dict(gamma=gamma, T=T, success_v_last=float(v01[-1]), success_v_first=float(v01[0]),
                         failure_all_minus1=bool(np.all(f == -1)),
                         steps_to_go_where_v_below_0p05=int(math.ceil(math.log(0.05) / math.log(gamma))))
    # ALOHA continuous success_score (aloha_dataset.py:548,564): score cancels in dataset_common.py:51
    a1 = cp.compute_monte_carlo_returns(1000, 1.0, 0.998)
    a03 = cp.compute_monte_carlo_returns(1000, 0.3, 0.998)
    out["aloha_success_score_0.3_vs_1.0_max_abs_label_diff"] = float(np.abs(a1 - a03).max())
    out["aloha_success_score_0.01_vs_1.0_max_abs_label_diff"] = float(np.abs(a1 - cp.compute_monte_carlo_returns(1000, 0.01, 0.998)).max())
    R["M5_return_labels"] = out


# ---------------------------------------------------------------- suspicions
def sa_lcb_single(seed=3):
    rng = np.random.default_rng(seed)
    N = 8
    vals = rng.uniform(0, 1, N).astype(np.float32)
    lcb1 = [float(cp.aggregate_value_predictions([np.array([v])], "lcb")[0]) for v in vals]
    qr = [{"actions": i, "value_prediction": lcb1[i]} for i in range(N)]
    sel_l = cp.libero_select(qr, 0)
    sel_a = cp.aloha_select(qr, 0)
    true_best = int(np.argmax(vals))
    # other schemes with single sample
    others = {s: float(cp.aggregate_value_predictions([np.array([0.7])], s)[0]) for s in ["average", "success_vote", "majority_mean"]}
    # n=2 and future-state ensemble of 3 x value 1 -> list length 3 -> fine
    lcb3 = float(cp.aggregate_value_predictions([np.array([0.6]), np.array([0.7]), np.array([0.8])], "lcb")[0])
    fixed = [float(fixes.aggregate_value_predictions_fixed([np.array([v])], "lcb")[0]) for v in vals]
    qr_f = [{"actions": i, "value_prediction": fixed[i]} for i in range(N)]
    R["Sa_lcb_single_sample"] = dict(seed=seed, raw_values=vals.tolist(), lcb_n1=lcb1, all_nan=bool(np.all(np.isnan(lcb1))),
                                     libero_selected_seed=sel_l[0], aloha_selected_seed=sel_a[0], true_best_query=true_best,
                                     other_schemes_n1=others, lcb_n3=lcb3, ensemble_steps_n1=cp.value_ensemble_steps(5, 1),
                                     ensemble_steps_n5=cp.value_ensemble_steps(5, 5), fixed_lcb_n1=fixed,
                                     fixed_selected=cp.libero_select(qr_f, 0)[0])


def sb_majority_threshold(seed=4):
    rng = np.random.default_rng(seed)
    out = {}
    # (1) invariance test: scale all ensemble values by c; does the candidate ranking change?
    M, N, trials = 5, 8, 2000
    flips = {"average": 0, "majority_mean": 0, "lcb": 0}
    ties_mm = {}
    for c in [1.0, 0.5, 0.1, 0.05]:
        ties = 0
        for _ in range(trials):
            base = rng.uniform(0, 1, (N, 1)) ** 3  # many small values
            ens = np.clip(base + 0.05 * rng.standard_normal((N, M)), 0, 1)
            for s in flips:
                v1 = np.array([cp.aggregate_value_predictions([ens[i, m:m + 1] for m in range(M)], s)[0] for i in range(N)])
                vc = np.array([cp.aggregate_value_predictions([c * ens[i, m:m + 1] for m in range(M)], s)[0] for i in range(N)])
                if c != 1.0 and int(np.argmax(v1)) != int(np.argmax(vc)):
                    flips[s] += 1
                if s == "majority_mean" and np.sum(vc == vc.max()) > 1:
                    ties += 1
        ties_mm[str(c)] = ties / trials
    out["argmax_flip_count_over_3_scales_x_trials"] = flips
    out["trials_per_scale"] = trials
    out["majority_mean_frac_trials_with_top_tie"] = ties_mm
    # (2) label scale: perfect predictor, two candidates far from goal: success path k steps vs failure (v=0)
    rows = []
    for name, gamma in [("libero", 0.99), ("aloha", 0.998)]:
        for k in [100, 250, 299, 300, 500, 1000, 1496, 1497, 1800]:
            v_succ = gamma ** k
            vs = [np.array([v_succ])] * 5
            vf = [np.array([0.0])] * 5
            rows.append(dict(suite=name, gamma=gamma, steps_to_go=k, true_v_success=v_succ,
                             mm_success=float(cp.aggregate_value_predictions(vs, "majority_mean")[0]),
                             mm_failure=float(cp.aggregate_value_predictions(vf, "majority_mean")[0]),
                             avg_success=float(cp.aggregate_value_predictions(vs, "average")[0]),
                             fixed_rel_mm_success=float(fixes.majority_mean_relative(vs, gamma=gamma, horizon_steps=(520 if name == "libero" else 2000))[0])))
    out["perfect_predictor_success_vs_failure"] = rows
    R["Sb_majority_mean_threshold"] = out


def sc_seeds(seed=5):
    rng = np.random.default_rng(seed)
    out = {}
    cfgs = [("libero_doc 4q/4gpu", 4, [0, 1, 2, 3], True), ("libero 8q/8gpu default", 8, list(range(8)), True),
            ("8q on 4 gpus", 8, [0, 1, 2, 3], True), ("4q on gpus 4-7", 4, [4, 5, 6, 7], True),
            ("aloha deploy 4q, 8 gpus listed (no truncation)", 4, list(range(8)), False),
            ("aloha deploy 8q on '2,3'", 8, [2, 3], False)]
    for name, N, gpus, trunc in cfgs:
        s = cp.serial_seeds(195, N)
        p = cp.parallel_seeds(195, N, gpus, trunc)
        out[name] = dict(serial=s, parallel=p, identical_to_serial=s == p, n_unique_parallel=len(set(p)),
                         logged_seed_labels=[195 + q for q in range(N)])
    # randomize_seed=True: % 256 collisions (exact + Monte Carlo)
    coll = {}
    for N in [4, 5, 8, 16]:
        exact = 1 - np.prod([(256 - i) / 256 for i in range(N)])
        mc = np.mean([len(set(cp.randomized_seed(rng) for _ in range(N))) < N for _ in range(20000)])
        coll[N] = dict(p_collision_exact=float(exact), p_collision_mc_20000=float(mc))
    out["randomize_seed_mod256_collision"] = coll
    # cross-stage noise reuse: query q value-ensemble member i uses seed+q+i (cosmos_utils.py:1481 with seed=cfg.seed+q)
    reuse = [(q, i, q + i) for q in range(8) for i in range(5) if 0 < q + i < 8]
    out["value_member_noise_equal_to_other_query_action_noise_pairs"] = len(reuse)
    # deterministic sampler => same seed => same initial noise (misc.arch_invariant_rand -> np.random.RandomState(seed))
    n1 = np.random.RandomState(196).standard_normal((1, 16, 9, 28, 28)).astype(np.float32)
    n2 = np.random.RandomState(196).standard_normal((1, 16, 9, 28, 28)).astype(np.float32)
    out["same_seed_identical_initial_noise"] = bool(np.array_equal(n1, n2))
    # effect on best-of-N: duplicates reduce effective N. toy: candidate value = f(seed) iid U(0,1)
    vals = rng.uniform(0, 1, 1000)
    eff = {}
    for name, N, gpus, trunc in cfgs:
        p = cp.parallel_seeds(0, N, gpus, trunc)
        eff[name] = dict(expected_max_serial=float(np.mean([max(rng.uniform(0, 1, N)) for _ in range(5000)])),
                         expected_max_parallel=float(np.mean([max(rng.uniform(0, 1, len(set(p)))) for _ in range(5000)])),
                         unique=len(set(p)), N=N)
    out["toy_best_of_N_effect_uniform_values"] = eff
    R["Sc_seeds"] = out


def sd_partial_failure():
    out = {}

    def make_task(fail_gpus, slow=0.0):
        def task_fn(gpu_id, seed, task_id):
            if gpu_id in fail_gpus:
                raise RuntimeError(f"simulated CUDA OOM on gpu {gpu_id}")
            time.sleep(slow)
            return {"actions": f"A(seed={seed})", "value_prediction": 0.1 + 0.1 * gpu_id, "true_seed": seed}
        return task_fn

    for name, gpus, fail in [("no failure", [0, 1, 2, 3], set()), ("gpu0 fails", [0, 1, 2, 3], {0}),
                             ("gpu1,3 fail", [0, 1, 2, 3], {1, 3}), ("all fail", [0, 1, 2, 3], {0, 1, 2, 3})]:
        pool = cp.FakePool(gpus, 4, parallel_timeout=1.0, base_seed=195, task_fn=make_task(fail, slow=0.02))
        qr = cp.query_model_parallel(pool)
        info = dict(pool.last)
        pool.shutdown()
        row = dict(info, returned_type=type(qr).__name__)
        if qr:
            labels = [195 + q for q in range(len(qr))]
            row["logged_seed_labels"] = labels
            row["true_seeds"] = [r["true_seed"] for r in qr]
            row["labels_match_truth"] = labels == row["true_seeds"]
            bs, _ = cp.libero_select(qr, 195)
            row["libero_selected_label"] = bs
            row["libero_selected_true_seed"] = [r["true_seed"] for r in qr][bs - 195]
            abs_, _, _ = cp.aloha_select(qr, 195)
            row["aloha_base_seed_label_points_to_true_seed"] = qr[0]["true_seed"]
        else:
            try:
                _ = f"Parallel queries completed: {len(qr)} results"  # libero/run_libero_eval.py:436
                row["libero_log_line"] = "ok"
            except TypeError as e:
                row["libero_log_line_raises"] = f"TypeError: {e}"
            # robust version
        fx = fixes.submit_tasks_recorded(gpus, 4, parallel_timeout=1.0, base_seed=195, task_fn=make_task(fail, slow=0.02))
        row["fixed_record"] = {k: fx[k] for k in ["n_requested", "n_returned", "n_errors", "n_timeout", "elapsed",
                                                  "returned_seeds", "failed_seeds"]}
        out[name] = row
    R["Sd_partial_failure"] = out


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    for f in [m1_pack_unpack, m2_normalization, m3_condition_mask, m4_time_alignment, m5_return_labels,
              sa_lcb_single, sb_majority_threshold, sc_seeds, sd_partial_failure]:
        f()
    R["_meta"] = dict(label="[toy-run] numpy port of cosmos-policy@18a2acc", wall_s=time.time() - t0,
                      numpy=np.__version__)
    p = os.path.join(OUT, "probe_results.json")
    with open(p, "w") as fh:
        json.dump(R, fh, indent=1, default=str)
    print(json.dumps(R, indent=1, default=str))
