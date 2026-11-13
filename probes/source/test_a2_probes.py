"""Targeted regression tests for A2 probes. Run: cd /workspace/wmprobe && . .venv/bin/activate && pytest -q probes/source"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import cosmos_port as cp  # noqa: E402
import fixes  # noqa: E402


def test_pack_unpack_roundtrip_all_suites():
    rng = np.random.default_rng(0)
    for C, d in [(16, 7), (32, 7), (50, 14)]:
        x0 = rng.standard_normal((2, 16, 11, 28, 28)).astype(np.float32)
        a = rng.uniform(-1, 1, (2, C, d)).astype(np.float32)
        idx = np.array([5, 5])
        back = cp.extract_action_chunk_from_latent_sequence(cp.replace_latent_with_action_chunk(x0, a, idx), (C, d), idx)
        assert np.allclose(back, a, atol=1e-5)


def test_normalization_roundtrip():
    rng = np.random.default_rng(1)
    a = rng.uniform(-3, 3, (100, 7)); mn, mx = a.min(0), a.max(0)
    assert np.allclose(cp.unnormalize_actions(cp.rescale(a, mn, mx), mn, mx), a)


def test_tail_alignment_interior_ok_tail_misaligned():
    T, C = 40, 16
    for t in range(T):
        ix = cp.libero_sample_indices(t, C, T)
        if t + C <= T - 1:
            assert ix["future_frame_idx"] - t == C
        else:
            assert ix["future_frame_idx"] - t < C  # chunk claims C actions, endpoint shows fewer


def test_sa_reference_lcb_single_is_nan_and_fix_is_finite():
    v = [np.array([0.3])]
    assert np.isnan(cp.aggregate_value_predictions(v, "lcb")[0])
    assert np.isclose(fixes.aggregate_value_predictions_fixed(v, "lcb")[0], 0.3)
    # fix must not change n>1 behaviour
    v3 = [np.array([0.6]), np.array([0.7]), np.array([0.8])]
    assert np.isclose(fixes.aggregate_value_predictions_fixed(v3, "lcb")[0], cp.aggregate_value_predictions(v3, "lcb")[0])


def test_sa_nan_selection_collapses_to_first_query():
    qr = [{"actions": i, "value_prediction": float("nan")} for i in range(8)]
    assert cp.libero_select(qr, 0)[0] == 0
    qr[0]["value_prediction"] = 0.1; qr[5]["value_prediction"] = float("nan")
    assert cp.libero_select(qr, 0)[0] == 0


def test_sb_majority_mean_zeroes_far_successes():
    vs = [np.array([0.99 ** 300])] * 5
    assert cp.aggregate_value_predictions(vs, "majority_mean")[0] == 0.0
    assert fixes.majority_mean_relative(vs, 0.99, 520)[0] > 0


def test_sc_parallel_seed_duplicates_when_queries_exceed_gpus():
    assert len(set(cp.parallel_seeds(0, 8, [0, 1, 2, 3]))) == 4
    assert cp.parallel_seeds(0, 8, list(range(8))) == cp.serial_seeds(0, 8)


def test_sd_fixed_record_counts_everything():
    def fn(g, s, tid):
        if g == 1:
            raise RuntimeError("x")
        return {"v": s}
    rec = fixes.submit_tasks_recorded([0, 1, 2, 3], 4, 1.0, 10, fn)
    assert rec["n_requested"] == 4 and rec["n_returned"] == 3 and rec["n_errors"] == 1 and rec["n_timeout"] == 0
    assert rec["failed_seeds"] == [11] and rec["returned_seeds"] == [10, 12, 13]
    assert rec["elapsed"] < 0.5  # early exit instead of waiting full timeout
