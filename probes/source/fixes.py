"""Isolated fixes (probe-side only; reference repo untouched). Each mirrors a port in cosmos_port.py."""
from __future__ import annotations

import queue
import threading
import time
import uuid

import numpy as np

import cosmos_port as cp


def aggregate_value_predictions_fixed(all_value_predictions, scheme):
    """Fix for Sa: LCB with a single ensemble member -> std treated as 0 (== 'average') instead of NaN."""
    if scheme != "lcb":
        return cp.aggregate_value_predictions(all_value_predictions, scheme)
    stacked = np.stack(all_value_predictions)
    mean_value = stacked.mean(0)
    std_value = stacked.std(0, ddof=1) if stacked.shape[0] > 1 else np.zeros_like(mean_value)
    return np.clip(mean_value - 1.0 * std_value, 0, 1)


def majority_mean_relative(all_value_predictions, gamma, horizon_steps):
    """Sb alternative (NOT a drop-in fix): threshold tied to the label scale: half the value of a success that is
    `horizon_steps` away. horizon_steps must come from the task's max episode length, not from test results."""
    if horizon_steps is None:
        horizon_steps = 2000
    thr = 0.5 * gamma ** horizon_steps
    stacked = np.stack(all_value_predictions)
    mask = stacked > thr
    n = mask.sum(0)
    mean_s = np.where(mask, stacked, 0).sum(0) / np.maximum(n, 1)
    return np.clip(np.where(n / stacked.shape[0] >= 0.5, mean_s, 0.0), 0, 1)


def submit_tasks_recorded(available_gpus, N, parallel_timeout, base_seed, task_fn):
    """Sd fix: same dispatch as WorkerPoolManager.submit_tasks, but (1) carry the true seed with each result,
    (2) exit as soon as results+errors == N, (3) return a full record of requested/returned/failed/timed-out."""
    tq = {g: queue.Queue() for g in available_gpus}
    rq, eq = queue.Queue(), queue.Queue()

    def worker(g):
        while True:
            task = tq[g].get()
            if task is None:
                return
            seed = base_seed + g
            try:
                rd = task_fn(g, seed, task["task_id"])
                rq.put({"task_id": task["task_id"], "gpu_id": g, "seed": seed, "return_dict": rd})
            except Exception as e:  # noqa: BLE001
                eq.put({"task_id": task["task_id"], "gpu_id": g, "seed": seed, "error": str(e)})

    ths = [threading.Thread(target=worker, args=(g,), daemon=True) for g in available_gpus]
    for t in ths:
        t.start()
    ids = []
    for q in range(N):
        tid = str(uuid.uuid4())
        tq[available_gpus[q % len(available_gpus)]].put({"task_id": tid})
        ids.append(tid)
    res, err = {}, {}
    t0 = time.time()
    while len(res) + len(err) < N and time.time() - t0 < parallel_timeout:
        try:
            r = rq.get(timeout=0.01)
            res[r["task_id"]] = r
        except queue.Empty:
            pass
        while not eq.empty():
            e = eq.get_nowait()
            err[e["task_id"]] = e
    for g in available_gpus:
        tq[g].put(None)
    returned = [res[t] for t in ids if t in res]
    return dict(n_requested=N, n_returned=len(returned), n_errors=len(err),
                n_timeout=N - len(returned) - len(err), elapsed=round(time.time() - t0, 3),
                returned_seeds=[r["seed"] for r in returned], failed_seeds=[e["seed"] for e in err.values()],
                results=returned)
