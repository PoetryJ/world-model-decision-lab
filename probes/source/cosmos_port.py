"""Faithful numpy ports of Cosmos Policy functions (NVlabs/cosmos-policy @ 18a2accadf4e7a3531e56754102af5a24d2316da).

Paths are relative to adapters/cosmos/src/cosmos_policy/. Each port cites file:line. Torch semantics that matter:
  * torch.Tensor.std(dim) default is unbiased (correction=1)  -> numpy ddof=1
  * torch.clamp propagates NaN                                -> np.clip propagates NaN
  * torch.stack(list).mean(0)                                 -> np.stack(list).mean(0)
Reference code is NEVER modified; this module only re-implements it.
"""
from __future__ import annotations

import queue
import threading
import time
import uuid
import warnings

import numpy as np

# ---------------------------------------------------------------- datasets/dataset_common.py:25-55
def compute_monte_carlo_returns(num_steps: int, terminal_reward: float, gamma: float) -> np.ndarray:
    T = num_steps
    rewards = np.zeros(T, dtype=np.float32)
    rewards[-1] = terminal_reward
    returns = np.zeros_like(rewards)
    G = 0.0
    for t in reversed(range(T)):
        G = rewards[t] + gamma * G
        returns[t] = G
    if terminal_reward > 0:
        returns = 2 * returns / terminal_reward - 1
    else:
        returns = 2 * returns - 1
    return returns


# ---------------------------------------------------------------- datasets/dataset_common.py:58-95
def get_action_chunk_with_padding(actions, relative_step_idx, chunk_size, num_steps):
    remaining_actions = num_steps - relative_step_idx
    if remaining_actions >= chunk_size:
        return actions[relative_step_idx: relative_step_idx + chunk_size]
    available_actions = actions[relative_step_idx:]
    num_padding_needed = chunk_size - remaining_actions
    padding = np.tile(actions[-1], (num_padding_needed, 1))
    return np.concatenate([available_actions, padding], axis=0)


# ---------------------------------------------------------------- datasets/libero_dataset.py:561-565, 673-713
def libero_sample_indices(relative_step_idx, chunk_size, num_steps):
    """Index bookkeeping of LIBERODataset.__getitem__ (world-model / value sample)."""
    future_frame_idx = relative_step_idx + chunk_size
    max_possible_idx = num_steps - 1
    if future_frame_idx > max_possible_idx:
        future_frame_idx = max_possible_idx
    next_relative_step_idx = min(relative_step_idx + chunk_size, num_steps - 1)
    next_future_frame_idx = min(next_relative_step_idx + chunk_size, max_possible_idx)
    return dict(current=relative_step_idx, future_frame_idx=future_frame_idx,
                return_timestep=future_frame_idx, next_relative_step_idx=next_relative_step_idx,
                next_future_frame_idx=next_future_frame_idx)


# ---------------------------------------------------------------- datasets/dataset_utils.py:179-222 (per-array)
def rescale(arr, mn, mx, non_negative_only=False, scale_multiplier=1.0):
    if not non_negative_only:
        out = 2 * ((arr - mn) / (mx - mn)) - 1
    else:
        out = (arr - mn) / (mx - mn)
    return scale_multiplier * out


# ---------------------------------------------------------------- experiments/robot/cosmos_utils.py:623-648
def unnormalize_actions(actions, actions_min, actions_max, scale_multiplier=1.0):
    original_shape = actions.shape
    a = actions.reshape(-1, actions_min.shape[0])
    a = a / scale_multiplier
    a = 0.5 * (a + 1) * (actions_max - actions_min) + actions_min
    return a.reshape(original_shape)


# ---------------------------------------------------------------- models/policy_text2world_model.py:45-109 / 112-173
def replace_latent_with_action_chunk(x0, action_chunk, action_indices):
    x0 = x0.copy()  # torch version mutates in place (new_x0 = x0); copy keeps the probe side-effect free
    b = np.arange(x0.shape[0])
    tgt = x0[b, :, action_indices, :, :]
    B, C, H, W = tgt.shape
    flat_action = action_chunk.reshape(B, -1)
    n = flat_action.shape[1]
    L = C * H * W
    assert n <= L
    reps = (L + n - 1) // n
    rep = np.tile(flat_action, (1, reps))[:, :L]
    x0[b, :, action_indices, :, :] = rep.reshape(B, C, H, W)
    return x0


def replace_latent_with_proprio(x0, proprio, proprio_indices):
    return replace_latent_with_action_chunk(x0, proprio, proprio_indices)  # identical algorithm (lines 129-171)


# ---------------------------------------------------------------- models/policy_text2world_model.py:~395-405 (value injection)
def replace_latent_with_value(x0, value, value_indices):
    x0 = x0.copy()
    b = np.arange(x0.shape[0])
    _, C, _, H, W = x0.shape
    x0[b, :, value_indices, :, :] = np.broadcast_to(value.reshape(-1, 1, 1, 1), (len(b), C, H, W))
    return x0


# ---------------------------------------------------------------- experiments/robot/cosmos_utils.py:557-599
def extract_action_chunk_from_latent_sequence(output_latent, action_shape, action_indices):
    b = np.arange(output_latent.shape[0])
    fr = output_latent[b, :, action_indices, :, :]
    B = fr.shape[0]
    flat = fr.reshape(B, -1)
    L = flat.shape[1]
    n = action_shape[0] * action_shape[1]
    assert n <= L
    k = L // n
    chunks = flat[:, : k * n].reshape(B, k, action_shape[0], action_shape[1])
    return chunks.mean(axis=1)


# ---------------------------------------------------------------- experiments/robot/cosmos_utils.py:602-620
def extract_value_from_latent_sequence(output_latent, value_indices):
    b = np.arange(output_latent.shape[0])
    fr = output_latent[b, :, value_indices, :, :]
    return fr.reshape(fr.shape[0], -1).mean(axis=1)


# ---------------------------------------------------------------- config/conditioner/video2world_conditioner.py set_video_condition (prefix mask)
# + models/policy_video2world_model.py:127-260 (rollout overrides). Simplified per-sample, frame-level (C,H,W collapsed).
def condition_mask(T, num_conditional_frames, sample_type, action_idx, value_idx,
                   current_idxs=(), mask_current_state_action_for_value_prediction=False):
    """sample_type in {'policy','world_model','value'}; returns (T,) 1=clean conditioning, 0=denoised/generated."""
    m = np.zeros(T, dtype=np.int64)
    m[:num_conditional_frames] = 1
    if sample_type in ("world_model", "value", "policy_rollout"):  # 'rollout_data_mask' in batch
        wm = 1 if sample_type == "world_model" else 0
        vf = 1 if sample_type == "value" else 0
        m[action_idx] = wm                      # line 150: overwritten for every rollout-batch sample
        if vf:
            m[:] = 1                            # line 157
            m[value_idx] = 0                    # line 163
            if mask_current_state_action_for_value_prediction:
                for i in current_idxs:
                    if i != -1:
                        m[i] = 0                # lines 178-257
                m[action_idx] = 0               # line 259
    return m


# ---------------------------------------------------------------- experiments/robot/cosmos_utils.py:766-821
def aggregate_value_predictions(all_value_predictions, scheme):
    stacked = np.stack(all_value_predictions)  # [num_ensemble, batch]
    if scheme == "average":
        v = stacked.mean(0)
    elif scheme == "lcb":
        mean_value = stacked.mean(0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            std_value = stacked.std(0, ddof=1)  # torch .std default unbiased
        beta = 1.0
        v = mean_value - beta * std_value
    elif scheme == "success_vote":
        v = (stacked > 0.05).astype(np.float32).mean(0)
    elif scheme == "majority_mean":
        thr = 0.05
        mask = stacked > thr
        num_success = mask.astype(np.float32).sum(0)
        success_ratio = num_success / stacked.shape[0]
        sum_success = np.where(mask, stacked, 0.0).sum(0)
        mean_success = sum_success / np.maximum(num_success, 1)
        v = np.where(success_ratio >= 0.5, mean_success, 0.0)
    else:
        raise ValueError(f"Invalid search value scheme: {scheme}")
    return np.clip(v, 0, 1)


# ---------------------------------------------------------------- cosmos_utils.py:1482-1484 / 1616-1618
def value_ensemble_steps(num_denoising_steps_value, n):
    return np.linspace(1, num_denoising_steps_value, n).astype(int).tolist()


# ---------------------------------------------------------------- seeds
def serial_seeds(base_seed, N):
    """libero/run_libero_eval.py:455, robocasa/run_robocasa_eval.py:452, aloha/deploy.py:375 -> seed=cfg.seed+query_idx"""
    return [base_seed + q for q in range(N)]


def parallel_seeds(base_seed, N, available_gpus, truncate_to_N=True):
    """libero/run_libero_eval.py:830 (truncate available_gpus[:N]); cosmos_utils.py:1777 gpu=gpus[q % len];
    worker uses seed=cfg.seed+gpu_id (cosmos_utils.py:1920,1952,1977,1997,2105,2124). aloha/deploy.py:314 does NOT truncate."""
    gpus = available_gpus[:N] if truncate_to_N else list(available_gpus)
    return [base_seed + gpus[q % len(gpus)] for q in range(N)]


def randomized_seed(rng):
    """cosmos_utils.py:885 / 1287 / 1462: secrets.randbits(32) % 256 (port uses an explicit rng for repeatability)."""
    return int(rng.integers(0, 2**32)) % 256


# ---------------------------------------------------------------- selection
def libero_select(query_results, base_seed):
    """libero/run_libero_eval.py:579-591: labels by POSITION, python max over value."""
    seed_to = {base_seed + qi: (rd["actions"], rd["value_prediction"]) for qi, rd in enumerate(query_results)}
    best_seed, best = max(seed_to.items(), key=lambda x: x[1][1])
    return best_seed, best


def aloha_select(query_results, base_seed):
    """aloha/deploy.py:603-656: base seed kept unless strictly better."""
    if not query_results:
        raise RuntimeError("No query results returned from model inference")
    seed_to = {base_seed + qi: (rd.get("actions"), float(rd.get("value_prediction", 0.0))) for qi, rd in enumerate(query_results)}
    base_a, base_v = seed_to[base_seed]
    best_seed, best_v, best_a = base_seed, base_v, base_a
    hs, (ha, hv) = max(seed_to.items(), key=lambda x: x[1][1])
    if hv > base_v:
        best_seed, best_v, best_a = hs, hv, ha
    return best_seed, best_a, best_v


# ---------------------------------------------------------------- WorkerPoolManager.submit_tasks (cosmos_utils.py:1765-1819)
class FakePool:
    """Threads stand in for GPU processes; queue logic copied from submit_tasks / persistent_parallel_worker (1893-2170)."""

    def __init__(self, available_gpus, num_queries, parallel_timeout, base_seed, task_fn):
        self.available_gpus = available_gpus
        self.N = num_queries
        self.timeout = parallel_timeout
        self.base_seed = base_seed
        self.task_fn = task_fn  # task_fn(gpu_id, seed, task_id) -> return_dict or raise
        self.task_queues = {g: queue.Queue() for g in available_gpus}
        self.result_queue = queue.Queue()
        self.error_queue = queue.Queue()
        self.stop = threading.Event()
        self.threads = [threading.Thread(target=self._worker, args=(g,), daemon=True) for g in available_gpus]
        for t in self.threads:
            t.start()

    def _worker(self, gpu_id):
        while not self.stop.is_set():
            try:
                task = self.task_queues[gpu_id].get(timeout=0.01)
                if task is None:
                    break
                rd = self.task_fn(gpu_id, self.base_seed + gpu_id, task["task_id"])
                self.result_queue.put({"gpu_id": gpu_id, "task_id": task["task_id"], "type": "result", "return_dict": rd})
            except Exception as e:
                if not isinstance(e, queue.Empty):
                    self.error_queue.put({"gpu_id": gpu_id, "task_id": task.get("task_id", "unknown"),
                                          "type": "task_error", "error": str(e)})

    def submit_tasks(self):
        task_ids = []
        for qi in range(self.N):
            tid = str(uuid.uuid4())
            g = self.available_gpus[qi % len(self.available_gpus)]
            self.task_queues[g].put({"task_id": tid})
            task_ids.append(tid)
        results, errors = {}, []
        t0 = time.time()
        while len(results) < self.N and (time.time() - t0) < self.timeout:
            while not self.result_queue.empty():
                r = self.result_queue.get_nowait()
                if r.get("type") == "result" and r.get("task_id") in task_ids:
                    results[r["task_id"]] = r
            while not self.error_queue.empty():
                e = self.error_queue.get_nowait()
                if e.get("type") == "task_error" and e.get("task_id") in task_ids:
                    errors.append(e)
            time.sleep(0.01)
        out = [results[t]["return_dict"] for t in task_ids if t in results]
        self.last = dict(elapsed=time.time() - t0, n_errors=len(errors), n_results=len(out))
        return out

    def shutdown(self):
        self.stop.set()
        for q in self.task_queues.values():
            q.put(None)


def query_model_parallel(pool):
    """cosmos_utils.py:2217-2230: returns None (implicit) if no results."""
    query_results = pool.submit_tasks()
    if query_results:
        return query_results
