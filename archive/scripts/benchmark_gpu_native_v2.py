import os
import sys
import time
import json
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "2")
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import OmegaConf

from evorl.algorithms.mo_td3 import PDMORLGPUWorkflow
from train import setup_recorders


from hydra import compose, initialize

def run_benchmark_for_k(k: int, num_chunks: int = 20) -> dict:
    chunk_transitions = 160 * 4 * 4  # 2560 transitions per chunk
    # Prefill: random_timesteps=1600, learning_start_timesteps=3200
    prefill_transitions = 3200
    total_timesteps = prefill_transitions + chunk_transitions * num_chunks
    output_dir = f"outputs/benchmark_k{k}"

    from hydra.core.global_hydra import GlobalHydra
    GlobalHydra.instance().clear()
    with initialize(version_base=None, config_path="../configs"):
        cfg = compose(
            config_name="experiment/pd_morl_brax_gpu_native",
            overrides=[
                f"critic_updates_per_rollout={k}",
                "replay_batch_size=4096",
                "batch_size=4096",
                "envs_per_preference_group=16",
                "process_count=10",
                "num_envs=160",
                "rollout_length=4",
                "fold_iters=4",
                "seed=42",
                f"total_timesteps={total_timesteps}",
                f"output_dir={output_dir}",
                "run_final_evaluation=false",
            ],
        )

    os.makedirs(cfg.output_dir, exist_ok=True)

    print(f"\n=======================================================")
    print(f"Starting Benchmark for K={k} ({num_chunks} chunks, {chunk_transitions * num_chunks} transitions)")
    print(f"Replay Batch Size: {cfg.replay_batch_size}, Num Envs: {cfg.num_envs}, Rollout Length: {cfg.rollout_length}")
    print(f"=======================================================")

    workflow = PDMORLGPUWorkflow.build_from_config(cfg, enable_jit=cfg.enable_jit)
    recorders = setup_recorders(cfg, workflow.name())
    workflow.add_recorders(recorders)

    state = workflow.init(jax.random.PRNGKey(cfg.seed))

    # Warm-up / compile first step
    _, state = workflow.step(state)
    jax.block_until_ready(state.agent_state.params)

    # Timed benchmark loop
    start_time = time.perf_counter()
    measured_transitions = 0
    measured_critic_updates = 0
    measured_actor_updates = 0

    target_iterations = cfg.total_timesteps // chunk_transitions
    for iteration in range(num_chunks):
        _, state = workflow.step(state)
        measured_transitions += chunk_transitions
        # Each chunk has fold_iters rollouts, each rollout does k critic updates
        chunk_critic_updates = cfg.fold_iters * k
        measured_critic_updates += chunk_critic_updates
        # Actor update interval is 10
        measured_actor_updates += chunk_critic_updates // cfg.actor_update_interval

    jax.block_until_ready(state.agent_state.params)
    elapsed_time = time.perf_counter() - start_time

    workflow.close()

    transitions_per_s = measured_transitions / elapsed_time
    critic_updates_per_s = measured_critic_updates / elapsed_time
    actor_updates_per_s = measured_actor_updates / elapsed_time
    replay_samples_per_s = critic_updates_per_s * cfg.replay_batch_size
    data_reuse = (k * cfg.replay_batch_size) / (cfg.rollout_length * cfg.num_envs)

    # Memory info from JAX devices
    device = jax.devices()[0]
    try:
        mem_info = device.memory_stats()
        peak_mem_mb = mem_info.get("peak_bytes_in_use", 0) / (1024 * 1024)
    except Exception:
        peak_mem_mb = 0.0

    result = {
        "k": k,
        "batch_size": cfg.replay_batch_size,
        "num_envs": cfg.num_envs,
        "rollout_length": cfg.rollout_length,
        "transitions_per_rollout": cfg.num_envs * cfg.rollout_length,
        "data_reuse_intensity": data_reuse,
        "measured_transitions": measured_transitions,
        "measured_critic_updates": measured_critic_updates,
        "elapsed_time_s": elapsed_time,
        "transitions_per_s": transitions_per_s,
        "critic_updates_per_s": critic_updates_per_s,
        "actor_updates_per_s": actor_updates_per_s,
        "replay_samples_per_s": replay_samples_per_s,
        "peak_gpu_mem_mb": peak_mem_mb,
    }

    print(f"Results for K={k}:")
    print(f"  Elapsed: {elapsed_time:.3f} s")
    print(f"  Transitions/s: {transitions_per_s:.1f}")
    print(f"  Critic updates/s: {critic_updates_per_s:.1f}")
    print(f"  Replay samples/s: {replay_samples_per_s:.1f}")
    print(f"  Data reuse intensity: {data_reuse:.1f} samples/transition")
    print(f"  Peak GPU Memory: {peak_mem_mb:.1f} MB")

    return result


def main():
    results = []
    for k in (5, 10, 20):
        res = run_benchmark_for_k(k, num_chunks=8)
        results.append(res)

    out_file = Path("docs/benchmark_gpu_native_v2_results.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved benchmark results to {out_file}")


if __name__ == "__main__":
    main()
