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
from omegaconf import OmegaConf
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra

from evorl.algorithms.mo_td3 import PDMORLGPUWorkflow
from train import setup_recorders

def run_bench(epg: int, k: int = 10, num_chunks: int = 8):
    num_envs = 10 * epg
    chunk_transitions = num_envs * 4 * 4
    total_timesteps = 3200 + chunk_transitions * num_chunks
    output_dir = f"outputs/bench_epg{epg}_k{k}"

    GlobalHydra.instance().clear()
    with initialize(version_base=None, config_path="../configs"):
        cfg = compose(
            config_name="experiment/pd_morl_brax_gpu_native",
            overrides=[
                f"critic_updates_per_rollout={k}",
                "replay_batch_size=4096",
                "batch_size=4096",
                f"envs_per_preference_group={epg}",
                "process_count=10",
                f"num_envs={num_envs}",
                "rollout_length=4",
                "fold_iters=4",
                "seed=42",
                f"total_timesteps={total_timesteps}",
                f"output_dir={output_dir}",
                "run_final_evaluation=false",
            ],
        )

    os.makedirs(cfg.output_dir, exist_ok=True)
    workflow = PDMORLGPUWorkflow.build_from_config(cfg, enable_jit=cfg.enable_jit)
    workflow.add_recorders(setup_recorders(cfg, workflow.name()))
    state = workflow.init(jax.random.PRNGKey(cfg.seed))

    # Warm-up / compile
    _, state = workflow.step(state)
    jax.block_until_ready(state.agent_state.params)

    start_time = time.perf_counter()
    measured_transitions = 0
    measured_critic_updates = 0

    for _ in range(num_chunks):
        _, state = workflow.step(state)
        measured_transitions += chunk_transitions
        measured_critic_updates += cfg.fold_iters * k

    jax.block_until_ready(state.agent_state.params)
    elapsed = time.perf_counter() - start_time
    workflow.close()

    trans_per_s = measured_transitions / elapsed
    critics_per_s = measured_critic_updates / elapsed
    data_reuse = (k * cfg.replay_batch_size) / (cfg.rollout_length * num_envs)

    res = {
        "epg": epg,
        "num_envs": num_envs,
        "k": k,
        "elapsed_s": elapsed,
        "trans_per_s": trans_per_s,
        "critic_updates_per_s": critics_per_s,
        "data_reuse": data_reuse,
    }
    print(f"EPG={epg} (Envs={num_envs}): Elapsed={elapsed:.2f}s, Trans/s={trans_per_s:.1f}, CriticUpdates/s={critics_per_s:.1f}, Reuse={data_reuse:.1f}")
    return res

if __name__ == "__main__":
    results = []
    for epg in (8, 16):
        results.append(run_bench(epg, k=10, num_chunks=8))
    out = Path("docs/benchmark_env_counts_results.json")
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
