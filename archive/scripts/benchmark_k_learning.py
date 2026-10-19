# 导入标准库与依赖库
import os
import sys
import time
import json
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"]=os.environ.get("CUDA_VISIBLE_DEVICES","2")
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false"

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
sys.path.insert(0,str(Path(__file__).resolve().parent))

import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import OmegaConf
from hydra import compose,initialize
from hydra.core.global_hydra import GlobalHydra

from evorl.algorithms.mo_td3 import PDMORLGPUWorkflow
from train import setup_recorders

# 测试微基准吞吐量（Throughput Benchmark）
def run_throughput_benchmark(k,num_chunks=20):
    chunk_transitions=160*4*4
    prefill_transitions=4160
    total_timesteps=prefill_transitions+chunk_transitions*num_chunks
    output_dir=f"outputs/microbench_k{k}"

    GlobalHydra.instance().clear()
    with initialize(version_base=None,config_path="../configs"):
        cfg=compose(
            config_name="experiment/pd_morl_brax_gpu_native_v2",
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

    os.makedirs(cfg.output_dir,exist_ok=True)
    workflow=PDMORLGPUWorkflow.build_from_config(cfg,enable_jit=cfg.enable_jit)
    recorders=setup_recorders(cfg,workflow.name())
    workflow.add_recorders(recorders)

    state=workflow.init(jax.random.PRNGKey(cfg.seed))

    # 首次 step 触发 JIT 编译
    t0=time.perf_counter()
    _,state=workflow.step(state)
    jax.block_until_ready(state.agent_state.params)
    compile_time=time.perf_counter()-t0

    # 稳态测速循环
    t1=time.perf_counter()
    for _ in range(num_chunks):
        _,state=workflow.step(state)
    jax.block_until_ready(state.agent_state.params)
    steady_time=time.perf_counter()-t1

    workflow.close()

    measured_transitions=chunk_transitions*num_chunks
    measured_critic_updates=cfg.fold_iters*k*num_chunks
    measured_actor_updates=(cfg.fold_iters*k//cfg.actor_update_interval)*num_chunks

    trans_per_s=measured_transitions/steady_time
    critic_per_s=measured_critic_updates/steady_time
    actor_per_s=measured_actor_updates/steady_time
    samples_per_s=critic_per_s*cfg.replay_batch_size
    data_reuse=(k*cfg.replay_batch_size)/(cfg.rollout_length*cfg.num_envs)

    device=jax.devices()[0]
    try:
        mem_info=device.memory_stats()
        peak_mem_mb=mem_info.get("peak_bytes_in_use",0)/(1024*1024)
    except Exception:
        peak_mem_mb=0.0

    return {
        "k":k,
        "compile_time_s":compile_time,
        "steady_time_s":steady_time,
        "transitions_per_s":trans_per_s,
        "critic_updates_per_s":critic_per_s,
        "actor_updates_per_s":actor_per_s,
        "replay_samples_per_s":samples_per_s,
        "data_reuse_intensity":data_reuse,
        "peak_gpu_mem_mb":peak_mem_mb,
    }

# 测试短程学习（200k base transitions Short Learning Benchmark）
def run_short_learning_benchmark(k,budget=201280):
    output_dir=f"outputs/benchmark_k{k}_200k"
    os.makedirs(output_dir,exist_ok=True)

    GlobalHydra.instance().clear()
    with initialize(version_base=None,config_path="../configs"):
        cfg=compose(
            config_name="experiment/pd_morl_brax_gpu_native_v2",
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
                f"total_timesteps={budget}",
                f"output_dir={output_dir}",
                "run_final_evaluation=true",
            ],
        )

    print(f"\n=======================================================")
    print(f"Starting 200k Learning Benchmark for K={k}")
    print(f"Budget: {budget} transitions, Batch: 4096, Envs: 160")
    print(f"=======================================================")

    workflow=PDMORLGPUWorkflow.build_from_config(cfg,enable_jit=cfg.enable_jit)
    recorders=setup_recorders(cfg,workflow.name())
    workflow.add_recorders(recorders)

    t_start=time.perf_counter()
    state=workflow.init(jax.random.PRNGKey(cfg.seed))
    state=workflow.learn(state)
    jax.block_until_ready(state.agent_state.params)
    total_wall_clock=time.perf_counter()-t_start

    workflow.close()

    # 从 results.json 提取双轨评估结果与指标
    res_path=Path(output_dir)/"final_training_eval"/"results.json"
    with open(res_path,"r",encoding="utf-8") as f:
        res=json.load(f)

    pareto_returns=np.asarray(res.get("pareto_returns",[]))
    if len(pareto_returns)>0:
        obj0_min=float(pareto_returns[:,0].min())
        obj0_max=float(pareto_returns[:,0].max())
        obj1_min=float(pareto_returns[:,1].min())
        obj1_max=float(pareto_returns[:,1].max())
    else:
        obj0_min,obj0_max,obj1_min,obj1_max=0.0,0.0,0.0,0.0

    return {
        "k":k,
        "budget_transitions":budget,
        "total_wall_clock_s":total_wall_clock,
        "source_hv":res.get("source_hv",0.0),
        "source_sparsity":res.get("source_sparsity",0.0),
        "source_pareto_point_count":res.get("source_pareto_point_count",0),
        "mean_repeat_hv":res.get("mean_repeat_hv",0.0),
        "mean_repeat_sparsity":res.get("mean_repeat_sparsity",0.0),
        "mean_repeat_pareto_count":res.get("mean_repeat_pareto_count",0.0),
        "repeat_hv":res.get("repeat_hv",[]),
        "repeat_sparsity":res.get("repeat_sparsity",[]),
        "obj0_range":[obj0_min,obj0_max],
        "obj1_range":[obj1_min,obj1_max],
    }

# 主程序调度
def main():
    bench_results=[]
    for k in (5,10,20):
        print(f"\n--- Benchmark K={k} ---")
        micro=run_throughput_benchmark(k,num_chunks=20)
        print(f"Micro-throughput K={k}: {micro['transitions_per_s']:.1f} trans/s, {micro['critic_updates_per_s']:.1f} critic/s")

        learning=run_short_learning_benchmark(k,budget=201280)
        print(f"Learning K={k}: Wall-clock={learning['total_wall_clock_s']:.1f}s, Source HV={learning['source_hv']:.2f}, Pareto Count={learning['source_pareto_point_count']}")

        entry={
            "k":k,
            "microbenchmark":micro,
            "short_learning":learning,
        }
        bench_results.append(entry)

    out_file=Path("docs/benchmark_k_results.json")
    out_file.parent.mkdir(parents=True,exist_ok=True)
    out_file.write_text(json.dumps(bench_results,indent=2),encoding="utf-8")
    print(f"\nSaved all K benchmark results to {out_file}")

if __name__=="__main__":
    main()
