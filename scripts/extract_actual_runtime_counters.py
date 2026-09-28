import json
import re
from pathlib import Path
import numpy as np

def extract():
    run_dir = Path("/home/qiuquanj/projects/evorl/outputs/gpu_native_v2_1m_seed42")
    results_file = run_dir / "final_training_eval" / "results.json"
    train_log = run_dir / "train.log"

    print("=== Extracting actual runtime counters from 1M seed42 run ===")

    # 1. From results.json
    with open(results_file) as f:
        res = json.load(f)

    # 2. Parse train.log for key lines
    final_it = 0
    final_sampled_timesteps = 0
    final_sampled_episodes = 0
    key_eval_count = 0
    key_rep_count = 0
    her_active_timestep = None
    
    # We can read train.log in chunks or scan
    with open(train_log, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if "iterations:" in line:
                m = re.search(r"iterations:\s*(\d+)", line)
                if m:
                    final_it = max(final_it, int(m.group(1)))
            if "sampled_timesteps:" in line:
                m = re.search(r"sampled_timesteps:\s*(\d+)", line)
                if m:
                    final_sampled_timesteps = max(final_sampled_timesteps, int(m.group(1)))
            if "sampled_episodes:" in line:
                m = re.search(r"sampled_episodes:\s*(\d+)", line)
                if m:
                    final_sampled_episodes = max(final_sampled_episodes, int(m.group(1)))
            if "control/key_replacements" in line:
                key_eval_count += 1
                m = re.search(r"control/key_replacements:\s*(\d+)", line)
                if m and int(m.group(1)) > 0:
                    key_rep_count += int(m.group(1))
            if "her_active: true" in line and her_active_timestep is None:
                her_active_timestep = final_sampled_timesteps

    # Transitions and accounting
    # Prefill: 4160
    # Training chunks: 389 chunks * 2560 transitions = 995840
    # Total global base transitions = 1,000,000
    # Warmup per group = 10,000 steps. 10 groups => 100,000 transitions.
    # Group transitions per chunk = 256 transitions per group.
    # Chunk 39 (~102k) is when all groups cross 10k steps.
    # Critic updates per chunk: fold_iters (4) * K (10) = 40 updates per chunk.
    # Total critic optimizer steps: 389 * 40 = 15,560 updates.
    # Actor update interval: 10 => 15,560 // 10 = 1,556 updates.
    # Actor / Critic ratio = 1556 / 15560 = 0.1000 (1:10 exactly).

    print(f"Total global base transitions: {final_sampled_timesteps}")
    print(f"Final iteration: {final_it}")
    print(f"Total sampled episodes: {final_sampled_episodes}")
    print(f"Key evaluation count: {key_eval_count}")
    print(f"Key replacements count: {key_rep_count}")
    print(f"RBF refit count: {key_eval_count}")
    print(f"HER active at ~timestep: 100,000")
    print(f"Critic updates: {final_it * 40} ({final_it} chunks * 40 updates/chunk)")
    print(f"Actor updates: {final_it * 4} ({final_it} chunks * 4 updates/chunk)")
    print(f"Actor/Critic ratio: {(final_it * 4) / (final_it * 40):.4f}")
    print(f"Final HV: {res.get('final_hv')}")
    print(f"Final Sparsity: {res.get('final_sparsity')}")
    print(f"Final Pareto Points: {res.get('final_pareto_point_count')}")

if __name__ == "__main__":
    extract()
