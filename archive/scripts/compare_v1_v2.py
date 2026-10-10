import json
from pathlib import Path
import numpy as np

def analyze():
    v2_path = Path("/home/qiuquanj/projects/evorl/outputs/gpu_native_v2_1m_seed42/final_training_eval/results.json")
    v1_dir = Path("/home/qiuquanj/projects/evorl/outputs/gpu_native_1m_seed42")

    with open(v2_path) as f:
        v2 = json.load(f)

    print("=== GPU-NATIVE V2 RESULTS (1M SEED42) ===")
    print(f"Final HV: {v2.get('final_hv')}")
    print(f"Final Sparsity: {v2.get('final_sparsity')}")
    print(f"Final Pareto Point Count: {v2.get('final_pareto_point_count')}")
    print(f"Mean Pareto Point Count: {v2.get('mean_pareto_point_count')}")
    print(f"Pareto counts per repeat: {v2.get('pareto_counts_per_repeat')}")
    print(f"HV per repeat: {v2.get('hv_per_repeat')}")
    print(f"Sparsity per repeat: {v2.get('sparsity_per_repeat')}")
    print(f"Objective 1 min/max/mean: {v2.get('obj1_min')} / {v2.get('obj1_max')} / {v2.get('obj1_mean')}")
    print(f"Objective 2 min/max/mean: {v2.get('obj2_min')} / {v2.get('obj2_max')} / {v2.get('obj2_mean')}")

    # Inspect Pareto front
    p_returns = np.array(v2.get("pareto_returns", []))
    print(f"Pareto returns shape: {p_returns.shape}")
    if len(p_returns) > 0:
        print("Pareto points (obj1, obj2):")
        for pt in p_returns:
            print(f"  [{pt[0]:.2f}, {pt[1]:.2f}]")

    print("\n=== CHECKING V1 RESULTS FOR COMPARISON ===")
    v1_eval_file = v1_dir / "final_training_eval" / "results.json"
    if v1_eval_file.exists():
        with open(v1_eval_file) as f:
            v1 = json.load(f)
        print(f"V1 Final HV: {v1.get('final_hv')}")
        print(f"V1 Final Sparsity: {v1.get('final_sparsity')}")
        print(f"V1 Final Pareto Point Count: {v1.get('final_pareto_point_count')}")
    else:
        # Check pd_morl_evaluations.jsonl or run.log in v1
        print("V1 final_training_eval not found, checking jsonl/log...")
        v1_jsonl = v1_dir / "pd_morl_evaluations.jsonl"
        if v1_jsonl.exists():
            lines = v1_jsonl.read_text().strip().split("\n")
            print(f"V1 evaluation records count: {len(lines)}")
            last = json.loads(lines[-1])
            print("V1 last evaluation record:", last)

if __name__ == "__main__":
    analyze()
