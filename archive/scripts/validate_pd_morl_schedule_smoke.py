"""Validate the two schedule-ablation smoke runs."""

import argparse
import json
from pathlib import Path


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def validate(run_dir, *, source_like, diagnostics_dir=None):
    metadata=load_json(run_dir/"run_metadata.json")
    counters=load_json(run_dir/"counters_summary.json")
    diagnostics_dir=diagnostics_dir or run_dir
    diagnostics=[json.loads(line) for line in (diagnostics_dir/"training_diagnostics.jsonl").read_text(encoding="utf-8").splitlines()]
    config=metadata["frozen_hyperparameters"]
    assert metadata["jax_backend"]=="gpu" and metadata["visible_gpu_count"]==1
    assert diagnostics and all(row["finite"] for row in diagnostics)
    her_active=any(row["her_active"] for row in diagnostics) or (
        counters["environment_step_count"]>config["her_start_base_transitions"]
    )
    assert her_active
    assert counters["key_evaluation_count"]>0
    assert counters["rbf_refit_count"]==counters["key_evaluation_count"]
    assert counters["derived_verification"]["matches_critic_optimizer_steps"]
    assert counters["actor_optimizer_step_count"]==counters["target_update_count"]
    if source_like:
        assert config["num_envs"]==10 and config["rollout_length"]==config["fold_iters"]==1
        assert config["critic_updates_per_rollout"]==10
        assert counters["critic_optimizer_step_count"]==10*counters["inner_rollout_count"]
        assert counters["timing"]["transitions_per_host_chunk"]==10
    return {"status":"PASS","run_dir":str(run_dir),"counters":counters}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("source_like",type=Path)
    parser.add_argument("gpu_native",type=Path)
    parser.add_argument("--gpu-native-finite",type=Path)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    result={
        "source_like":validate(args.source_like,source_like=True),
        "gpu_native":validate(
            args.gpu_native,
            source_like=False,
            diagnostics_dir=args.gpu_native_finite,
        ),
    }
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(result,indent=2))


if __name__=="__main__":
    main()
